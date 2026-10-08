import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .models import HaloToDM, HaloRandomToDM


@dataclass
class DatasetSplit:
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor

    def validate(self, n_queries):
        groups = [self.train, self.validation, self.test]
        joined = torch.cat([group.cpu() for group in groups])
        return self

    def to(self, device):
        return DatasetSplit(self.train.to(device), self.validation.to(device), self.test.to(device))


class SpatialSplitter:
    def __init__(self, box_size=1000.0, axis=0):
        self.box_size = box_size
        self.axis = axis

    def split(self, positions):
        positions = torch.as_tensor(positions)
        x = (positions[:, self.axis] % self.box_size) / self.box_size
        masks = [((x < .44) | (x >= .86)),
                 ((x >= .50) & (x < .60)), ((x >= .70) & (x < .80))]
        return DatasetSplit(*(torch.where(mask)[0].cpu() for mask in masks)).validate(len(x))


@dataclass
class TrainingConfig:
    epochs: int = 501
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int | None = None
    seed: int = 42
    device: str | None = None
    log_every: int = 25

    def __post_init__(self):
        if self.epochs < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError('epochs y learning_rate must be positive; weight_decay must be non-negative.')
        if self.patience is not None and self.patience < 1:
            raise ValueError('patience must be positive or None.')


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    mse = np.mean((target - prediction) ** 2)
    variance = np.sum((target - target.mean()) ** 2)
    r2 = (1.0 - np.sum((target - prediction) ** 2) / variance) if variance > 0 else float('nan')
    pearson = (float(np.corrcoef(target, prediction)[0, 1])
               if target.std() > 0 and prediction.std() > 0 else float('nan'))
    return {'rmse': float(np.sqrt(mse)), 'r2': float(r2), 'pearson_r': pearson}


class GNNTrainer:
    def __init__(self, model, config=None):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device)
        self.history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': []}
        self.target_mean = None
        self.target_std = None
        self.best_epoch = None


    def fit(self, dataset, split):
        split.validate(dataset.links.n_queries)
        seed_everything(self.config.seed)
        data = dataset.to(self.device)
        split = split.to(self.device)
        y = data.targets

        self.target_mean = y[split.train].mean().detach()
        self.target_std = y[split.train].std().detach()

        normalized = (y - self.target_mean) / self.target_std
        self.history = {key: [] for key in self.history}
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                     weight_decay=self.config.weight_decay)
        best_loss, best_state, stale = float('inf'), None, 0

        for epoch in range(self.config.epochs):
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(data)
            train_loss = F.mse_loss(pred[split.train], normalized[split.train])

            train_loss.backward()
            optimizer.step()
            self.model.eval()
            with torch.no_grad():
                pred = self.model(data)
                val_loss = F.mse_loss(pred[split.validation], normalized[split.validation]).item()
                pred_dm = pred * self.target_std + self.target_mean

            self.history['train_loss'].append(train_loss.item())
            self.history['val_loss'].append(val_loss)
            for key, indices in (('train_r2', split.train), ('val_r2', split.validation)):
                metrics = regression_metrics(y[indices].cpu().numpy(), pred_dm[indices].cpu().numpy())
                self.history[key].append(metrics['r2'])

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                stale = 0
            else:
                stale += 1

            if self.config.log_every and epoch % self.config.log_every == 0:
                print(f'{epoch:4d} train_loss={train_loss.item():.5f} val_loss={val_loss:.5f}')
            if self.config.patience is not None and stale >= self.config.patience:
                break

        self.model.load_state_dict(best_state)
        self.model.eval()
        return self.history

    @torch.no_grad()
    def predict(self, dataset):
        self.model.eval()
        prediction = self.model(dataset.to(self.device))
        return (prediction * self.target_std + self.target_mean).cpu()


    def evaluate(self, dataset, indices=None):
        pred = self.predict(dataset)
        target = dataset.targets.detach().cpu()
        if indices is not None:
            indices = torch.as_tensor(indices, dtype=torch.long).cpu()
            pred, target = pred[indices], target[indices]
        return regression_metrics(target.numpy(), pred.numpy())


    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_class': type(self.model).__name__, 'model_config': self.model.config,
                    'model_state': {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                    'training_config': asdict(self.config), 'target_mean': self.target_mean.cpu(),
                    'target_std': self.target_std.cpu(), 'history': self.history, 'best_epoch': self.best_epoch}, path)


    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        models = {'HaloToDM': HaloToDM, 'HaloRandomToDM': HaloRandomToDM}
        model = models[checkpoint['model_class']](**checkpoint['model_config'])
        config = TrainingConfig(**{**checkpoint['training_config'], 'device': device})

        trainer = cls(model, config)
        trainer.model.load_state_dict(checkpoint['model_state'])

        trainer.target_mean = checkpoint['target_mean'].to(trainer.device)
        trainer.target_std = checkpoint['target_std'].to(trainer.device)
        trainer.history = checkpoint['history']
        trainer.best_epoch = checkpoint['best_epoch']

        trainer.model.eval()
        return trainer