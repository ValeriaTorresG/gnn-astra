import argparse
import json

from . import (GNNTrainer, HaloRandomToDM, HaloToDM, PeriodicGraphBuilder,
               QuijoteReader, SpatialSplitter, TrainingConfig, seed_everything)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', default='data/graph_fid_0.pkl.gz')
    parser.add_argument('--targets', default='data/delta_dm_randoms_R10.pkl')
    parser.add_argument('--output', default='outputs/gnn.pt')
    parser.add_argument('--box-size', type=float, default=1000.0)
    parser.add_argument('--neighbors', type=int, default=8)
    parser.add_argument('--radius', type=float, default=30.0)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=501)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--device')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--with-randoms', action='store_true')
    parser.add_argument('--use-probabilities', action='store_true')
    args = parser.parse_args()

    seed_everything(args.seed)
    catalog = QuijoteReader(args.box_size).from_cache(args.graph, args.targets, use_probabilities=args.use_probabilities)
    dataset = PeriodicGraphBuilder(args.box_size, args.neighbors, args.radius).build(catalog, include_random_graph=args.with_randoms)
    split = SpatialSplitter(args.box_size).split(dataset.query_positions)

    model_class = HaloRandomToDM if args.with_randoms else HaloToDM
    model = model_class(args.hidden, dataset.halo.x.shape[1], args.radius)

    trainer = GNNTrainer(model, TrainingConfig(epochs=args.epochs, patience=args.patience, device=args.device, seed=args.seed))
    trainer.fit(dataset, split)

    print(json.dumps(trainer.evaluate(dataset, split.test), indent=2))
    trainer.save(args.output)


if __name__ == '__main__':
    main()