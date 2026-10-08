import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class DistanceConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.message_mlp = nn.Sequential(nn.Linear(in_channels + 1, out_channels), nn.ReLU(),
                                         nn.Linear(out_channels, out_channels))
        self.update_mlp = nn.Sequential(nn.Linear(in_channels + out_channels, out_channels), nn.ReLU(),
                                        nn.Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_dist):
        return self.propagate(edge_index, x=x, edge_dist=edge_dist)

    def message(self, x_j, edge_dist):
        return self.message_mlp(torch.cat([x_j, edge_dist], dim=1))

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=1))


class HaloEncoder(nn.Module):
    def __init__(self, hidden=32, in_channels=1, distance_scale=30.0):
        super().__init__()
        self.distance_scale = distance_scale
        self.conv1 = DistanceConv(in_channels, hidden)
        self.conv2 = DistanceConv(hidden, hidden)

    def forward(self, data):
        distance = data.edge_attr / self.distance_scale
        h = F.relu(self.conv1(data.x, data.edge_index, distance))
        return self.conv2(h, data.edge_index, distance)


class QueryAggregator(nn.Module):
    def __init__(self, hidden=32, distance_scale=30.0, embedding_only=False):
        super().__init__()
        self.distance_scale = distance_scale
        self.message_mlp = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        if embedding_only:
            self.predictor = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        else:
            self.predictor = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(),
                                           nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.embedding_only = embedding_only

    def forward(self, halo_embed, links):
        distance = (links.distance / self.distance_scale).unsqueeze(1)
        messages = self.message_mlp(torch.cat([halo_embed[links.halo_index], distance], dim=1))
        embedding = scatter(messages, links.query_index, dim=0,
                            dim_size=links.n_queries, reduce='mean')
        counts = scatter(torch.ones_like(links.distance), links.query_index,
                         dim=0, dim_size=links.n_queries, reduce='sum')
        result = self.predictor(torch.cat([embedding, torch.log1p(counts)[:, None]], dim=1))
        return result if self.embedding_only else result.squeeze(1)


class HaloToDM(nn.Module):
    def __init__(self, hidden=32, in_channels=1, distance_scale=30.0):
        super().__init__()
        self.config = dict(hidden=hidden, in_channels=in_channels, distance_scale=distance_scale)
        self.encoder = HaloEncoder(**self.config)
        self.query = QueryAggregator(hidden, distance_scale)

    def forward(self, dataset):
        return self.query(self.encoder(dataset.halo), dataset.links)


class HaloRandomToDM(nn.Module):
    def __init__(self, hidden=64, in_channels=1, distance_scale=30.0):
        super().__init__()
        self.config = dict(hidden=hidden, in_channels=in_channels, distance_scale=distance_scale)
        self.halo_encoder = HaloEncoder(**self.config)
        self.halo_query = QueryAggregator(hidden, distance_scale, embedding_only=True)
        self.random_encoder = HaloEncoder(hidden, 1, distance_scale)
        self.predictor = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                       nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, dataset):
        halos = self.halo_query(self.halo_encoder(dataset.halo), dataset.links)
        randoms = self.random_encoder(dataset.random)
        return self.predictor(torch.cat([halos, randoms], dim=1)).squeeze(1)