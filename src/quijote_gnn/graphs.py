from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data

from .data import periodic_positions


@dataclass
class QueryLinks:
    halo_index: torch.Tensor
    query_index: torch.Tensor
    distance: torch.Tensor
    n_queries: int

    def to(self, device):
        return QueryLinks(self.halo_index.to(device), self.query_index.to(device),
                          self.distance.to(device), self.n_queries)


@dataclass
class GraphDataset:
    halo: Data
    query_positions: torch.Tensor
    links: QueryLinks
    targets: torch.Tensor | None = None
    random: Data | None = None

    def to(self, device):
        return GraphDataset(self.halo.clone().to(device), self.query_positions.to(device),
                            self.links.to(device),
                            None if self.targets is None else self.targets.to(device),
                            None if self.random is None else self.random.clone().to(device))


class PeriodicGraphBuilder:
    def __init__(self, box_size=1000.0, k_neighbors=8, query_radius=30.0):
        periodic_positions(np.empty((0, 3)), box_size)
        self.box_size = float(box_size)
        self.k_neighbors = k_neighbors
        self.query_radius = float(query_radius)


    def build_graph(self, positions, features=None):
        positions = periodic_positions(positions, self.box_size)
        n = len(positions)
        k = min(self.k_neighbors, n - 1)

        pairs = set()
        if k:
            _, indices = cKDTree(positions, boxsize=self.box_size).query(positions, k=k + 1)
            for i, neighbors in enumerate(indices):
                for j in neighbors[neighbors != i][:k]:
                    pairs.add((min(i, int(j)), max(i, int(j))))

        edges = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
        delta = positions[edges[:, 0]] - positions[edges[:, 1]]
        delta -= self.box_size * np.round(delta / self.box_size)

        distances = np.linalg.norm(delta, axis=1).astype(np.float32)
        edge_index = np.concatenate([edges.T, edges[:, ::-1].T], axis=1)
        x = np.ones((n, 1), dtype=np.float32) if features is None else np.asarray(features, dtype=np.float32)

        return Data(x=torch.from_numpy(x), pos=torch.tensor(positions, dtype=torch.float32),
                    edge_index=torch.from_numpy(edge_index),
                    edge_attr=torch.from_numpy(np.tile(distances, 2)[:, None]))


    def build_links(self, halo_positions, query_positions):
        halos = periodic_positions(halo_positions, self.box_size)
        queries = periodic_positions(query_positions, self.box_size)

        neighbors = cKDTree(halos, boxsize=self.box_size).query_ball_point(queries, r=self.query_radius)
        counts = np.array([len(row) for row in neighbors], dtype=np.int64)

        query_index = np.repeat(np.arange(len(queries), dtype=np.int64), counts)
        halo_index = np.array([i for row in neighbors for i in row], dtype=np.int64)

        delta = halos[halo_index] - queries[query_index]
        delta -= self.box_size * np.round(delta / self.box_size)
        distance = np.linalg.norm(delta, axis=1).astype(np.float32)
        return QueryLinks(torch.from_numpy(halo_index), torch.from_numpy(query_index),
                          torch.from_numpy(distance), len(queries))


    def build(self, catalog, include_random_graph=False):
        return GraphDataset(halo=self.build_graph(catalog.halo_positions, catalog.halo_features),
                            query_positions=torch.tensor(catalog.query_positions, dtype=torch.float32),
                            links=self.build_links(catalog.halo_positions, catalog.query_positions),
                            targets=None if catalog.targets is None else torch.tensor(catalog.targets),
                            random=(self.build_graph(catalog.query_positions) if include_random_graph else None))