from dataclasses import dataclass
from pathlib import Path
import gzip
import pickle
import numpy as np


PROBABILITY_COLUMNS = ('PVOID', 'PSHEET', 'PFILAMENT', 'PKNOT')


def load_pickle(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        return pickle.load(stream)


def save_pickle(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wb') as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def periodic_positions(value, box_size):
    value = np.asarray(value, dtype=np.float64)
    return np.mod(value, box_size)


@dataclass
class QuijoteCatalog:
    halo_positions: np.ndarray
    query_positions: np.ndarray
    box_size: float = 1000.0
    targets: np.ndarray | None = None
    halo_features: np.ndarray | None = None

    def __post_init__(self):
        self.halo_positions = periodic_positions(self.halo_positions, self.box_size)
        self.query_positions = periodic_positions(self.query_positions, self.box_size)
        if self.targets is not None:
            self.targets = np.asarray(self.targets, dtype=np.float32)
        if self.halo_features is not None:
            self.halo_features = np.asarray(self.halo_features, dtype=np.float32)


class QuijoteReader:

    def __init__(self, box_size=1000.0):
        periodic_positions(np.empty((0, 3)), box_size)
        self.box_size = float(box_size)

    def read_halos(self, directory, snapshot_number=3):
        import readfof
        fof = readfof.FoF_catalog(str(directory), snapshot_number, long_ids=False,
                                  swap=False, SFR=False, read_IDs=False)
        return periodic_positions(fof.GroupPos / 1e3, self.box_size)

    def read_particles(self, snapshot, particle_types=(1,)):
        import readgadget
        positions = readgadget.read_block(str(snapshot), 'POS ', list(particle_types))
        return periodic_positions(positions / 1e3, self.box_size)

    def read_randoms(self, path, iteration=0):
        from astropy.table import Table
        table = Table.read(path)
        table = table[np.asarray(table['RANDITER']) == iteration]
        positions = np.column_stack([table[name] for name in ('X', 'Y', 'Z')])
        return periodic_positions(positions, self.box_size)

    def read_probabilities(self, path):
        from astropy.table import Table
        table = Table.read(path)
        return np.column_stack([table[name] for name in PROBABILITY_COLUMNS]).astype(np.float32)

    def from_files(self, halo_directory, randoms_path, snapshot_number=3,
                   iteration=0, targets_path=None, probabilities_path=None):
        return QuijoteCatalog(halo_positions=self.read_halos(halo_directory, snapshot_number),
                              query_positions=self.read_randoms(randoms_path, iteration),
                              box_size=self.box_size,
                              targets=None if targets_path is None else load_pickle(targets_path),
                              halo_features=(None if probabilities_path is None else self.read_probabilities(probabilities_path)))


    def from_cache(self, graph_path, targets_path=None, use_probabilities=False):
        graph = load_pickle(graph_path)
        halos = [n for n, a in graph.nodes(data=True) if a.get('node_type') == 'halo']
        queries = [n for n, a in graph.nodes(data=True) if a.get('node_type') == 'random']
        features = None
        if use_probabilities:
            features = [[graph.nodes[n][c] for c in PROBABILITY_COLUMNS] for n in halos]
        return QuijoteCatalog(halo_positions=np.array([graph.nodes[n]['pos'] for n in halos]),
                              query_positions=np.array([graph.nodes[n]['pos'] for n in queries]),
                              box_size=self.box_size,
                              targets=None if targets_path is None else load_pickle(targets_path),
                              halo_features=features)