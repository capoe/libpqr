import os
import re
import csv
import gzip
import multiprocessing as mp
import functools

import numpy as np
import torch
import torch.nn.functional as nnf

from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import (
    InMemoryDataset, 
    Dataset,
    Data
)

from .aux import Timer


class DatasetPartition:
    def __init__(self, 
            datalist, 
            follow_batch=[], 
            exclude_keys=[],
            batchsize=1,
            shuffle_iter=True,
            collate=True
    ):
        self.dataclass = datalist[0].__class__
        self.do_collate = collate
        if self.do_collate:
            self.data, self.slices, _ = collate(
                self.dataclass,
                data_list=datalist,
                increment=False,
                add_batch=False,
                follow_batch=follow_batch,
                exclude_keys=exclude_keys
            )
        else:
            self.data = datalist
        self.collater = Collater(
            follow_batch=follow_batch, 
            exclude_keys=exclude_keys
        )
        self.batchsize = batchsize
        self.shuffle_iter = shuffle_iter

    def __len__(self):
        if self.do_collate:
            raise NotImplementedError("Determine __len__ manually from data")
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.slice(idx)
        else:
            return self.get(idx)

    def get(self, idx):
        if self.do_collate:
            data = separate(
                cls=self.dataclass,
                batch=self.data,
                idx=idx,
                slice_dict=self.slices,
                decrement=False
            )
        else:
            data = self.data[idx] 
        return data

    def slice(self, idcs):
        datalist = [ self.get(_) for _ in idcs ]
        return self.collater(datalist)

    def __iter__(self):
        n_graphs = len(self)
        n_batches = n_graphs // self.batchsize
        batchsize = int(n_graphs / n_batches)
        n_rest = n_graphs % batchsize
        if self.shuffle_iter:
            order = np.random.permutation(n_graphs)
        else:
            order = np.arange(n_graphs)
        i = 0
        for b in range(n_batches):
            j = i + batchsize
            if b < n_rest: j += 1
            sec = self.slice(order[i:j])
            yield sec
            i = j


class DatasetPartitionSampler:
    def __init__(self, files, batchsize, shuffle=True):
        self.partition_files = files
        self.batchsize = batchsize
        self.timer = None
        if shuffle:
            self.order = np.random.permutation(len(self.partition_files))
            self.partition_files = [ self.partition_files[_] for _ in self.order ]
        else:
            self.order = None
    def __iter__(self):
        timer = Timer()
        for pidx, p in enumerate(self.partition_files):
            with timer.time("load_%03d" % pidx):
                part = torch.load(p)
            n_graphs = len(part.data.path)
            print("Partition:", p, "  n_graphs=", n_graphs)
            n_batches = n_graphs // self.batchsize
            n_rest = n_graphs % self.batchsize
            idcs = np.arange(0, n_graphs)
            order = np.random.permutation(len(idcs))
            i = 0
            for b in range(n_batches):
                j = i + self.batchsize
                if b < n_rest: j += 1
                with timer.time("slice_%03d" % pidx):
                    sec = part.slice(order[i:j])
                yield sec
                i = j
        self.timer = timer


class MolecularSmilesDataset(InMemoryDataset):
    def __init__(self, 
            datafile,
            smiles,
            targets,
            convert_target=lambda t: float(t),
            root="./data", 
            transform=None, 
            pre_transform=None, 
            pre_filter=None):
        self.datafile = datafile
        self.name, _ = os.path.splitext(os.path.basename(self.datafile))
        self.smiles = smiles
        self.targets = targets
        self.convert_target = convert_target
        super(MolecularSmilesDataset, self).__init__(
            root=root, 
            transform=transform, 
            pre_transform=pre_transform, 
            pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        assert False
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        if self.datafile.endswith('csv'):
            import csv
            reader = csv.DictReader(open(os.path.join(self.root, self.datafile)))
        elif self.datafile.endswith('json'):
            import json
            reader = json.load(open(self.datafile))
        else: raise IOError("Unknown file type: '%s'" % self.datafile)
        datalist = []
        for row in reader:
            smiles = row[self.smiles]
            ys = [self.convert_target(row[t]) for t in self.targets]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)
            data = Data(smiles=smiles, y=y)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            datalist.append(data)
        torch.save(self.collate(datalist), self.processed_paths[0])


class VolatileMolecularSmilesDataset(object):
    def __init__(
            self, 
            root='data', 
            target_key='y',
            target_conv=float,
            truncate=-1,
            csv_files=[],
            pre_filter=None,
            pre_transform=None,
            post_transform=None,
            transform_chunksize=50000,
            n_procs=1):
        self.csv_files = csv_files
        self.truncate = truncate
        self.target_key = target_key
        self.target_conv = target_conv
        self.transform_chunksize = transform_chunksize
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.n_procs = n_procs
        self.current = -1
        self.raw_paths = [ os.path.join(root, 'raw', _) for _ in csv_files ]

    def next(self):
        self.current += 1
        if self.current >= len(self.raw_paths): 
            return None
        else:
            return self.process_path(self.raw_paths[self.current])

    def process_path(self, path):
        print("Reading", path)
        if path.endswith('.gz'):
            reader = csv.DictReader(gzip.open(path, 'rt'))
        else:
            reader = csv.DictReader(open(path))
        print("Collecting data rows", path)
        rows = [ (ridx, row) for ridx, row in enumerate(reader) ]
        if self.truncate > 0 and len(rows) > self.truncate: 
            rows = rows[0:self.truncate]
        print("Pooling data transform", path)
        pool_func = functools.partial(process_row, 
            filter_func=self.pre_filter,
            data_trafo=self.pre_transform,
            data_post_trafo=self.post_transform,
            target_key=self.target_key,
            target_trafo=self.target_conv)
        chunksize = self.transform_chunksize
        n_chunks = len(rows) // chunksize + (1 if (len(rows) % chunksize) > 0 else 0)
        datalist = []
        for chunk in range(n_chunks):
            i = chunk*chunksize
            j = i + chunksize
            print(" Mapping chunk %6d:%-6d" % (i, j), flush=True)
            if self.n_procs > 1:
                with mp.Pool(self.n_procs) as pool:
                    datalist_chunk = pool.map(pool_func, rows[i:j])
                    datalist.extend(datalist_chunk)
            else:
                datalist_chunk = map(pool_func, rows[i:j])
                datalist.extend(datalist_chunk)
        datalist = list(filter(lambda d: d is not None, datalist))
        datalist = datalist_to_torch(datalist)
        return datalist


class BatchMolecularSmilesDataset(InMemoryDataset):
    def __init__(
            self, 
            root='data', 
            target_key='y',
            target_conv=float,
            truncate=-1,
            csv_files=[],
            transform=None, 
            pre_transform=None,
            pre_filter=None,
            transform_chunksize=50000,
            n_procs=1):
        self.csv_files = csv_files
        self.processed = []
        self.truncate = truncate
        self.target_key = target_key
        self.target_conv = target_conv
        self.transform_chunksize = transform_chunksize
        self.n_procs = n_procs
        InMemoryDataset.__init__(self, 
            root=root, 
            transform=transform, 
            pre_transform=pre_transform,
            pre_filter=pre_filter)
        self.current = -1
        self.next()

    def next(self):
        self.current += 1
        load_path = self.processed_paths[self.current % len(self.processed_paths)]
        self.data, self.slices = torch.load(load_path)

    @property
    def raw_file_names(self):
        return self.csv_files

    @property
    def processed_file_names(self):
        return [ "%s_proc.pt" % c for c in self.csv_files ]

    def get(self, idx):
        return InMemoryDataset.get(self, idx)

    def process_path(self, path):
        print("Reading", path)
        if path.endswith('.gz'):
            reader = csv.DictReader(gzip.open(path, 'rt'))
        else:
            reader = csv.DictReader(open(path))
        print("Collecting data rows", path)
        rows = [ (ridx, row) for ridx, row in enumerate(reader) ]
        if self.truncate > 0 and len(rows) > self.truncate: 
            rows = rows[0:self.truncate]
        print("Pooling data transform", path)
        pool_func = functools.partial(process_row, 
            filter_func=self.pre_filter,
            data_trafo=self.pre_transform,
            target_key=self.target_key,
            target_trafo=self.target_conv)
        chunksize = self.transform_chunksize
        n_chunks = len(rows) // chunksize + (1 if (len(rows) % chunksize) > 0 else 0)
        datalist = []
        for chunk in range(n_chunks):
            i = chunk*chunksize
            j = i + chunksize
            print(" Mapping chunk %6d:%-6d" % (i, j))
            with mp.Pool(self.n_procs) as pool:
                datalist_chunk = pool.map(pool_func, rows[i:j])
                datalist.extend(datalist_chunk)
        datalist = list(filter(lambda d: d is not None, datalist))
        datalist = datalist_to_torch(datalist)
        print("Collating and saving to disk ...")
        out = os.path.join(
            self.processed_dir, 
            '%s_proc.pt' % os.path.basename(path))
        torch.save(self.collate(datalist), out)

    def process(self):
        for path in self.raw_paths:
            self.process_path(path)


def datalist_to_torch(datalist):
    for data in datalist:
        data.x = torch.from_numpy(data.x)
        data.y = torch.from_numpy(data.y)
        data.edge_attr = torch.from_numpy(data.edge_attr)
        data.edge_index = torch.from_numpy(data.edge_index)
        if hasattr(data, "edge_index_full"):
            data.edge_index_full = torch.from_numpy(data.edge_index_full)
        if hasattr(data, "edge_attr_full"):
            data.edge_attr_full = torch.from_numpy(data.edge_attr_full)
        if hasattr(data, "edge_dist_full"):
            data.edge_dist_full = torch.from_numpy(data.edge_dist_full)
        if hasattr(data, "w"):
            data.w = torch.from_numpy(data.w)
        if hasattr(data, "h"):
            data.h = torch.from_numpy(data.h)
    return datalist


def data_to_numpy(data):
    data.x = data.x.numpy()
    data.y = data.y.numpy()
    data.edge_index = data.edge_index.numpy()
    data.edge_attr = data.edge_attr.numpy()
    if hasattr(data, "edge_index_full"):
        data.edge_index_full = data.edge_index_full.numpy()
    if hasattr(data, "edge_attr_full"):
        data.edge_attr_full = data.edge_attr_full.numpy()
    if hasattr(data, "edge_dist_full"):
        data.edge_dist_full = data.edge_dist_full.numpy()
    if hasattr(data, "w"):
        data.w = data.w.numpy()
    if hasattr(data, "h"):
        data.h = data.h.numpy()
    return data


def process_row(
        row_data, 
        filter_func,
        data_trafo, 
        data_post_trafo,
        target_key, 
        target_trafo,
        verbose=False):
    with torch.no_grad():
        (ridx, row) = row_data
        if verbose and ridx % 1000 == 0: 
            print(" Transforming row %-8d" % ridx)
        if target_key is not None:
            y = target_trafo(row[target_key])
            if isinstance(y, float):
                y = torch.tensor(y).reshape((-1,1))
            else:
                pass
        else:
            y = torch.tensor([-1.])
        data = Data(smiles=row["smiles"], y=y)
        if filter_func is not None and not filter_func(data):
            return None
        if data_trafo is not None:
            data = data_trafo(data)
        if data_post_trafo is not None:
            data = data_post_trafo(data)
        return data_to_numpy(data) # <- torch tensors result in odd ulimit issues


