from torch_geometric.data import Dataset, InMemoryDataset
import numpy as np
import lmdb
import os
from tqdm import tqdm
import pickle5 as pickle
import torch

from multiprocessing import Pool
from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector)
from rdkit import Chem
import pandas as pd
from data.wrapper import smiles2graph, smiles2graph_cut
import copy
from functools import lru_cache
from torch_geometric.data import Data
from fairseq.data import data_utils


def load_lmdb(lmdb_path, test_or_valid=False):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    smiles_lst = []
    target_lst = []
    coordinates_lst = []
    atoms_lst = []
    for idx in tqdm(keys):
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        smiles_lst.append(str(data['smi']))
        target_lst.append(np.array(data['target']))
        if test_or_valid:
            coordinates_lst.append(np.array([data['coordinates'][0]]))
        else:
            coordinates_lst.append(np.array(data['coordinates']))
        atoms_lst.append(data['atoms'])
    

    return [smiles_lst, np.array(target_lst), coordinates_lst, atoms_lst]


# Molenet dataset class
# dataset_name: cliontox, sider, bace
# data_type: {train, valid, test}

MOLNET_DATASET = ['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace', 'freesolv', 'esol', 'lipo']
REG_DATASET = ['freesolv', 'esol', 'lipo']

class MolNetData(InMemoryDataset):
    def __init__(self, dataset_name, split='train', dataset_path='/share/project/sharefs-skfeng/OtherCode/UniMol/molecular_property_prediction', seed=1, cut_length=False):

        assert dataset_name in MOLNET_DATASET
        if dataset_name in REG_DATASET:
            self.is_cls = False
        else:
            self.is_cls = True

        self.dataset_name = dataset_name
        self.split = split

        self.cut_length = cut_length

        self.data_df = load_lmdb(os.path.join(dataset_path, dataset_name, f"{split}.lmdb"))  # 0: smiles, 1: labels, bbbp 2 classes, 1369 vs 262, 2: 3D position, 11 conformations, 11 * 50 * 3, 3: atom types

        # train_df = load_lmdb(os.path.join(dataset_path, dataset_name, "train.lmdb"))
        # valid_df = load_lmdb(os.path.join(dataset_path, dataset_name, "valid.lmdb"))
        # test_df = load_lmdb(os.path.join(dataset_path,dataset_name, "test.lmdb"))
        self.process()

        self.data_len = len(self.labels)
        # task weight
        if self.is_cls:
            self.task_weights = (self.labels != -1).astype(np.float32)
        else:
            self.task_weights = np.ones(self.data_len, dtype=np.float32).reshape(-1, 1) # for reg task
            self.mean = self.labels.mean()
            self.std = self.labels.std()
        # get number tasks
        self.num_tasks = self.data_list[0].y.shape[0]
        self.__indices__ = None

        # seed for sample conformation
        self.seed = seed
    # process, smiles--> input format (self.data_list)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def process(self):
    

        smiles_list = self.data_df[0]
        self.labels = self.data_df[1]
        self.poses= self.data_df[2]
        atom_list = self.data_df[3]

        print('Converting SMILES strings into graphs...')
        self.data_list = []
       
        with Pool(processes=30) as pool:
            if self.cut_length:
                iter = pool.imap(smiles2graph_cut, smiles_list)
            else:
                iter = pool.imap(smiles2graph, smiles_list)

            for i, graph in tqdm(enumerate(iter), total=len(smiles_list)):
                try:
                    data = Data()

                    label = self.labels[i]
                    pos_array = self.poses[i]

                    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                    assert (len(graph['node_feat']) == graph['num_nodes'])

                    data.__num_nodes__ = int(graph['num_nodes'])
                    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                    data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                    data.y = torch.Tensor(label)

                    # add pos
                    H_mask = torch.BoolTensor([atom!='H' for atom in atom_list[i]]).unsqueeze(0).unsqueeze(-1)
                    data.pos = torch.from_numpy(pos_array).to(torch.float32)[H_mask.expand(11, -1, 3)].reshape(11, -1, 3)
                    if self.cut_length:
                        data.pos = data.pos[:, :128, :]
                    # data.pos = torch.from_numpy(pos_array).to(torch.float32)
                    self.data_list.append(data)
                except:
                    continue

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        # self.data, slices = self.collate(self.data_list)

        # print('Saving...')
        # torch.save((data, slices), self.processed_paths[0])

    def len(self):
        return self.data_len

    # getitem
    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = copy.copy(self.data_list[idx])
        pos_size = len(item.pos)
        with data_utils.numpy_seed(self.seed, self.epoch, idx):
            sample_idx = np.random.randint(pos_size)
        item.pos = item.pos[sample_idx]
        item.idx = idx
        item.weight = self.task_weights[idx]
        return item
    
    