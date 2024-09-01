# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from torch_geometric.datasets import *
from torch_geometric.data import Dataset
from .pyg_dataset import GraphormerPYGDataset
import torch.distributed as dist

import sys
import os
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(grandparent_dir)
from data.mol_datasets import MolNetData


class MyQM7b(QM7b):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).process()
        if dist.is_initialized():
            dist.barrier()



# if molnet, dataset_spec is in format( moleculenet:bbbp ), bbbp as param

class PYGDatasetLookupTable:
    @staticmethod
    def GetPYGDataset(dataset_spec: str, seed: int, dataset_path: str = 'dataset', task_idx: int = 0, cut_length: bool = False) -> Optional[Dataset]:
        split_result = dataset_spec.split(":")
        if len(split_result) == 2:
            name, params = split_result[0], split_result[1]
            params = params.split(",")
        elif len(split_result) == 1:
            name = dataset_spec
            params = []
        inner_dataset = None
        num_class = 1

        train_set = None
        valid_set = None
        test_set = None

        root = dataset_path
        if name == "qm7b":
            inner_dataset = MyQM7b(root=root)
        elif name == "moleculenet":
            subtask_name = params[0]
            train_set = MolNetData(subtask_name, split="train", dataset_path=dataset_path, cut_length=cut_length)
            valid_set = MolNetData(subtask_name, split="valid", dataset_path=dataset_path)
            test_set = MolNetData(subtask_name, split="test", dataset_path=dataset_path)
            return GraphormerPYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                    data_name=name
                )
        else:
            raise ValueError(f"Unknown dataset name {name} for pyg source.")
        if train_set is not None:
            return GraphormerPYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                )
        else:
            return (
                None
                if inner_dataset is None
                else GraphormerPYGDataset(inner_dataset, seed, task_idx=task_idx)
            )
