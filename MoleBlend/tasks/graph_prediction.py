# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os, math

import contextlib
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING, II, open_dict, OmegaConf

import torch
import numpy as np
from fairseq.data import (
    NestedDictionaryDataset,
    NumSamplesDataset,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from fairseq.dataclass import ChoiceEnum
from fairseq.optim.amp_optimizer import AMPOptimizer

from ..data.dataset import (
    PCQPreprocessedData,
    BatchedDataDataset,
    EpochShuffleDataset,
    TargetDataset,
    PYGPreprocessedData,
    MOLPreprocessedData
)
import pickle

# checkpoint load utils
from fairseq import checkpoint_utils
from fairseq.distributed import utils as distributed_utils

logger = logging.getLogger(__name__)
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])


@dataclass
class GraphPredictionConfig(FairseqDataclass):

    # task / dataset arguments flag

    # data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    data_path: str = field(  
        default="",
        metadata={
            "help": "path to data file"
        },
    )

    dataset_name: str = field( 
        default="PCQM4M-LSC",
        metadata={"help": "name of the dataset"},
    )

    noise_scale: float = field(
        default=0.01,
        metadata={
            "help": "noise scale"
        },
    )

    sandwich_ln: bool = field(
        default=False,
        metadata={"help": "apply layernorm via sandwich form"},
    )

    num_classes: int = field(
        default=-1,
        metadata={"help": "number of classes or regression targets"},
    )
    init_token: Optional[int] = field(
        default=None,
        metadata={"help": "add token at the beginning of each batch item"},
    )
    separator_token: Optional[int] = field(
        default=None,
        metadata={"help": "add separator token between inputs"},
    )
    no_shuffle: bool = field(
        default=False,
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed tokens_per_sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    add_prev_output_tokens: bool = field(
        default=False,
        metadata={
            "help": "add prev_output_tokens to sample, used for encoder-decoder arch"
        },
    )
    max_positions: int = field(
        default=512,
        metadata={"help": "max tokens per example"},
    )

    num_atoms: int = field(
        default=512 * 9,
        metadata={"help": "number of atom types in the graph"},
    )

    num_edges: int = field(
        default=512 * 3,
        metadata={"help": "number of edge types in the graph"},
    )

    num_in_degree: int = field(
        default=512,
        metadata={"help": "number of in degree types in the graph"},
    )

    num_out_degree: int = field(
        default=512,
        metadata={"help": "number of out degree types in the graph"},
    )

    num_spatial: int = field(
        default=512,
        metadata={"help": "number of spatial types in the graph"},
    )

    num_edge_dis: int = field(
        default=128,
        metadata={"help": "number of edge dis types in the graph"},
    )

    multi_hop_max_dist: int = field(
        default=5,
        metadata={"help": "max distance of multi-hop edges"},
    )

    edge_type: str = field(
        default="multi_hop",
        metadata={"help": "edge type in the graph"},
    )

    regression_target: bool = II("criterion.regression_target")
    # classification_head_name: str = II("criterion.classification_head_name")
    seed: int = II("common.seed")

    numpy_seed: int = field(
        default=1,
        metadata={"help": "seed of numpy to make subset indices"}
    )

    regularization_3d_denosing: bool = field(
        default=False,
    )
    blending: bool = field(
        default=False,
        metadata={
            "help": "code switch"
        }
    )
    # blending
    blend_pred_spd: bool = field(
        default=False,
        metadata={
            "help": "predict the spd loss"
        }
    )
    blend_pred_edge: bool = field(
        default=False,
        metadata={
            "help": "predict the edge loss"
        }
    )
    blend_pred_3d: bool = field(
        default=False,
        metadata={
            "help": "predict the 3d loss"
        }
    )

    cut_length: bool = field(
        default=False,
    )




@register_task("graph_prediction", dataclass=GraphPredictionConfig)
class GraphPredictionTask(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        if self.cfg.dataset_name.startswith("PCQM4M-LSC"):
            self.dm = PCQPreprocessedData(dataset_name=self.cfg.dataset_name, dataset_path=self.cfg.data_path)
        elif self.cfg.dataset_name.startswith("molecu"):
            self.mol = MOLPreprocessedData(dataset_name=self.cfg.dataset_name, dataset_path=self.cfg.data_path, seed=self.cfg.seed, cut_length=self.cfg.cut_length)
            if self.mol.dataset_train.dataset.is_cls:
                self.moltask_type = 'classification'
            else:
                self.moltask_type = 'regression'
            
            self.num_tasks = self.mol.dataset_train.dataset.num_tasks

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"
        return cls(cfg)

    def load_dataset(self, split, combine=False, **kwargs):
        if self.cfg.dataset_name.startswith("PCQM4M-LSC"):
            return self.load_dm_dataset(split, combine, **kwargs)
        else:
            return self.load_mol_dataset(split, combine, **kwargs)


    def load_mol_dataset(self, split, combine=False, **kwargs):
        assert split in ["train", "valid", "test"]

        if split == "train":
            # batched_data = self.dm.dataset_train
            batched_data = self.mol.dataset_train
        elif split == "valid":
            batched_data = self.mol.dataset_val
            # batched_data = self.dm.dataset_val
        elif split == "test":
            batched_data = self.mol.dataset_test
            # batched_data = self.dm.dataset_test


        batched_data = BatchedDataDataset(batched_data,
            dataset_version="3D",
            max_node=self.mol.max_node,
            multi_hop_max_dist=self.mol.multi_hop_max_dist,
            spatial_pos_max=self.mol.spatial_pos_max
        )

        target = TargetDataset(batched_data)

        dataset = NestedDictionaryDataset({
            "nsamples": NumSamplesDataset(),
            "net_input": {
                "batched_data": batched_data
            },
            "target": target
        }, sizes=np.array([1] * len(batched_data))) # FIXME: workaroud, make all samples have a valid size

        if split == "train":
            dataset = EpochShuffleDataset(dataset, size=len(batched_data), seed=self.cfg.seed)

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def load_dm_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        assert split in ["train", "valid", "test"]

        if split == "train":
            batched_data = self.dm.dataset_train
        elif split == "valid":
            batched_data = self.dm.dataset_val
        elif split == "test":
            batched_data = self.dm.dataset_test

        batched_data = BatchedDataDataset(batched_data,
            dataset_version="2D" if self.cfg.dataset_name == 'PCQM4M-LSC-V2' else "3D",
            max_node=self.dm.max_node,
            multi_hop_max_dist=self.dm.multi_hop_max_dist,
            spatial_pos_max=self.dm.spatial_pos_max
        )

        target = TargetDataset(batched_data)

        dataset = NestedDictionaryDataset({
            "nsamples": NumSamplesDataset(),
            "net_input": {
                "batched_data": batched_data
            },
            "target": target
        }, sizes=np.array([1] * len(batched_data))) # FIXME: workaroud, make all samples have a valid size

        if split == "train":
            dataset = EpochShuffleDataset(dataset, size=len(batched_data), seed=self.cfg.seed)

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_positions = self.cfg.max_positions

        model = models.build_model(cfg, self)

        if cfg.finetune_from_model:
            cfg.blending = False

        return model

    def max_positions(self):
        return self.cfg.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary