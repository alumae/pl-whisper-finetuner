#from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

#warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
#warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import sys
import os
import logging
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.core.module import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, Callback
import data
from model import LitWhisper

from lightning.pytorch.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.model", "data.model")




if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    root_dir = os.path.dirname(os.path.realpath(__file__))
    cli = MyLightningCLI(
        LitWhisper, data.WhisperDataModule, seed_everything_default=1234, save_config_kwargs={"overwrite": True}, run=True
    )

