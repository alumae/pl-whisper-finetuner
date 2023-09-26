from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

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


#seed_everything(234)

# def main(args):

#     train_loader = data.get_dataloader(
#         json=args.train_json,
#         tokenizer=tokenizer,
#         batch_size=args.batch_size,           
#         no_timestamps_training=args.no_timestamps_training,
#         prompt_use_rate=args.prompt_use_rate,
#         no_timestamps_rate=no_timestamps_rate,
#         shuffle=True,
#         do_spec_augment=False,
#         specaug_conf=None
#     )

#     dev_loader = data.get_dataloader(
#         json=args.dev_json,
#         tokenizer=tokenizer,
#         batch_size=args.dev_batch_size,           
#         no_timestamps_training=args.no_timestamps_training,
#         prompt_use_rate=1.0,
#         no_timestamps_rate=0.0,
#         shuffle=False,
#         do_spec_augment=False
#     )

#     model = LitWhisper(**vars(args))

#     checkpoint_callback = ModelCheckpoint(
#             save_top_k=1,
#             save_last=True,
#             verbose=True,
#             monitor='val_loss',
#             mode='min'                
#     )    

#     callbacks=[checkpoint_callback]
#     trainer = Trainer.from_argparse_args(args, callbacks=callbacks)    
#     trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader)



class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.model", "data.model")

    # @staticmethod
    # def subcommands(self):
    #     """Defines the list of available subcommands and the arguments to skip."""
    #     commands = super().subcommands()
    #     commands["export"] = {"model"}
    #     return commands
    

    # def export(self):
    #     model_path = f"{self.config['trainer']['default_root_dir']}/exported_model.pt"
    #     whisper_model = copy.deepcopy(self.model.model).half()
    #     torch.save({"model_state_dict": whisper_model.state_dict(), "dims": asdict(whisper_model.dims)}, model_path)

    def run(self):
        super().run()
        
        if self.subcommand == 'export':
            self.export()

    

if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    #numba_logger = logging.getLogger('numba')
    #numba_logger.setLevel(logging.WARNING)

    root_dir = os.path.dirname(os.path.realpath(__file__))
    cli = MyLightningCLI(
        LitWhisper, data.WhisperDataModule, seed_everything_default=1234, save_config_kwargs={"overwrite": True}, run=True
    )


    # # data
    # parser.add_argument('--train_json', required=True, type=str)       
    # parser.add_argument('--dev_json', required=True, type=str)
    # parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    # parser.add_argument("--dev_batch_size", type=int, default=16, help="Batch size for validation")
    # parser.add_argument(
    #     "--no_timestamps_training",
    #     action="store_true",
    #     help="Always use the no-timestamps training mode",
    # )
    # parser.add_argument(
    #     "--prompt_use_rate",
    #     type=float,
    #     default=0.5,
    #     help="How often to use prompts for conditioning the generation",
    # )
    # parser.add_argument(
    #     "--no_timestamps_rate",
    #     type=float,
    #     default=0.5,
    #     help=(
    #         "How often to use the no-timestamps mode. Only used if --no-timestamps-training "
    #         "is NOT set"
    #     ),
    # )



    # args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    #main(args)
