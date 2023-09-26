import torch
import pytorch_lightning as pl
import argparse
from dataclasses import asdict
import copy

from model import LitWhisper

def export_model(checkpoint_path, output_path):
    # Load the model with LightningModule from the checkpoint
    model = LitWhisper.load_from_checkpoint(checkpoint_path, map_location=torch.device("cpu"))
    model.eval()  # Set the model to evaluation mode
    
    # Extract the underlying PyTorch model from the LightningModule
    whisper_model = model.model
    
    # save model in half precision to save space
    whisper_model = copy.deepcopy(whisper_model).half()
    # save model weights and config in a dictionary that can be loaded with `whisper.load_model`
    torch.save({"model_state_dict": whisper_model.state_dict(), "dims": asdict(whisper_model.dims)}, output_path)

    print(f'Model exported to {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export PyTorch Lightning checkpoint to a pure Whisper model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the PyTorch Lightning checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Path to save the exported Whisper model')
    
    args = parser.parse_args()
    export_model(args.checkpoint, args.output)
