"""
Bonito model evaluator
"""

import random,os,sys
import torch
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from deeperband.model import Net

__dir__ = os.path.dirname(os.path.realpath(__file__))
__data__ = os.path.join(__dir__, "dataset/")
__models__ = os.path.join(__dir__, "pretrain/")
def set_manual_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        print("the used gpu: ", torch.cuda.device_count(), torch.cuda.is_available())
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def main(args):
    
    set_manual_seed(args.seed)
    device = torch.device(args.device)
    
    model = Net()
    
    dirname = args.model_directory
    if not os.path.exists(dirname) and os.path.exists(os.path.join(__models__, dirname)):
        dirname = os.path.join(__models__, dirname)
    
    print(f">Using model {args.model_directory}")
    
    # Fixed loading for newer PyTorch (weights_only=False is safe here because you trust the file)
    checkpoint = torch.load(dirname, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    print(">Loading directory {}".format(args.directory))
    from deeperband.band_data import get_bands_sc
    vasp_run=args.directory+'/vasprun.xml'
    inputs=get_bands_sc(vasp_run)
    inputs=inputs.float().to(device)
    outputs = model(inputs).squeeze().cpu().detach().numpy()
    print(f">Evaluate the possible Tc: {np.exp(outputs)-1:.4f} K")

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("directory")
    parser.add_argument("--model_directory", default="0724.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=9, type=int)
    return parser
