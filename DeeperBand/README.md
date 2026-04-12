# DeeperBand
DeeperBand is a deep learning approach to predict Tc for superconductors from electronic bands. The universal features provided on our web https://www.superband.work/ interface facilitate the design of novel superconductors with a wide-range of applications. We describe the database and the algorithm to generate it in our paper [1] https://arxiv.org/abs/2409.07721. 

# Installation

1. Download the DeeperBand repository into the current directory
   ```sh
   git clone https://github.com/ljcj007/DeeperBand.git
   cd DeeperBand
   pip install --upgrade pip
   pip install -e .
   ```

# How to use

 - `deeperband train` - train a DeeperBand model.
 - `deeperband evaluate` - evaluate the Tc from "vasprun.xml" file.
 - `deeperband download` - download pretrained models and training datasets.

 
## Evaluate the Tc
```bash
$ deeperband basecaller /vasp/run_dir  pretrain/0724.pt > Tc.info
```

DeeperBand will download and cache the pretrain model automatically on first use but all models can be downloaded with -

``` bash
$ deeperband download --models --show  # show all available pretrain models
$ deeperband download --models         # download all available pretrain models
$ deeperband download --training         # download the SuperBand training set
```
## Training your own model

To train a model using your own reads, should run deeperband download --training for training.

```bash
$ deeperband train --epochs 1 --lr 5e-4 --directory /data/training/ctc-data /data/training/model-dir
```

# License
The DeeperBand is subject to the Creative Commons Attribution 4.0 License, implying that the content may be copied, distributed, transmitted, and adapted, without obtaining specific permission from the repository owner, provided proper attribution is given to the repository owner. All software in this repository is subject to the MIT license. See `LICENSE.md` for more information.
