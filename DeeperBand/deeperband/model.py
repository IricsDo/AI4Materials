import torch.nn as nn
from vit_pytorch.vit_3d import ViT

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = ViT(image_size = 32, frames = 32, 
            image_patch_size = 8,
            frame_patch_size = 8,
            channels = 18,    num_classes = 1,
            dim = 534,       depth = 3,       
            heads = 64,      mlp_dim = 1038,
            dropout = 0.107,  emb_dropout = 0.197
        )
    def forward(self, x):
        return self.v(x)