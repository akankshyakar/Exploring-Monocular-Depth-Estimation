import numpy as np
import dataloader.transforms as transforms
from dataloader.dataloader import MyDataloader

iheight, iwidth = 480, 640 # raw image size

class NYUDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(NYUDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (448, 448) #(224, 224) #(228, 304) #(iheight, iwidth) 
