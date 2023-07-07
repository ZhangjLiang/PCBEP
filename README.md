# PCBEP
PCBEP is a B cell epitope predict tool which is based on esm-2 model and feature training consisting of atomic physicochemical properties, PSSM matrix.
## dataset 

## train model
#```
from itertools import product
from random import random
from turtle import shape

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ones
from torch_geometric.data import Data, DataLoader
import numpy as np
from torch.nn import Sequential as Seq, Dropout, GELU, Linear as Lin, ReLU, BatchNorm1d as BN, LayerNorm as LN, Softmax
import torch_geometric.transforms as T
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, radius, global_mean_pool, knn
from torch_geometric.data import batch
from sklearn import metrics
from zy_pytorchtools import EarlyStopping
#```
