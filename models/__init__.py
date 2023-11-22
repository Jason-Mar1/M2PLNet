# from .p2pnet import build
import argparse

import torch

from .M2PLNet import  build
# build the M2PLNet model
# set training to 'True' during training
def build_model(args, training=False):
    return build(args, training)

