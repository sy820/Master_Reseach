import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2

from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout

# ====================================================
# Library
# ====================================================
import os
import gc
import sys
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM

sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
import timm

from torch.cuda.amp import autocast, GradScaler

import warnings
warnings.filterwarnings('ignore')

class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(self.cfg.model_name, pretrained=pretrained)
        self.n_features = self.model.head.in_features
        self.model.head = nn.Identity()
        self.fc = nn.Linear(self.n_features, self.cfg.target_size)

    def feature(self, image):
        feature = self.model(image)
        return feature
        
    def forward(self, image):
        feature = self.feature(image)
        output = self.fc(feature)
        return output

class CFG:
    apex=False
    debug=False
    print_freq=10
    num_workers=4
    size=224
    model_name='vit_large_patch32_224'
    scheduler='CosineAnnealingLR' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epochs=5
    #factor=0.2 # ReduceLROnPlateau
    #patience=4 # ReduceLROnPlateau
    #eps=1e-6 # ReduceLROnPlateau
    T_max=3 # CosineAnnealingLR
    #T_0=3 # CosineAnnealingWarmRestarts
    lr=1e-4
    min_lr=1e-6
    batch_size=32
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    target_size=1
    target_col='KIc'
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=True
    grad_cam=True
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args(args=[])
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    
    df = pd.read_csv('../KIc/vgg16_KIC/inout_data.csv', header=None, names=['Id', 'KIc'])
    df['file_path'] = ['../KIc/vgg16_KIC/imagedata/1' + str(i).zfill(5) + '.jpg' for i in df['Id']]
    
    args = get_args()
    model = CustomModel(CFG, pretrained=False)
    state = torch.load(f'../KIc/{CFG.model_name}_fold1_best.pth', 
                               map_location=torch.device(device))['model']
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,], std=[0.5,]),
    ])
#     transform = A.Compose([
#         A.RandomResizedCrop(CFG.size, CFG.size, scale=(0.85, 1.0)),
#         A.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225],
#         ),
#         ToTensorV2(),
#     ])
    
    for i, row in df.iterrows():
        args.image_path = row['file_path']
        Id = row['Id']
#         img = Image.open(args.image_path)
#         img = img.resize((224, 224))
#         input_tensor = transform(img).unsqueeze(0)
#         if args.use_cuda:
#             input_tensor = input_tensor.cuda()
        
#         img = cv2.imread(args.image_path, 1)[:, :, ::-1]
#         img = cv2.imread(args.image_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         input_tensor = transform(image=img)['image'].unsqueeze(0).cuda()
        
        img = Image.open(args.image_path)
        img = img.resize((224, 224))
        input_tensor = transform(img).unsqueeze(0).cuda()


        if args.category_index is None:
            print("Doing Attention Rollout")
            attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, 
                discard_ratio=args.discard_ratio)
            mask = attention_rollout(input_tensor)
            name = "attention_rollout_{}_{:.3f}_{}.png".format(Id, args.discard_ratio, args.head_fusion)
        else:
            print("Doing Gradient Attention Rollout")
            grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
            mask = grad_rollout(input_tensor, args.category_index)
            name = "grad_rollout_{}_{:.3f}_{}.png".format(args.category_index,
                args.discard_ratio, args.head_fusion)

        np_img = np.array(img)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask = show_mask_on_image(np_img, mask)
#         cv2.imshow("Input Image", np_img)
#         cv2.imshow(name, mask)
        cv2.imwrite("input.png", np_img)
        cv2.imwrite(name, mask)
        cv2.waitKey(-1)
        break