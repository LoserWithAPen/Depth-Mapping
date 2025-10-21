import cv2
import torch
from picamera2 import Preview
import numpy as np
import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitb' # or 'vits', 'vitl', 'vitg'


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'/home/liemme/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()
while True:
    raw_img = cv2.imread('/home/liemme/Downloads/Elmo.jpeg')
    
    
    cv2.imshow('Raw image', raw_img)
    depth = model.infer_image(raw_img) # HxW raw depth map in numpy
    #cv2.imshow('Depthed', depth)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    cv2.imshow('Raw Depth', depth)
    key = cv2.waitKey(1)
    if key == 27:
        break	

cam.release()
cv2.destroyAllWindows()