import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from picamera2 import Preview


from depth_anything_v2.dpt import DepthAnythingV2

raw_video = cv2.VideoCapture(0)
frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitl', 'vitg'
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
depth_anything = DepthAnythingV2(**model_configs[encoder])
depth_anything.load_state_dict(torch.load(f'/home/liemme/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()
output_width = frame_width

margin_width = 50
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

#output_path = os.path.join("/home/liemme/Depth-Anything-V2", "video" + '.mp4')
#out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))

while raw_video.isOpened():
            ret, raw_frame = raw_video.read(0)
            
            if not ret:
                break
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            img3 = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
            img4 = cv2.GaussianBlur(img3, (3, 3), 1)

            depth = depth_anything.infer_image(img4)
            
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            
            
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            
            #depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            depth = depth.astype(np.uint8)

            
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            #combined_frame = cv2.hconcat([raw_frame, split_region, depth])
            #combined_frame = cv2.hconcat(raw_frame, split_region)
                
            #out.write(combined_frame)
            cv2.imshow("Depth Video", depth)
        
raw_video.release()
#out.release()