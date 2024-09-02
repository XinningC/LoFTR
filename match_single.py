import os
os.chdir("..")
from copy import deepcopy
 
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
import tqdm
 
from src.loftr import LoFTR, default_cfg
from match_utils import * 
# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
img_size = [416,416]


img0_pth = "/mnt/ssd/home/chuxinning/LoFTR/data/rgb_ir/eval/train_A/20220116_2105_000003.png"
img1_pth = "/mnt/ssd/home/chuxinning/LoFTR/data/rgb_ir/eval/train_B/20220116_2105_000003.png"
homo_pth = "/mnt/ssd/home/chuxinning/LoFTR/data/rgb_ir/eval/homo_test/20220116_2105_000003.npy"
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
# img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//2, img0_raw.shape[0]//2))  # input size shuold be divisible by 8
# img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//2, img1_raw.shape[0]//2))
img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//8*8, img0_raw.shape[0]//8*8))  # input size shuold be divisible by 8
img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//8*8, img1_raw.shape[0]//8*8))
homo_matrix = np.load(homo_pth)
img1_raw = cv2.warpPerspective(img1_raw,homo_matrix,(img1_raw.shape[1]//8*8, img1_raw.shape[0]//8*8))                                
img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("/mnt/ssd/home/chuxinning/LoFTR/logs/tb_logs/outdoor-ds-416-bs=12_RGB_IR_homo/version_0/checkpoints/epoch=1-auc@5=0.000-auc@10=0.000-auc@20=0.000.ckpt")['state_dict'])####
matcher = matcher.eval().cuda()
# Inference with LoFTR and get prediction
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()

# Draw
color = cm.jet(mconf)
text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)),
]

fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text,path="/mnt/ssd/home/chuxinning/LoFTR/outputs/figs/RGB_IR_Demo.pdf")

print(mkpts0.shape)