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
# dataset_dir = "/mnt/ssd/home/chuxinning/LoFTR/data/rgb_ir/test_A"
dataset_dir = "/mnt/ssd/home/chuxinning/LoFTR/data/rgb_ir/eval/train_A"
homo_npy_dir = "/mnt/ssd/home/chuxinning/LoFTR/data/pir_ir/homo_test"
imgs_lst = os.listdir(dataset_dir)
matcher = LoFTR(config=default_cfg) #使用default_cfg,对不对？

model_ckpts_dir = "/mnt/ssd/home/chuxinning/LoFTR/logs/tb_logs/outdoor-ds-416-bs=12_RGB_IR_homo/version_0/checkpoints"

model_lst = os.listdir(model_ckpts_dir)
model_lst = [i for i in model_lst if i.endswith(".ckpt")]
metrix_txt = open(os.path.join(model_ckpts_dir,"..",'metrics.txt'),"a")
metrix_txt.write("\nepoch\tREP\tLOC\tHA_1\tHA_3\tHA_5\tHA_10...\n")
metrix_txt.close()
for epoch,model_ckpt in tqdm.tqdm(enumerate(model_lst)):
    matcher.load_state_dict(torch.load(os.path.join(model_ckpts_dir,model_ckpt))['state_dict'])####
    matcher = matcher.eval().cuda()
    # default_cfg['coarse']
    dist_list = []
    REP = 0
    LOC = 0
    DIST = 0
    # Load example images
    for i,img in tqdm.tqdm(enumerate(imgs_lst)):
        img0_pth = os.path.join(dataset_dir,img)
        img1_pth = img0_pth.replace("_A","_B")
        homo_pth = os.path.join(homo_npy_dir,img.split(".")[0]+".npy")
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
        # print(batch.keys())
        # print(batch['mkpts0_f'].shape)
        # print(batch['mkpts1_f'].shape)
        
        kpts_match = torch.cat([batch['mkpts0_f'],batch['mkpts1_f']],dim=1)
        kpts_match = kpts_match.cpu().numpy()
        dist,inliers,rep,loc,H_pred,if_Work = compute_dist(kpts_match,H_gt=homo_matrix,img=img_size)
        fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text,path="/mnt/ssd/home/chuxinning/LoFTR/outputs/figs/LoFTR-colab-demo.pdf")
        # print(dist,rep,loc)
        dist_list.append(dist)
        DIST += dist
        REP += rep
        LOC += loc
        
    REP /=len(imgs_lst)
    LOC /=len(imgs_lst)
    DIST /=len(imgs_lst)
    homos,homo_thres = eval_summary_homography(dist_list)
    
    metrix_txt = open(os.path.join(model_ckpts_dir,"..",'metrics.txt'),"a")
    metrix_txt.write(str(epoch)+'\t')
    metrix_txt.write(str(REP)+'\t'+str(LOC)+'\t')
    for num_homo,homo in enumerate(homos):
        metrix_txt.write(str(homo)+'\t')
    metrix_txt.write(os.path.join(model_ckpts_dir,model_ckpt)+'\t')
    metrix_txt.write('\n')
    metrix_txt.close()
