import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger
import kornia
from src.utils.dataset import read_megadepth_gray, read_megadepth_depth,read_rgb_ir_depth


class RGB_IRDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0
        # self.scene_info = np.load(npz_path, allow_pickle=True)
        self.scene_info = dict(np.load(npz_path, allow_pickle=True))
        # print("self.scene_infos.keys():",self.scene_info.keys())
        self.pair_infos = self.scene_info['pair_infos'].copy()
        # del self.scene_info['pair_infos']
        # self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)


    def generateRandomHomography(self,shape, GLOBAL_MULTIPLIER = 0.3):
        #Generate random in-plane rotation [-theta,+theta]
        theta = np.radians(np.random.uniform(-30, 30))

        #Generate random scale in both x and y
        scale_x, scale_y = np.random.uniform(0.35, 1.2, 2)

        #Generate random translation shift
        tx , ty = -shape[1]/2.0 , -shape[0]/2.0 
        txn, tyn = np.random.normal(0, 120.0*GLOBAL_MULTIPLIER, 2) 

        c, s = np.cos(theta), np.sin(theta)

        #Affine coeffs
        sx , sy = np.random.normal(0,0.6*GLOBAL_MULTIPLIER,2)

        #Projective coeffs
        p1 , p2 = np.random.normal(0,0.006*GLOBAL_MULTIPLIER,2)


        # Build Homography from parmeterizations
        H_t = np.array(((1,0, tx), (0, 1, ty), (0,0,1))) #t

        H_r = np.array(((c,-s, 0), (s, c, 0), (0,0,1))) #rotation,
        H_a = np.array(((1,sy, 0), (sx, 1, 0), (0,0,1))) # affine
        H_p = np.array(((1, 0, 0), (0 , 1, 0), (p1,p2,1))) # projective

        H_s = np.array(((scale_x,0, 0), (0, scale_y, 0), (0,0,1))) #scale
        H_b = np.array(((1.0,0,-tx +txn), (0, 1, -ty + tyn), (0,0,1))) #t_back,

        #H = H_e * H_s * H_a * H_p
        H = np.dot(np.dot(np.dot(np.dot(np.dot(H_b,H_s),H_p),H_a),H_r),H_t)

        return H
    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        # (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]

        # read grayscale image and mask. (1, h, w) and (h, w)
        # img_name0 = osp.join(self.root_dir, self.scene_info['rgb_image_paths'][idx])
        # img_name1 = osp.join(self.root_dir, self.scene_info['ir_image_paths'][idx])
        img_name0 = osp.join(self.scene_info['rgb_image_paths'][idx])
        img_name1 = osp.join(self.scene_info['ir_image_paths'][idx])
                
        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0, mask0, scale0 = read_megadepth_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1 = read_megadepth_gray(
            img_name1, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # read depth. shape: (h, w)
        if self.mode in ['train', 'val']:
            depth0 = read_rgb_ir_depth(
                image0, pad_to=self.depth_max_size)                 
            depth1 = read_rgb_ir_depth(
                image1, pad_to=self.depth_max_size)            
            # depth0 = read_megadepth_depth(
            #     osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size)
            # depth1 = read_megadepth_depth(
            #     osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size)
        else:
            depth0 = depth1 = torch.tensor([])

        # read intrinsics of original size
        # K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
        # K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)
        K_0 = torch.eye(3)
        K_1 = torch.eye(3)
        # K_0 and K_1 can be set to random Homography matrix due to calculation results (when K_0= I)
        # mainwhile, use transformation on input images
        Homo_Trans = 1
        self.use_unsuperpoint_homo = 1
        if Homo_Trans:
            image0_h = int(image0.shape[-2])
            image0_w = int(image0.shape[-1])
            image1_h = int(image1.shape[-2])
            image1_w = int(image1.shape[-1])  
            if self.use_unsuperpoint_homo:
                K_0 = torch.tensor(enhance([image0_h, image0_w]),dtype = torch.float32)
                K_1 = torch.tensor(enhance([image1_h, image1_w]),dtype = torch.float32)
            else:
                K_0 = self.generateRandomHomography((image0_h,image0_w),GLOBAL_MULTIPLIER = 0.1).astype(np.float32)
                K_1 = self.generateRandomHomography((image1_h,image1_w),GLOBAL_MULTIPLIER = 0.1).astype(np.float32)
            image0 = kornia.geometry.transform.warp_perspective(image0.unsqueeze(0), torch.tensor(K_0,dtype=torch.float).unsqueeze(0),dsize = (image0_h,image0_w), padding_mode = 'zeros').squeeze(0)
            image1 = kornia.geometry.transform.warp_perspective(image1.unsqueeze(0), torch.tensor(K_1,dtype=torch.float).unsqueeze(0),dsize = (image1_h,image1_w), padding_mode = 'zeros').squeeze(0)
        # read and compute relative poses
        # T0 = self.scene_info['poses'][idx0]
        # T1 = self.scene_info['poses'][idx1]
        T0 = np.eye(4)
        T1 = np.eye(4)
        
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,  # (1, h, w)
            'depth0': depth0,  # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'RGB_IR',
            'scene_id': self.scene_id,
            'pair_id': idx,
            'pair_names': (self.scene_info['rgb_image_paths'][idx], self.scene_info['ir_image_paths'][idx]),
        }

        # for LoFTR training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data


def enhance(IMAGE_SHAPE= [416,416]):
    
    src_point = np.array([[               0,                0],
                          [IMAGE_SHAPE[1]-1,                0],
                          [               0, IMAGE_SHAPE[0]-1],
                          [IMAGE_SHAPE[1]-1, IMAGE_SHAPE[0]-1]], dtype=np.float32)  # 圖片的四個頂點

    dst_point = get_dst_point(0.2, IMAGE_SHAPE)  # 透视信息

    # rot = random.randint(-2, 2) * config['homographic']['rotation'] + random.randint(0, 15)  # 旋转
    rotation = 25
    rot = random.randint(-rotation, rotation)  # [low, high] 和numpy的随机不一样，high是可以取的

    # scale = 1.2 - config['homographic']['scale'] * random.random()  # 缩放 1.2 - 0.2 * (0,1.0) -> (1.2,1.0)
    scale = 1.0 + 0.2 * random.randint(-10, 20) * 0.1  # 缩放 1.2 - 0.2 * (0,1.0) -> (1.2,1.0)

    center_offset = 40
    center = (IMAGE_SHAPE[1] / 2 + random.randint(-center_offset, center_offset),
              IMAGE_SHAPE[0] / 2 + random.randint(-center_offset, center_offset))

    RS_mat = cv2.getRotationMatrix2D(center, rot, scale)
    f_point = np.matmul(dst_point, RS_mat.T).astype('float32')
    mat = cv2.getPerspectiveTransform(src_point, f_point)

    return mat
