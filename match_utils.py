# -*- coding: utf-8 -*-
"""
@author:Wen
"""
import torch as t
import torch.nn.functional as F
import numpy as np
from numba import jit
import time
import cv2
import torch
import pydegensac
# import util.util as util
@jit
def fast_numba_nms(grid,pad,rcorners):
    """

    :param grid:h*pad,w+pad
    :param pad: pad
    :param rcorners:N,2
    :return:keepy,keepx
    """
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    return keepy,keepx

def compose(img1, img2):
    return t.cat((img1, img2), axis=0)  # 2N,C,H,W


def decompose(res):
    N = res[0].size()[0]
    assert N % 2 == 0, "error with N,N should not be {}".format(N)
    N = int(N / 2)

    # by cxn:
    # score, pos, des = res[0], res[1], res[2]
    # return score[:N], pos[:N], des[:N], score[N:], pos[N:], des[N:]



    # score, pos, des = res[0], res[1], res[2]
    # return score[:N], pos[:N], des[:N], score[N:], pos[N:], des[N:]
    # s1s2是score map  p1p2是position map   d1d2是des map
    s1, s2, p1, p2, d1, d2 = res[0], res[1], res[2], res[3], res[4], res[5]
    # 这里用N分开，因为N是batch size，把原图和单应变换图分开
    return s1[:N], s2[:N], p1[:N], p2[:N], d1[:N], d2[:N], s1[N:], s2[N:], p1[N:], p2[N:], d1[N:], d2[N:],

def calculate_pos(pos_map):
    # pos_map N 2 h/8 w/8
    N, _, h, w = pos_map.size()
    x_base_coor = torch.arange(w).view(1, 1, 1, w).expand(N, 1, h, w).type_as(pos_map)
    y_base_coor = torch.arange(h).view(1, 1, h, 1).expand(N, 1, h, w).type_as(pos_map)
    x_coor = (x_base_coor + pos_map[:, [0], :, :]) * 8
    y_coor = (y_base_coor + pos_map[:, [1], :, :]) * 8
    coord = torch.cat((x_coor, y_coor), axis=1)
    return coord  # N,2,h,w 在原图上的坐标

def compose_cord_des_score_pos_index(coord, des, score_map, relative_coor):
    '''
    :param coord: N,2,h,w 原图坐标
    :param des: N,256,h,w
    :param score_map:N,1,h,w
    :return:compose_res N,M,(2+1+256) coord,score,des
    还有索引
    '''
    N, _, h, w = score_map.size()
    score_map = score_map.view(N, h * w,1)
    coord = coord.permute(0, 2, 3, 1).view(N, h * w, -1)
    des = des.permute(0, 2, 3, 1).view(N, h * w, -1)
    relative_coor = relative_coor.permute(0, 2, 3, 1).view(N, h * w, -1)
    
    # 其实应该只定义一次，后面直接调用
    loc_base_coor = torch.arange(w*h,dtype=torch.int).view(1, w*h).expand(N,h*w).unsqueeze(-1).to(coord.device)
    # y_base_coor = torch.arange(h,dtype=torch.int).view(h, 1).expand(N,h,w).unsqueeze(-1)
    # xy_index = torch.cat([x_base_coor,y_base_coor],dim=-1).to(coord.device)
    # xy_index = xy_index.view(N,h*w,2)
    # print(xy_index)
    res = t.cat([coord,relative_coor,score_map,des,loc_base_coor ], axis=-1)
    return res

def nms_fast(in_corners, H, W, dist_thresh):

        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.

        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T): #  N,3
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

def postprocess_training(compose_all_1): 
    compose_all_1 = compose_all_1[(compose_all_1[:, 4] > 0.5) &
                    (compose_all_1[:, 0] > 0) &
                    (compose_all_1[:, 1] > 0) &
                    (compose_all_1[:, 0] < 416 - 0.5) &
                    (compose_all_1[:, 1] < 416 - 0.5)
                    ]
    pts = torch.cat((compose_all_1[:, :2], compose_all_1[:, [4]]), axis=1)  # M ,3 pts+score
    des = compose_all_1[:, 5:-1]  # M,256

    # 在nms之后point的数量会变化，index仍然指向原长度的数组
    pts, index = nms_fast(pts.transpose(0, 1).detach().cpu().numpy(), 416, 416, 4)
    pts = torch.from_numpy(pts.transpose((1, 0)))
    des = des[index]
    loc_index = compose_all_1[:, -1:][index] # 获得了nms之后的pts和des以及对齐的local_index,pts排序为score降序
    return pts,des,loc_index,index

class PostProcess(object):
    def __init__(self, infer):

        self.infer = infer
        self.score_threshold = 0.0
        self.align_corners = False
        self.nms_pad = 4
        # self.opt = opt
    @staticmethod
    def cacluate_pos(pos_map):
        # pos_map N 2 h/8 w/8
        N, _, h, w = pos_map.size()
        x_base_coor = t.arange(w).view(1, 1, 1, w).expand(N, 1, h, w).type_as(pos_map)
        y_base_coor = t.arange(h).view(1, 1, h, 1).expand(N, 1, h, w).type_as(pos_map)
        x_coor = (x_base_coor + pos_map[:, [0], :, :]) * 8
        y_coor = (y_base_coor + pos_map[:, [1], :, :]) * 8
        coord = t.cat((x_coor, y_coor), axis=1)
        return coord  # N,2,h,w 在原图上的坐标

    @staticmethod
    def cacluate_des(coord, des, step,align_corners):
        N, _, h, w = coord.size()
        coord_for_des = t.zeros_like(coord)
        coord_for_des[:, :1, :, :] = coord[:, :1, :, :] / step / w * 2 - 1  # 转换到-1到1
        coord_for_des[:, 1:, :, :] = coord[:, 1:, :, :] / step / h * 2 - 1
        # 这里使用了bilinear进行插值，是否合理？是不是可以在一个3x3等的局部范围内进行加权注意力？
        # 先不使用插值
        # des = F.grid_sample(input=des, grid=coord_for_des.permute(0, 2, 3, 1),align_corners=align_corners)
        return des

    @t.no_grad()
    def compose(self, coord, des, score_map, relative_coor):
        '''

        :param coord: N,2,h,w 原图坐标
        :param des: N,256,h,w
        :param score_map:N,1,h,w
        :return:compose_res N,M,(2+1+256) coord,score,des
        '''
        N, _, h, w = score_map.size()

        score_map = score_map.view(N, h * w,1)
        coord = coord.permute(0, 2, 3, 1).view(N, h * w, -1)
        des = des.permute(0, 2, 3, 1).view(N, h * w, -1)
        des_shape = des.size()[-1]
        relative_coor = relative_coor.permute(0, 2, 3, 1).view(N, h * w, -1)
        


        res = t.cat([coord,relative_coor,score_map,des], axis=-1)
        return res

    def __call__(self, pos_map, pos_map2, des_map, des_map2, score_map, score_map2, chose="singlehead"):

        coord = self.cacluate_pos(pos_map)
        _, _, self.h, self.w = score_map.size()
        des = self.cacluate_des(coord, des_map, 8.0,align_corners=self.align_corners) # N,C,H,W
        """
        # des = des_map
        # des2 = self.cacluate_des(coord+t.tensor([2,2]).type_as(coord).view(1,2,1,1), des_map, 8.0)
        # print("here!!!!!!!!!!!!1")
        # diff = 1 - t.linalg.norm(des2 - des,dim=1) #1,
        # diff[score_map[0]<0.5] = 0
        # diff_a = t.tensor(diff.detach())
        # diff_a[score_map[0]>=0.5] = 1
        # temp = diff[score_map[0]>0.5]
        # print("max:",t.max(temp))
        # print("min:",t.min(temp))
        # print("mean:",t.mean(temp))
        # diff = t.exp(diff[0]*7).detach().cpu()
        # diff_a = t.exp(diff_a[0]*7).detach().cpu()
        # diff1 = cv2.resize(diff.numpy(),dsize=None,fx=8,fy=8)
        # diff_a = cv2.resize(diff_a.numpy(),dsize=None,fx=8,fy=8)
        # img = img[0][0].cpu().numpy()
        # # diff = np.concatenate( (img,),axis=0)
        # diff_a = img.astype(np.float)+diff_a.astype(np.float)
        # diff_a = diff_a/np.max(diff_a) *255
        # res = img.astype(np.float)+diff1.astype(np.float)
        # res = res / np.max(res) * 255
        # res= np.concatenate((res,img),axis=0)
        # res = np.concatenate([res,diff_a],axis=0)
        # cv2.imwrite(f"./desdiff/{i:06d}.jpg",res)
        # print("diff:" , diff[0]  )
        """
        if self.infer:
            compose_all1 = self.compose(coord, des, score_map, pos_map)[0]  # H*W,261
            self.h = 8 * self.h
            self.w = 8 * self.w
            if chose == "big" or chose == "singlehead":
                compose_all = compose_all1
            elif chose == "small":
                coord2 = self.cacluate_pos(pos_map2) * 2
                des2 = self.cacluate_des(coord2, des_map2, 16.0,self.opt.align_corners)
                compose_all2 = self.compose(coord2, des2, score_map2, pos_map2)[0]  # H*W,261
                compose_all = compose_all2
            elif chose == "all":
                coord2 = self.cacluate_pos(pos_map2) * 2
                des2 = self.cacluate_des(coord2, des_map2, 16.0)
                compose_all2 = self.compose(coord2, des2, score_map2, pos_map2)[0]  # H*W,261
                compose_all = t.cat([compose_all1, compose_all2], 0)
            else:
                print("chose what?")
                exit()
            compose_all = compose_all[(compose_all[:, 4] > self.score_threshold) &
                                      (compose_all[:, 0] > 0) &
                                      (compose_all[:, 1] > 0) &
                                      (compose_all[:, 0] < self.w - 0.5) &
                                      (compose_all[:, 1] < self.h - 0.5)
                                      ]  # M,261
            # print("point_thres:",self.t/hreshold)
            
            # print(compose_all.size())
            # exit()
            pts = t.cat((compose_all[:, :2], compose_all[:, [4]]), axis=1)  # M ,3 pts+score
            des = compose_all[:, 5:].cpu()  # M,256
            pts, index = self.nms_fast(pts.transpose(0, 1).cpu().numpy(), self.h, self.w, self.nms_pad)
            pts = t.from_numpy(pts.transpose((1, 0)))
            des = des[index]
            return pts, des  # tensor # M,3  // M ,256
        else:
            if chose == "all":
                coord2 = self.cacluate_pos(pos_map2) * 2
                self.h = 8 * self.h
                self.w = 8 * self.w
                des2 = self.cacluate_des(coord2, des_map2, 16.0)
                return t.cat([self.reshape(score_map), self.reshape(score_map2)], 1), \
                       t.cat([self.reshape(coord), self.reshape(coord2)], 1), \
                       t.cat([self.reshape(pos_map), self.reshape(pos_map2)], 1), \
                       t.cat([self.reshape(des), self.reshape(des2)], 1)  # compose_res N,hw,(2+1+256) coord,score,des
            if chose == "singlehead":
                # print(score_map.shape)
                # print(self.reshape(score_map).shape)
                # print(self.reshape(coord).shape)
                # print(self.reshape(pos_map).shape)

                # print(self.reshape(des).shape)
                return self.reshape(score_map), self.reshape(coord), self.reshape(pos_map), self.reshape(des)
                # N,_, H, W = score_map.size()
                #  return self.nms_fast_torch(H,W,8,score_map,coord,pos_map,des)

    def reshape(self, a):
        N, _, h, w = a.size()
        return a.permute(0, 2, 3, 1).view(N, h * w, -1)

    def nms_fast(self, in_corners, H, W, dist_thresh):

        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.

        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T): #  N,3
            grid[rcorners[1, i], rcorners[0, i]] = 1 # rcorners[1, i], rcorners[0, i]: rcorners的第i个点的y,x坐标
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds
    def nms_fast_torch(self, H, W, dist_thresh,s,c,p,d):
        """
        :param H:
        :param W:
        :param dist_thresh:
        :param s: N,1,H,W
        :param c: N,2,H,W
        :param p: N,2,H,W
        :param d: N,256,H,W
        :return:
        """
        sl = []
        cl = []
        pl = []
        dl = []
        H,W = int(H),int(W)
        for i in range(s.size()[0]):
            si = s[i].view(H*W)
            ci = c[i].view(2,H*W)
            pi = p[i].view(2,H*W)
            di = d[i].view(-1,H*W)
            grid = np.zeros((H*8+1, W*8+1)).astype(int)  # Track NMS data.
            inds = np.zeros((H*8+1, W*8+1)).astype(int)  # Store indices of points.
            # Sort by confidence and round to nearest int.
            sorts , inds1 = t.sort(-si)
            ci = ci[:,inds1]
            rcorners = ci.round().long()
            rcorners_np = rcorners.cpu().numpy()# N,2
            pi = pi[:,inds1]
            si = si[inds1]
            di = di[:,inds1]
            grid[rcorners_np[1],rcorners_np[0]] = 1
            inds[rcorners_np[1],rcorners_np[0]] = np.arange(rcorners_np.shape[1])
            pad = dist_thresh
            grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
            keepy,keepx = fast_numba_nms(grid,pad,rcorners_np)
            inds_keep = inds[keepy, keepx]
            inds_keep = t.from_numpy(inds_keep).long()
            sl.append(si[inds_keep][:,None])
            cl.append(ci[:,inds_keep].transpose(1,0))
            pl.append(pi[:,inds_keep].transpose(1,0))
            dl.append(di[:,inds_keep].transpose(1,0))

        return [sl,cl,pl,dl]

def match_inputs_(kpts1, desc1,kpts2, desc2,match_threshold=0.0):
    kpts1 = kpts1.cpu().data.numpy()
    kpts2 = kpts2.cpu().data.numpy()
    
    # NN Match
    match_ids, scores = mutual_nn_matching(desc1, desc2, threshold=match_threshold)
    p1s = kpts1[match_ids[:, 0], :2]
    p2s = kpts2[match_ids[:, 1], :2]
    matches = np.concatenate([p1s, p2s], axis=1)
    return matches, kpts1, kpts2, scores

def mutual_nn_matching(desc1, desc2, threshold=None):
    if isinstance(desc1, np.ndarray):
        desc1 = torch.from_numpy(desc1)
        desc2 = torch.from_numpy(desc2)
    matches, scores = mutual_nn_matching_torch_training(desc1, desc2, threshold=threshold)
    return matches.cpu().numpy(), scores.cpu().numpy()

def mutual_nn_matching_torch_training(desc1, desc2, threshold=None):
    if len(desc1) == 0 or len(desc2) == 0:
        return torch.empty((0, 2), dtype=torch.int64), torch.empty((0, 2), dtype=torch.int64)

    device = desc1.device
    desc1 = desc1 / desc1.norm(dim=1, keepdim=True)
    desc2 = desc2 / desc2.norm(dim=1, keepdim=True)
    similarity = torch.einsum('id, jd->ij', desc1, desc2)

    
    nn12 = similarity.max(dim=1)[1]
    nn21 = similarity.max(dim=0)[1]
    ids1 = torch.arange(0, similarity.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    scores = similarity.max(dim=1)[0][mask]    
    if threshold:
        mask = scores > threshold
        matches = matches[mask]    
        scores = scores[mask]
    return matches, scores

def compute_dist(matches,H_gt=None,img=None):
    # Inference with SuperGlue and get prediction
    data = {}
    if H_gt is None:
        H_gt = np.identity(3)
    try:
        H_pred, inliers = pydegensac.findHomography(matches[:, :2], matches[:, 2:4], 1,0.99,2000)
    except:
        corner_dist = 100
        return corner_dist,0,0,3,0,False
    if H_pred is None:
        corner_dist = 100
        return corner_dist,0,0,3,0,False # return corner_dist,0,0,3,_,False
    w,h = img
    corners = np.array([[0, 0, 1],
                        [0, w - 1, 1],
                        [h - 1, 0, 1],
                        [h - 1, w - 1, 1]])
    real_warped_corners = np.dot(corners, np.transpose(H_gt))
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    try:
        warped_corners = np.dot(corners, np.transpose(H_pred))
    except:
        print(H_pred)
    # print(real_warped_corners.shape)
    if real_warped_corners.shape==(4,2,1):
        real_warped_corners = real_warped_corners[:,:,0]
    # print(real_warped_corners.shape)
    # exit()
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    corner_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    data['homography'] = H_gt
    data['pts'] = matches[:, :2]
    data['warped_pts'] = matches[:, 2:4]
    rep, local_err = compute_rep_loc(data,img=img)  
    return corner_dist,inliers,rep, local_err,H_pred,True

def compute_rep_loc(data, keep_k_points=300,distance_thresh=3,img=None):
    
    localization_err = -1
    repeatability = []
    N1s = []
    N2s = []
    #print("----------------------thresh:",distance_thresh,"--------------------------")
    h,w = img
    shape = (h,w)
    H = data['homography'].copy()

    keypoints = data['pts'].copy()
    warped_keypoints = data['warped_pts'].copy()

    warped_keypoints = keep_true_keypoints(warped_keypoints, np.linalg.inv(H),shape)

    # Warp the original keypoints with the true homography
    true_warped_keypoints = keypoints
    # true_warped_keypoints[:,:2] = warp_keypoints(keypoints[:, [1, 0]], H)
    true_warped_keypoints[:,:2] = warp_keypoints(keypoints[:, :2], H) # make sure the input fits the (x,y)
    # true_warped_keypoints = np.stack([true_warped_keypoints[:, 1],
    #                                   true_warped_keypoints[:, 0],
    #                                   prob], axis=-1)
    true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)

    # Keep only the keep_k_points best predictions
    warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
    warped_keypoints_ = warped_keypoints.copy()
    true_warped_keypoints = select_k_best(true_warped_keypoints, keep_k_points)

    # Compute the repeatability
    N1 = true_warped_keypoints.shape[0]
    if N1==0:
        print("det: not enough pts")
        return 0, 3
    N2 = warped_keypoints.shape[0]
    N1s.append(N1)
    N2s.append(N2)
    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
    warped_keypoints = np.expand_dims(warped_keypoints, 0)
    # shapes are broadcasted to N1 x N2 x 2:
    norm = np.linalg.norm(true_warped_keypoints - warped_keypoints,
                          ord=None, axis=2)
    
    count1 = 0
    count2 = 0
    local_err1, local_err2 = None, None
    if N2 != 0:
        min1 = np.min(norm, axis=1)
        count1 = np.sum(min1 <= distance_thresh)
        local_err1 = min1[min1 <= distance_thresh]
    if N1 != 0:
        min2 = np.min(norm, axis=0)
        count2 = np.sum(min2 <= distance_thresh)
        local_err2 = min2[min2 <= distance_thresh]

    if N1 + N2 > 0:
        # repeatability.append((count1 + count2) / (N1 + N2))
        repeatability = (count1 + count2) / (N1 + N2)
    if count1 + count2 > 0:
        localization_err = 0
        if local_err1 is not None:
            localization_err += (local_err1.sum())/ (count1 + count2)
        if local_err2 is not None:
            localization_err += (local_err2.sum())/ (count1 + count2)
    else:
        repeatability = 0
        localization_err = 3
    # return np.mean(repeatability)
    return repeatability, localization_err

def keep_true_keypoints(points, H, shape):
    """ Keep only the points whose warped coordinates by H
    are still inside shape. """
    """
    input:
        points: numpy (N, (x,y))
        shape: (h, w)
    return:
        points: numpy (N, (x,y))
    """
    # warped_points = warp_keypoints(points[:, [1, 0]], H)
    warped_points = warp_keypoints(points[:, [0, 1]], H)
    # warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
    mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[1]) &\
            (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[0])
    mask = np.squeeze(mask)
    return points[mask, :]
def filter_keypoints(points, shape):
    """ Keep only the points whose coordinates are
    inside the dimensions of shape. """
    """
    points:
        numpy (N, (x,y))
    shape:
        (h, w)
    """
    mask = (points[:, 0] >= 0) & (points[:, 0] < shape[1]) &\
            (points[:, 1] >= 0) & (points[:, 1] < shape[0])
    return points[mask, :]

def warp_keypoints(keypoints, H):
    """
    :param keypoints:
    points:
        numpy (N, (x,y))
    :param H:
    :return:
    """
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))],
                                        axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))
    return warped_points[:, :2] / warped_points[:, 2:]

def select_k_best(points, k):
    """ Select the k most probable points (and strip their proba).
    points has shape (num_points, 3) where the last coordinate is the proba. """
    sorted_prob = points
    # false
    if points.shape[1] > 2:
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        sorted_prob = sorted_prob[-start:, :]
    return sorted_prob

def keypoints_from_match(matches,inliers=[],ran=False):
    num_matches = len(matches)
    # print(inliers)
    matches_draw=[]
    if ran:
        
        for i in range(num_matches):
            if inliers[i]:
                matches_draw.append([cv2.DMatch(i, i, 0) for i in range(num_matches)] )
    else:
        matches_draw = [[cv2.DMatch(i, i, 0)] for i in range(num_matches) ]
    match_1 = matches[:,:2]
    match_2 = matches[:,2:]
    # 创建一个空列表，用于存储 Keypoint 实例
    keypoints_list = []
    # 遍历 PyTorch 张量中的每个点，并创建 Keypoint 实例并添加到列表中
    for point in match_1:
        x, y = point.tolist()
        kp = cv2.KeyPoint(x, y, size=1)  # 这里可以根据需要调整 KeyPoint 的其他属性
        keypoints_list.append(kp)
    keypoints_1 = tuple(keypoints_list)


    keypoints_list = []
    for point in match_2:
        x, y = point.tolist()
        kp = cv2.KeyPoint(x, y, size=1)  # 这里可以根据需要调整 KeyPoint 的其他属性
        keypoints_list.append(kp)
    keypoints_2 = tuple(keypoints_list) 
    return keypoints_1,keypoints_2,matches_draw


def keypoints_mask_from_match(matches,inliers=[],ran=False):
    num_matches = len(matches)
    # print(inliers)
    matches_draw=[]
    # if ran:
        
    #     for i in range(num_matches):
    #         if inliers[i]:
    #             matches_draw.append([cv2.DMatch(i, i, 0) for i in range(num_matches)] )
    # else:
    matches_draw = [[cv2.DMatch(i, i, 0)] for i in range(num_matches) ]
    match_1 = matches[:,:2]
    match_2 = matches[:,2:]
    # 创建一个空列表，用于存储 Keypoint 实例
    keypoints_list = []
    # 遍历 PyTorch 张量中的每个点，并创建 Keypoint 实例并添加到列表中
    for point in match_1:
        x, y = point.tolist()
        kp = cv2.KeyPoint(x, y, size=1)  # 这里可以根据需要调整 KeyPoint 的其他属性
        keypoints_list.append(kp)
    keypoints_1 = tuple(keypoints_list)


    keypoints_list = []
    for point in match_2:
        x, y = point.tolist()
        kp = cv2.KeyPoint(x, y, size=1)  # 这里可以根据需要调整 KeyPoint 的其他属性
        keypoints_list.append(kp)
    keypoints_2 = tuple(keypoints_list) 
    mask = np.uint8((inliers + 0)[:,np.newaxis])
    return keypoints_1,keypoints_2,matches_draw,mask


def eval_summary_homography(dists,thres=[1, 3, 5, 10, 20,30,40, 50, 60, 70, 80,90,100]):
    correct = np.mean(
        [[float(dist <= t) for t in thres] for dist in dists], axis=0)
    print(correct)
    return correct,thres

# def draw_match_image(model,feat,feat_h,img,img_h,match_threshold,img_size=[416,416]):
#     res_h = model.module.netP.forward(feat_h)
#     res = model.module.netP.forward(feat)
#     score_h,pos_h,des_h = res_h[0], res_h[1], res_h[2]
#     score,pos,des = res[0], res[1], res[2]
#     pts,desc = model.module.postprocess_infer(pos, pos, des, des, score, score)
#     pts_h,desc_h = model.module.postprocess_infer(pos_h, pos_h, des_h, des_h, score_h,score_h)
#     pts_h = pts_h[:,:2]
#     pts = pts[:,:2]  
#     matches,_,_,_ = match_inputs_(pts, desc,pts_h, desc_h,match_threshold=match_threshold)
#     mat_homo = mat_homo.cpu().numpy()
#     dist,inliers,rep,loc,H_pred,if_Work = compute_dist(matches,H_gt=mat_homo,img=img_size)
#     keypoints1,keypoints2,matches_draw = keypoints_from_match(matches)
#     try:### mask = np.uint8((inliers + 0)[:,np.newaxis])TypeError: 'int' object is not subscriptable
#         keypoints1,keypoints2,matches_draw,mask = keypoints_mask_from_match(matches,inliers)
#         output_img_ran = cv2.drawMatchesKnn(img, keypoints1, img_h, keypoints2, matches_draw, None, matchColor=(0, 255, 0), 
#             singlePointColor=(0,0,255), matchesMask=mask , flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
#     except:
#         print("failed to draw matched images")
#         output_img_ran = img
#     return output_img_ran,dist,rep,loc
    
# 实际只能对一个batch做
def kpt_from_dict(pred_dict,score_thres = 0.7):
    keypoints_list = []
    descriptors_list = []
    # 遍历 PyTorch 张量中的每个点，并创建 Keypoint 实例并添加到列表中
    for j in pred_dict.keys():
    # cv2.imwrite(os.path.join(str(data_dir), img_name+suffix), img0[j])
        s1 = pred_dict[j]['s1']
        loc = np.where(s1 > score_thres)
        p1 = pred_dict[j]['p1'][loc]
        d1 = pred_dict[j]['d1'][loc]
        s1 = s1[loc]
        # kp = cv2.KeyPoint(x, y, size=1)
        # keypoints_list.append(kp)
    for i in range(p1.shape[0]):
        pos = p1[i]
        desc = d1[i]
        # cv2.circle(img, (int(pos[0]), int(pos[1])), 1, (0, 0, 255), 1)
        kp = cv2.KeyPoint(pos[0], pos[1], size=1)
        keypoints_list.append(kp)
        descriptors_list.append(np.array(desc))
    keypoints = tuple(keypoints_list)
    descriptors = np.array(descriptors_list)
    return keypoints,descriptors

def metric_from_dict_cv(pred_dict_1,pred_dict_2,H_gt,img_shape=[416,416],score_thres=0.5):
    # return: corner_dist,inliers,rep, local_err,H_pred,True
    # 这里不太确定H的计算关系对不对，需要进一步验证 
    data={}
    keypoints_1,descriptors_1=kpt_from_dict(pred_dict_1,score_thres)
    keypoints_2,descriptors_2=kpt_from_dict(pred_dict_2,score_thres)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    try:
        
        # output_img = cv2.drawMatches(img, keypoints_1, dst_img, keypoints_2, matches, None, matchColor=(0, 255, 0),  
        #         singlePointColor=(0,0,255), matchesMask=None , flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) # matches
        
        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H_pred, inliers = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC)
        w,h = img_shape
        corners = np.array([[0, 0, 1],
                            [0, w - 1, 1],
                            [h - 1, 0, 1],
                            [h - 1, w - 1, 1]])
        real_warped_corners = np.dot(corners, np.transpose(H_gt))
        real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
        try:
            warped_corners = np.dot(corners, np.transpose(H_pred))
        except:
            print(H_pred)
        # print(real_warped_corners.shape)
        if real_warped_corners.shape==(4,2,1):
            real_warped_corners = real_warped_corners[:,:,0]
        # print(real_warped_corners.shape)
        # exit()
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
        corner_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
        data['homography'] = H_gt
        # 这里应该是对应两边匹配的点的位置
        # data['pts'] = matches[:, :2]
        # data['warped_pts'] = matches[:, 2:4]
        data['pts'] = src_pts.squeeze()
        data['warped_pts'] = dst_pts.squeeze()      
        rep, local_err = compute_rep_loc(data,img=img_shape)  
        return corner_dist,inliers,rep, local_err,H_pred,True
        inliers = inliers.reshape(-1)
        # output_img_ran = cv2.drawMatches(img, keypoints_1, dst_img, keypoints_2, matches, None, matchColor=(0, 255, 0), 
        #         singlePointColor=(0,0,255), matchesMask=inliers , flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    except:
        corner_dist,inliers,rep, local_err,H_pred=100,None,0,3,True
        
        return corner_dist,inliers,rep, local_err,H_pred,True



def dist_to_HA(DIST,HA_1,HA_3,HA_5,HA_10,HA_20):
    if DIST<=1:
        HA_1+=1
    if DIST<=3:
        HA_3+=1
    if DIST<=5:
        HA_5+=1
    if DIST<=10:
        HA_10+=1
    if DIST<=20:
        HA_20+=1
    return HA_1,HA_3,HA_5,HA_10,HA_20
def draw_from_dict(pred_dict_1,pred_dict_2,img,dst_img,score_thres=0.5):

    keypoints_1,descriptors_1=kpt_from_dict(pred_dict_1,score_thres)
    keypoints_2,descriptors_2=kpt_from_dict(pred_dict_2,score_thres)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    matches_idx = np.array([m.queryIdx for m in matches])
    
    # good_matches = []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good_matches.append(m)
            
            
    # breakpoint()
    try:
        output_img = cv2.drawMatches(img, keypoints_1, dst_img, keypoints_2, matches, None, matchColor=(0, 255, 0),  
                singlePointColor=(0,0,255), matchesMask=None , flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) # matches
        
        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
        H, inliers = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC)

        inliers = inliers.reshape(-1)
        output_img_ran = cv2.drawMatches(img, keypoints_1, dst_img, keypoints_2, matches, None, matchColor=(0, 255, 0), 
                singlePointColor=(0,0,255), matchesMask=inliers , flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    except:
        img = cv2.drawKeypoints(img, keypoints_1,None, color=(0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        dst_img = cv2.drawKeypoints(dst_img, keypoints_2,None, color=(0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        output_img = np.concatenate([img, dst_img], axis=1)
        output_img_ran = output_img
    return output_img,output_img_ran

def draw_point_image(pred_dict,img,score_thres = 0.7):
    # print(pred_dict)
    # input()
    for j in pred_dict.keys():
    # cv2.imwrite(os.path.join(str(data_dir), img_name+suffix), img0[j])
        s1 = pred_dict[j]['s1']
        loc = np.where(s1 > score_thres)
        p1 = pred_dict[j]['p1'][loc]
        d1 = pred_dict[j]['d1'][loc]
        s1 = s1[loc]
    # print(p1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i in range(p1.shape[0]):
        pos = p1[i]
        score = s1[i]
        # score = (score-0.5)*2
        # print(score)
        color = (0, 0, int(255 * score))
        # color = (int(255 * score), int(255 * score), int(255 * score))
        # cv2.circle(img, (int(pos[0]), int(pos[1])), 1, color, 1)
        cv2.circle(img, (int(pos[0]), int(pos[1])), 4, color, 1)

    return img
    # cv2.imwrite(os.path.join(eval_out_dir,name + '.jpg'), img)

def draw_point_feat(img_tensor):
    # print(pred_dict)
    # input()
    # print(img_tensor)
    img_tensor = img_tensor.cpu().numpy()[0]
    img_tensor *=255
    img_tensor = np.clip(img_tensor,0,255).astype(np.uint8)

    img_tensor = cv2.resize(img_tensor,(416,416),cv2.INTER_NEAREST)
    heat_map = cv2.applyColorMap(img_tensor, cv2.COLORMAP_JET)
    
    
    
    # img = util.tensor2im(img_tensor.detach())

    return heat_map

def draw_desc_feat(desc,k=3,center=True):# (C,H,W)
    
    X = desc.detach().unsqueeze(0) # (N=1,C,H,W)
    B, C,H,W = X.shape
    X = X.permute(0, 2, 3, 1)  # BxHxWxC
    X = X.reshape(B, H * W, C)
    U, S, V = torch.pca_lowrank(X, center=center)
    # print(V.shape)
    Y = torch.bmm(X, V[:, :, :k])
    
    Y = Y.reshape(B, H, W, k)
    Y = Y.squeeze(0) # HxWxC
    Y = Y.cpu().numpy() # HxWxC
    # print(Y.shape)
    # desc_mean = torch.mean(desc.detach(),dim=0)
    
    Y *=255
    Y = np.clip(Y,0,255).astype(np.uint8)

    Y = cv2.resize(Y,(416,416),cv2.INTER_NEAREST)
    # print(Y.shape)
    # heat_map = cv2.applyColorMap(Y, cv2.COLORMAP_JET)
    return Y

# def PCA_svd(X, k, center=True):
#   n = X.size()[0]
#   ones = torch.ones(n).view([n,1])
#   h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
#   H = torch.eye(n) - h
# #   H = H.cuda()
#   X_center =  torch.mm(H.double(), X.double())
#   u, s, v = torch.svd(X_center)
#   components  = v[:k].t()
#   #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
#   return components
def PCA_Batch_Feat(X, k, center=True):
    """
    param X: BxCxHxW
    param k: scalar
    return:
    """
    B, C, H, W = X.shape
    X = X.permute(0, 2, 3, 1)  # BxHxWxC
    X = X.reshape(B, H * W, C)
    U, S, V = torch.pca_lowrank(X, center=center)
    Y = torch.bmm(X, V[:, :, :k])
    Y = Y.reshape(B, H, W, k)
    Y = Y.permute(0, 3, 1, 2)  # BxHxWxk
    return Y


def draw_evrthing(pred_dict,pred_dict_H,image,image_H,score,score_H,pos,pos_H,desc,desc_H):
    # print(pred_dict)
    # print(image.shape)
    # print(image.dtype)
    image = np.squeeze(image)
    image = np.transpose(image,(1,2,0))
    # print("image:",image.shape)
    image_H = np.squeeze(image_H)
    image_H = np.transpose(image_H,(1,2,0))
    image_match,image_match_Ran = draw_from_dict(pred_dict,pred_dict_H,
                    image,image_H, score_thres=0)
    image_point = draw_point_image(pred_dict,image,score_thres=0)
    image_point_H = draw_point_image(pred_dict_H,image_H,score_thres=0)
    image_point_both = np.concatenate([image_point, image_point_H], axis=1)
    # print("image_match:",image_match.shape)
    image_point_score = draw_point_feat(score.detach())
    image_point_score_H = draw_point_feat(score_H.detach())
    # print("image_point_score:",image_point_score.shape)
    image_point_score_both = np.concatenate([image_point_score, image_point_score_H], axis=1)
    image_both = np.concatenate([image, image_H], axis=1)
    # print(image_both.shape)
    
    image_point_mix = cv2.addWeighted(image_both, 0.5, image_point_score_both, 0.5, 0)
    # image_point_mix = 0.5*image_both+0.5*image_point_feat_both 
    image_point_pos_x = draw_point_feat(pos[0].detach().unsqueeze(0))
    image_point_pos_x_H = draw_point_feat(pos_H[0].detach().unsqueeze(0))
    image_point_pos_x_both = np.concatenate([image_point_pos_x, image_point_pos_x_H], axis=1)
    image_point_pos_y = draw_point_feat(pos[1].detach().unsqueeze(0))
    image_point_pos_y_H = draw_point_feat(pos_H[1].detach().unsqueeze(0))
    
    image_point_pos_y_both = np.concatenate([image_point_pos_y, image_point_pos_y_H], axis=1)


    image_desc = draw_desc_feat(desc)
    image_desc_H = draw_desc_feat(desc_H)

    # print(image_point_score_both.shape)
    # print(image_point_pos_x_both.shape)# (416, 832, 3)

    return {"image_match":image_match,"image_match_Ran":image_match_Ran,"image_point_both":image_point_both,"image_point_mix":image_point_mix,
            "image_point_score_both":image_point_score_both,"image_point_pos_x_both":image_point_pos_x_both,"image_point_pos_y_both":image_point_pos_y_both,
            "image_desc":image_desc,"image_desc_H":image_desc_H}

postprocess_infer = PostProcess(infer=True)

def draw_match_image(score,pos,des,score_h,pos_h,des_h,img,img_h,mat_homo,score_threshold=0,match_threshold=0,img_size=[416,416],nms=4):

    global postprocess_infer
    postprocess_infer.score_threshold = score_threshold
    postprocess_infer.nms_pad = nms
    
    try:
        pts,desc = postprocess_infer(pos, pos, des, des, score, score)
        pts_h,desc_h = postprocess_infer(pos_h, pos_h, des_h, des_h, score_h,score_h)
        pts_h = pts_h[:,:2]
        pts = pts[:,:2]  
        matches,_,_,_ = match_inputs_(pts, desc,pts_h, desc_h,match_threshold=match_threshold)
        mat_homo = mat_homo.cpu().numpy()
        dist,inliers,rep,loc,H_pred,if_Work = compute_dist(matches,H_gt=mat_homo,img=img_size)
        keypoints1,keypoints2,matches_draw = keypoints_from_match(matches)
    ### mask = np.uint8((inliers + 0)[:,np.newaxis])TypeError: 'int' object is not subscriptable
        keypoints1,keypoints2,matches_draw,mask = keypoints_mask_from_match(matches,inliers)
        
        if len(img.shape)==4:
            n_img,c_img,h_img,w_img = img.shape
            img = np.squeeze(img,axis=0).transpose(1,2,0)
            img_h = np.squeeze(img_h,axis=0).transpose(1,2,0)
        output_img_ran = cv2.drawMatchesKnn(img, keypoints1, img_h, keypoints2, matches_draw, None, matchColor=(0, 255, 0), 
            singlePointColor=(0,0,255), matchesMask=mask , flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        # breakpoint()
        return output_img_ran,dist,rep,loc
    except:
        print("failed to draw matched images")
        output_img_ran = img
        return output_img_ran,100,0,3
    
def draw_match_image_double_output(score,pos,des,score_h,pos_h,des_h,img,img_h,mat_homo,score_threshold=0,match_threshold=0,img_size=[416,416]):

    global postprocess_infer
    postprocess_infer.score_threshold = score_threshold
    try:
        if score.shape[0]%2:
            assert("number of double output channel should be even")
        C = int(score.shape[0]/2)
        
        score_rgb = score[:C,:,:]
        score_ir = score[C:,:,:]
        score_h_rgb = score_h[:C,:,:]
        score_h_ir = score_h[C:,:,:]
        C = int(pos.shape[0]/2)
        pos_rgb = pos[:C,:,:]
        pos_ir = pos[C:,:,:]
        pos_h_rgb = pos_h[:C,:,:]
        pos_h_ir = pos_h[C:,:,:]
        C = int(des.shape[0]/2)
        des_rgb = des[:C,:,:]
        des_ir = des[C:,:,:]
        des_h_rgb = des_h[:C,:,:]
        des_h_ir = des_h[C:,:,:]
        



        pts_rgb,desc_rgb = postprocess_infer(pos_rgb, pos_rgb, des_rgb, des_rgb, score_rgb, score_rgb)
        pts_h_rgb,desc_h_rgb = postprocess_infer(pos_h_rgb, pos_h_rgb, des_h_rgb, des_h_rgb, score_h_rgb,score_h_rgb)
        pts_ir,desc_ir = postprocess_infer(pos_ir, pos_ir, des_ir, des_ir, score_ir, score_ir)
        pts_h_ir,desc_h_ir = postprocess_infer(pos_h_ir, pos_h_ir, des_h_ir, des_h_ir, score_h_ir,score_h_ir)


        
        pts_h_ir = pts_h_ir[:,:2]
        pts_ir = pts_ir[:,:2]  
        pts_h_rgb = pts_h_rgb[:,:2]
        pts_rgb = pts_rgb[:,:2]  

        pts=pts_rgb+pts_ir
        pts_h = pts_h_rgb+pts_h_ir
        desc = desc_rgb+desc_ir
        desc_h = desc_h_rgb+desc_h_ir
        

        matches,_,_,_ = match_inputs_(pts, desc,pts_h, desc_h,match_threshold=match_threshold)
        mat_homo = mat_homo.cpu().numpy()
        dist,inliers,rep,loc,H_pred,if_Work = compute_dist(matches,H_gt=mat_homo,img=img_size)
        keypoints1,keypoints2,matches_draw = keypoints_from_match(matches)
    ### mask = np.uint8((inliers + 0)[:,np.newaxis])TypeError: 'int' object is not subscriptable
        keypoints1,keypoints2,matches_draw,mask = keypoints_mask_from_match(matches,inliers)
        
        if len(img.shape)==4:
            n_img,c_img,h_img,w_img = img.shape
            img = np.squeeze(img,axis=0).transpose(1,2,0)
            img_h = np.squeeze(img_h,axis=0).transpose(1,2,0)
        output_img_ran = cv2.drawMatchesKnn(img, keypoints1, img_h, keypoints2, matches_draw, None, matchColor=(0, 255, 0), 
            singlePointColor=(0,0,255), matchesMask=mask , flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        # breakpoint()
        return output_img_ran,dist,rep,loc
    except:
        print("failed to draw matched images")
        output_img_ran = img
        return output_img_ran,100,0,3
    
if __name__=="__main__":
    desc=torch.randn((128,52,52))
    desc_map = draw_desc_feat(desc)
    # desc_PCA = PCA_svd(desc,3)
    print(desc_map.shape)
    # print(desc_PCA.shape)

