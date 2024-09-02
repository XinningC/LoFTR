# 解决D2Net数据集unavailable的问题
# https://github.com/zju3dv/LoFTR/issues/276
# 修改了indices中的npz文件，另行存储在..._no_sfm目录
import numpy as np
from numpy import load
import os
import tqdm
#change scene_info_0
# directory_npz = './data/rgb_ir/index/train'
# directory_rgb = './data/rgb_ir/train_A'
# directory_ir = './data/rgb_ir/train_B'
directory_npz = './data/pir_ir/index/train'
directory_rgb = './data/pir_ir/train_A'
directory_ir = './data/pir_ir/train_B'

data = {}
data["rgb_image_paths"]=[]
data["ir_image_paths"]=[]
data["pair_infos"]=[]
for filename in tqdm.tqdm(os.listdir(directory_rgb)):
    # f_npz = os.path.join(directory,filename)
    # data = load(f_npz, allow_pickle = True)
    
    rgb_path = os.path.join(directory_rgb,filename)
    ir_path = os.path.join(directory_ir,filename)
    data["rgb_image_paths"].append(rgb_path)
    data["ir_image_paths"].append(ir_path)
    
    data["pair_infos"].append(filename)

    
    # data["rgb_image_paths"]=[]
    # for count, image_path in enumerate(data['image_paths']):
    #     if image_path is not None:
    #         if 'Undistorted_SfM' in image_path:
    #             data['image_paths'][count] = data['depth_paths'][count].replace('depths', 'imgs').replace('h5', 'jpg')
    
    # data['pair_infos'] = np.asarray(data['pair_infos'], dtype=object)
new_file = os.path.join(directory_npz,"indices.npz")
np.savez(new_file, **data)
print("Saved to ", new_file)



# directory_npz = './data/rgb_ir/index/eval'
# directory_rgb = './data/rgb_ir/eval/train_A'
# directory_ir = './data/rgb_ir/eval/train_B'

directory_npz = './data/pir_ir/index/eval'
directory_rgb = './data/pir_ir/eval/train_A'
directory_ir = './data/pir_ir/eval/train_B'
data = {}
data["rgb_image_paths"]=[]
data["ir_image_paths"]=[]
data["pair_infos"]=[]

for filename in tqdm.tqdm(os.listdir(directory_rgb)):
    # f_npz = os.path.join(directory,filename)
    # data = load(f_npz, allow_pickle = True)
    
    rgb_path = os.path.join(directory_rgb,filename)
    ir_path = os.path.join(directory_ir,filename)
    data["rgb_image_paths"].append(rgb_path)
    data["ir_image_paths"].append(ir_path)
    
    data["pair_infos"].append(filename)

    
    # data["rgb_image_paths"]=[]
    # for count, image_path in enumerate(data['image_paths']):
    #     if image_path is not None:
    #         if 'Undistorted_SfM' in image_path:
    #             data['image_paths'][count] = data['depth_paths'][count].replace('depths', 'imgs').replace('h5', 'jpg')
    
    # data['pair_infos'] = np.asarray(data['pair_infos'], dtype=object)
new_file = os.path.join(directory_npz,"indices.npz")
np.savez(new_file, **data)
print("Saved to ", new_file)











#change scene_info_val_1500
# directory = './data/megadepth/index/scene_info_val_1500'

# for filename in tqdm.tqdm(os.listdir(directory)):
#     f_npz = os.path.join(directory,filename)
#     data = load(f_npz, allow_pickle = True)
#     for count, image_path in enumerate(data['image_paths']):
#         if image_path is not None:
#             if 'Undistorted_SfM' in image_path:
#                 data['image_paths'][count] = data['depth_paths'][count].replace('depths', 'imgs').replace('h5', 'jpg')
    
#     data['pair_infos'] = np.asarray(data['pair_infos'], dtype=object)
#     new_file = './data/megadepth/index/scene_info_val_1500_no_sfm/' + filename
#     np.savez(new_file, **data)
#     print("Saved to ", new_file)

# # Then also run the following script to make sure all images have the ending 'jpg' 
# # (there are some hidden JPG and pngs in the dataset )
# # 修改了图像文件的后缀名到'jpg'



# import os
# from PIL import Image

# root_directory = '/mnt/hdd/chuxinning/MegaDepth/phoenix/S6/zl548/MegaDepth_v1'

# for folder in tqdm.tqdm(os.listdir(root_directory)):
#     four_digit_directory = os.path.join(root_directory,folder)
#     for dense_folder in os.listdir(four_digit_directory):
#         image_directory =  os.path.join(four_digit_directory,dense_folder,'imgs')
#         for image in os.listdir(image_directory):
#             if 'JPG' in image:
#                 new_name = image.replace('JPG', 'jpg')
#                 old_path = os.path.join(image_directory, image)
#                 new_path = os.path.join(image_directory, new_name)
#                 os.rename(old_path, new_path)
#             if 'png' in image:
#                 new_name = image.replace('png', 'jpg')
#                 old_path = os.path.join(image_directory, image)
#                 new_path = os.path.join(image_directory, new_name)
#                 png_img = Image.open(old_path)
#                 png_img.save(new_path)



# 还需要修改如下内容：

#  LoFTR/configs/data/megadepth_trainval_640.py:
# cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7_no_sfm"
# cfg.DATASET.VAL_NPZ_ROOT = cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}/scene_info_val_1500_no_sfm"

#  line 47 in LoFTR/src/datasets/megadepth.py:

# self.scene_info = dict(np.load(npz_path, allow_pickle=True))