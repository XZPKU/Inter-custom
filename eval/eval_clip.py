import cv2
import einops
import numpy as np
import torch
import random
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from omegaconf import OmegaConf
from PIL import Image
import os
import random
from tqdm import tqdm
import torch.nn as nn
from omegaconf import OmegaConf

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
import clip




if __name__ == '__main__': 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("/network_space/server128/shared/xuzhu/.cache/clip/ViT-B-32.pt", device=device)
    root_path = '/home/xuzhu/MoMA/data/process_data_test'
    gt_root_path = '/home/xuzhu/MoMA/data/process_data_test'
    cat_dirs = sorted(os.listdir(root_path))
    all_hoi_sim_list = []
    all_h_sim_list = []
    all_o_sim_list = []
    for cat_name in tqdm(cat_dirs):
        
        cat_dir = os.path.join(root_path,cat_name)
        gt_cat_dir = os.path.join(gt_root_path,cat_name)
        try:
            gt_image_path = os.path.join(gt_cat_dir,'hoi.jpg')
            gen_image_path = os.path.join(cat_dir,'llava_judge_all_cat_mask_gen.png')
            gt_human_mask_path = os.path.join(gt_cat_dir,'human_mask.jpg')
            gt_object_mask_path =  os.path.join(gt_cat_dir,'object_mask.jpg')
            gt_image = preprocess(Image.open(gt_image_path)).unsqueeze(0).to(device)
            gen_image = preprocess(Image.open(gen_image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                gt_image_features = model.encode_image(gt_image)
                gen_image_features = model.encode_image(gen_image)
                sim = gt_image_features@gen_image_features.T
                all_hoi_sim_list.append(sim.item())
        except:
            continue
        
    print(f'CLIP-I is {sum(all_hoi_sim_list)/len(all_hoi_sim_list)}')