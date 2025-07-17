import importlib
# import wandb
import logging
import os
import os.path as osp
import shutil
from omegaconf import OmegaConf
# from hydra import main
import torch
import torchvision.utils as vutils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import numpy as np
import os.path as osp
from random import randint, uniform, random
import cv2
import PIL
import glob
import albumentations as A
from .data_utils import *
from PIL import Image, ImageFilter
import pandas as pd
import json
import torch as th
import torchvision.transforms.functional as F
from torchvision import transforms as T
from torchvision.transforms import Compose, RandomApply, ToPILImage, ToTensor
from torch.utils.data import Dataset


def pad_to_square_postprocess(image, pad_value = 255, random = False):
    H,W = image.shape[0], image.shape[1]
    if H == W:
        return image

    padd = abs(H - W)
    if random:
        padd_1 = int(np.random.randint(0,padd))
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1

    if H > W:
        pad_param = ((0,0),(padd_1,padd_2),(0,0))
    else:
        pad_param = ((padd_1,padd_2),(0,0),(0,0))

    image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
    return image

class HICODataset():
    def __init__(self, image_dir):
        self.image_root = image_dir
        #self.data = os.listdir(self.image_root)
        self.size = (256,256)
        self.clip_size = (224,224)
        self.dynamic = 2
        self.id_path_root = '/home/xuzhu/MoMA/data/process_data/'
        self.test_id_path_root = '/home/xuzhu/MoMA/data/process_data_test/'

        self.files = glob.glob(os.path.join(self.image_root, 'HICO_train2015_*.pt'))

        # with open("/home/xuzhu/interactdiffusion/dataset/hoi_few_category_high_quality_ratio_0_2.json", "r") as file:
        #     info = json.load(file)
        # with open("/home/xuzhu/interactdiffusion/dataset/hoi_all_cat_test_llaval_judge.json", "r") as file:
        #     info = json.load(file)
        with open('/home/xuzhu/interactdiffusion/dataset/test_all_cat_liantong_10.json','r') as f:
            info = json.load(f)
        self.raw_files = info
        # with open('/home/xuzhu/interactdiffusion/dataset/train_test_caption.json','r') as f:
        #     self.captions = json.load(f)
        with open('/home/xuzhu/interactdiffusion/dataset/test_llava_judge_caption.json','r') as f:
            self.captions = json.load(f)
        self.data = self.raw_files
        assert len(self.files) > 0, f'No file found at {self.image_root}!'


    def __len__(self):
        return len(self.raw_files)
    def total_images(self):
        return len(self.raw_files)
    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H and w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H and w < W:
                pass_flag = False
        return pass_flag
    def get_item(self, index):
        item = torch.load(self.files[index], map_location="cpu")
        return item
    def aug_data_back(self, image):
        transform = A.Compose([
            A.ColorJitter(p=0.5, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            A.ChannelShuffle()
            ])
        transformed = transform(image=image.astype(np.uint8))
        transformed_image = transformed["image"]
        return transformed_image
    
    def aug_data_mask(self, image, mask):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            #A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,  value=(0,0,0)),
            ])

        transformed = transform(image=image.astype(np.uint8), mask = mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return transformed_image, transformed_mask
    def hico_process_pairs(self, ref_image, sub_mask, obj_mask,  max_ratio = 0.8):
        ###### only keep bbox context#####
        #ref_image = np.ones_like(ref_image)
        ###################################
        tar_image = ref_image  
        tar_mask = ref_mask = np.maximum(sub_mask,obj_mask)
        
        # Get the outline Box of the reference image
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        #assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.10, mode = 'min') == True
        
        # Filtering background for the reference image
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        #Image.fromarray((ref_mask_3*225).astype(np.uint8)).save('./ref_mask_3.png')
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)
        
        y1,y2,x1,x2 = ref_box_yyxx
        back_image = ref_image
        back_image[y1:y2,x1:x2] = 0
        # masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
        #ref_mask = ref_mask[y1:y2,x1:x2]
        
        
        
        #ratio = np.random.randint(11, 15) / 10 
        #masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

        # Padding reference image to square and resize to 224
        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (256,256) ).astype(np.uint8)
        #Image.fromarray(masked_ref_image).save('./ref_mask_img.png')
        
        
        
        ref_mask_3_pad = pad_to_square(ref_mask_3*255, pad_value = 0, random = False)
        ref_mask_3_pad_image = cv2.resize(ref_mask_3_pad.astype(np.uint8), (256,256) ).astype(np.uint8)
        #Image.fromarray(ref_mask_3_pad_image).save('./ref_mask_3_pad.png')
        
        back_image = pad_to_square(back_image, pad_value = 255, random = False)
        back_image = cv2.resize(back_image.astype(np.uint8), (256,256) ).astype(np.uint8)
        
        
        
        condition = back_image /127.5 -1.0
        target = ref_mask_3_pad_image /127.5 -1.0
        condition = torch.tensor(condition).permute(2,0,1)
        target = torch.tensor(target).permute(2,0,1)
        item = dict(
                target=target, 
                condition = condition
                ) 
        return item       
        
     
    def __getitem__(self, idx):
        # if os.path.exists(os.path.join(self.id_path_root,self.raw_files[idx])) and os.path.exists(os.path.join(self.image_root,f'{self.raw_files[idx]}.pt')):
        #     raw_path = os.path.join(self.id_path_root,self.raw_files[idx])
        #     pt_file = torch.load(os.path.join(self.image_root,f'{self.raw_files[idx]}.pt'),map_location="cpu") 
        # else:
        #     raw_path = os.path.join(self.id_path_root,self.raw_files[0])
        #     pt_file = torch.load(os.path.join(self.image_root,f'{self.raw_files[0]}.pt'),map_location="cpu") 
        try:
            hoi_sample_name = self.raw_files[idx]
            if hoi_sample_name.startswith('HICO_test'):
                raw_path = os.path.join(self.test_id_path_root,hoi_sample_name)
            elif hoi_sample_name.startswith('HICO_train'):
                raw_path = os.path.join(self.id_path_root,hoi_sample_name)        
            hoi_img_path = os.path.join(raw_path,'hoi.jpg')
            sub_mask_path = os.path.join(raw_path,'human_mask.jpg')
            obj_mask_path = os.path.join(raw_path,'object_mask.jpg')
            
            hoi_img = cv2.imread(hoi_img_path)
            hoi_img = cv2.cvtColor(hoi_img, cv2.COLOR_BGR2RGB)
            
            sub_mask = (cv2.imread(sub_mask_path) > 128).astype(np.uint8)[:,:,0]
            obj_mask = (cv2.imread(obj_mask_path) > 128).astype(np.uint8)[:,:,0]
            origin_h,origin_w = sub_mask.shape
            #### hoi_mask = sub_mask ^ obj_mask
            
            item = self.hico_process_pairs(hoi_img,sub_mask,obj_mask,max_ratio = 1.0)
            item['caption']  = self.captions[hoi_sample_name]
            item['name'] = hoi_sample_name
            item['origin_h'] = origin_h
            item['origin_w'] = origin_w
        except:
            hoi_sample_name = self.raw_files[0]
            if hoi_sample_name.startswith('HICO_test'):
                raw_path = os.path.join(self.test_id_path_root,hoi_sample_name)
            elif hoi_sample_name.startswith('HICO_train'):
                raw_path = os.path.join(self.id_path_root,hoi_sample_name)        
            hoi_img_path = os.path.join(raw_path,'hoi.jpg')
            sub_mask_path = os.path.join(raw_path,'human_mask.jpg')
            obj_mask_path = os.path.join(raw_path,'object_mask.jpg')
            
            hoi_img = cv2.imread(hoi_img_path)
            hoi_img = cv2.cvtColor(hoi_img, cv2.COLOR_BGR2RGB)
            
            sub_mask = (cv2.imread(sub_mask_path) > 128).astype(np.uint8)[:,:,0]
            obj_mask = (cv2.imread(obj_mask_path) > 128).astype(np.uint8)[:,:,0]
            origin_h,origin_w = sub_mask.shape
            #### hoi_mask = sub_mask ^ obj_mask
            
            item = self.hico_process_pairs(hoi_img,sub_mask,obj_mask,max_ratio = 1.0)
            item['caption']  = self.captions[hoi_sample_name]
            item['name'] = hoi_sample_name
            item['origin_h'] = origin_h
            item['origin_w'] = origin_w
        # target, condition =self.vis_get_item(item['target'],item['condition'])
        # target.save('./target.jpg')
        # condition.save('./condition.jpg')
        # print(f'caption is {self.captions[hoi_sample_name]}')
        return item
        # except:
        #     raw_path = os.path.join(self.id_path_root,self.raw_files[0])
        #     hoi_img_path = os.path.join(raw_path,'hoi.jpg')
        #     sub_mask_path = os.path.join(raw_path,'human_mask.jpg')
        #     obj_mask_path = os.path.join(raw_path,'object_mask.jpg')
            
        #     hoi_img = cv2.imread(hoi_img_path)
        #     hoi_img = cv2.cvtColor(hoi_img, cv2.COLOR_BGR2RGB)
            
        #     sub_mask = (cv2.imread(sub_mask_path) > 128).astype(np.uint8)[:,:,0]
        #     obj_mask = (cv2.imread(obj_mask_path) > 128).astype(np.uint8)[:,:,0]
            
            
        #     item = self.hico_process_pairs(hoi_img,sub_mask,obj_mask,max_ratio = 1.0)
        #     item['caption']  = pt_file['caption']

        #     return item
    def vis_get_item(self,target,condition):
        target = (target.permute(1,2,0).cpu().numpy()+1.0)*127.5
        condition =  (condition.permute(1,2,0).cpu().numpy()+1.0)*127.5
        return Image.fromarray(target.astype(np.uint8)), Image.fromarray(condition.astype(np.uint8))
    def vis_get_item_origin_shape(self,target,condition,result,h,w):
        target = (target.permute(1,2,0).cpu().numpy()+1.0)*127.5
        condition =  (condition.permute(1,2,0).cpu().numpy()+1.0)*127.5
        padd = abs(H - W)
        padd_1 = int(padd / 2)
        padd_2 = padd - padd_1
        if H > W:
            target_origin_shape = target[padd_1:256-padd_2,:]
            condition_origin_hsape = target[padd_1:256-padd_2,:]
            result_orogin_shape = result[padd_1:256-padd_2,:]
        else:
            target_origin_shape = target[:,padd_1:256-padd_2]
            condition_origin_hsape = target[:,padd_1:256-padd_2]
            result_orogin_shape = result[:,padd_1:256-padd_2]
        return Image.fromarray(target_origin_shape.astype(np.uint8)), Image.fromarray(condition_origin_hsape.astype(np.uint8)),Image.fromarray(result_orogin_shape.astype(np.uint8))
        
# if __name__ == '__main__':
#     dataset = HICODataset(image_dir = '/home/xuzhu/interactdiffusion/DATA/hico_det_clip_instance')
#     dataset.__getitem__(0)