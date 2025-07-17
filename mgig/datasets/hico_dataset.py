import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset
import albumentations as A
import glob
 

class HICODataset_concat_h_o(BaseDataset):
    def __init__(self, image_dir):
        self.image_root = image_dir
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 2
        self.id_path_root = '/home/xuzhu/MoMA/data/process_data_test/'
        
        self.files = glob.glob(os.path.join(self.image_root, 'HICO_train2015_*.pt'))
        with open("/home/xuzhu/MoMA/data/process_data/hico_test_instance_10_10.json", "r") as file:
            info = json.load(file)
        self.raw_files = info
        self.data = self.raw_files
        assert len(self.files) > 0, f'No file found at {self.image_root}!'


    def __len__(self):
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
    def hico_process_pairs(self, human_image, human_mask, ref_image, sub_mask, obj_mask,  max_ratio = 0.8):
        tar_image = ref_image 
        obj_box_yyxx = get_bbox_from_mask(obj_mask)
        human_box_yyxx = get_bbox_from_mask(human_mask)
        crop_obj = ref_image[obj_box_yyxx[0]:obj_box_yyxx[1],obj_box_yyxx[2]:obj_box_yyxx[3],:]
        crop_human = human_image[human_box_yyxx[0]:human_box_yyxx[1],human_box_yyxx[2]:human_box_yyxx[3],:]
        crop_obj_mask =  obj_mask[obj_box_yyxx[0]:obj_box_yyxx[1],obj_box_yyxx[2]:obj_box_yyxx[3]]
        crop_human_mask = human_mask[human_box_yyxx[0]:human_box_yyxx[1],human_box_yyxx[2]:human_box_yyxx[3]]
        crop_obj = cv2.resize(crop_obj.astype(np.uint8),(112,112)).astype(np.uint8)
        crop_human = cv2.resize(crop_human.astype(np.uint8),(112,112)).astype(np.uint8)
        crop_human_mask = cv2.resize(np.stack([crop_human_mask,crop_human_mask,crop_human_mask],-1).astype(np.uint8),(112,112)).astype(np.uint8)[:,:,0]
        crop_obj_mask =  cv2.resize(np.stack([crop_obj_mask,crop_obj_mask,crop_obj_mask],-1).astype(np.uint8),(112,112)).astype(np.uint8)[:,:,0]
        ref_image = np.ones((224,224,3))*255
        ref_mask = np.zeros((224,224))
        ref_image[0:112,0:112,:] = crop_human
        ref_image[112:224,112:224,:] = crop_obj
        ref_mask[0:112,0:112] = crop_human_mask
        ref_mask[112:224,112:224]  = crop_obj_mask
        #####################################3        
        
        tar_mask  = np.maximum(sub_mask,obj_mask)
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
       
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)
        y1,y2,x1,x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
        ref_mask = ref_mask[y1:y2,x1:x2]


        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224,224) ).astype(np.uint8)

        ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
        ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224,224) ).astype(np.uint8)
        ref_mask = ref_mask_3[:,:,0]
        masked_ref_image_compose, ref_mask_compose =  self.aug_data_mask(masked_ref_image, ref_mask) 
        masked_ref_image_aug = masked_ref_image_compose.copy()

        ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
        ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)
        

        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2]) #1.1  1.3
        tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])   
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
        y1,y2,x1,x2 = tar_box_yyxx_crop
        cropped_target_image = tar_image[y1:y2,x1:x2,:]
        cropped_tar_mask = tar_mask[y1:y2,x1:x2]
        tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
        y1,y2,x1,x2 = tar_box_yyxx

        ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

        collage = cropped_target_image.copy() 
        collage[y1:y2,x1:x2,:] = ref_image_collage

        collage_mask = cropped_target_image.copy() * 0.0
        collage_mask[y1:y2,x1:x2,:] = 1.0

        if np.random.uniform(0, 1) < 0.7: 
            cropped_tar_mask = perturb_mask(cropped_tar_mask)
            collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)

        H1, W1 = collage.shape[0], collage.shape[1]

        cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
        collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
        collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)
        H2, W2 = collage.shape[0], collage.shape[1]

        cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512,512)).astype(np.float32)
        collage = cv2.resize(collage.astype(np.uint8), (512,512)).astype(np.float32)
        collage_mask  = cv2.resize(collage_mask.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32)
        collage_mask[collage_mask == 2] = -1
        
        # Prepairing dataloader items
        masked_ref_image_aug = masked_ref_image_aug  / 255 
        cropped_target_image = cropped_target_image / 127.5 - 1.0
        collage = collage / 127.5 - 1.0 
        collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)
        
        item = dict(
                ref=masked_ref_image_aug.copy(), 
                jpg=cropped_target_image.copy(), 
                hint=collage.copy(), 
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array(tar_box_yyxx_crop) 
                ) 
        return item        
    def get_sample(self, idx):
        
        raw_path = os.path.join(self.id_path_root,self.raw_files[idx])
        try:
            hoi_img_path = os.path.join(raw_path,'hoi.jpg')
            sub_mask_path = os.path.join(raw_path,'human_mask.jpg')
            obj_mask_path = os.path.join(raw_path,'object_mask.jpg')
            
            hoi_img = cv2.imread(hoi_img_path)
            hoi_img = cv2.cvtColor(hoi_img, cv2.COLOR_BGR2RGB)
            
            sub_mask = (cv2.imread(sub_mask_path) > 128).astype(np.uint8)[:,:,0]
            obj_mask = (cv2.imread(obj_mask_path) > 128).astype(np.uint8)[:,:,0]
            
            h,w = hoi_img.shape[:-1]
            human_img_path = os.path.join(raw_path,'anydoor_human_0.jpg')
            human_mask_path = os.path.join(raw_path,'anydoor_human_mask_0.jpg')
            human_img = cv2.imread(human_img_path)
            human_img = cv2.cvtColor(human_img, cv2.COLOR_BGR2RGB)
            human_img = cv2.resize(human_img.astype(np.uint8),(w,h)).astype(np.uint8)
            human_mask = cv2.imread(human_mask_path)
            human_mask = cv2.resize(human_mask.astype(np.uint8),(w,h)).astype(np.uint8)
            human_mask = (human_mask>128).astype(np.uint8)[:,:,0]
            
            #### hoi_mask = sub_mask ^ obj_mask
            
            item_with_collage = self.hico_process_pairs(human_img,human_mask,hoi_img,sub_mask,obj_mask,max_ratio = 1.0)
            sampled_time_steps = self.sample_timestep()
            item_with_collage['time_steps'] = sampled_time_steps
            return item_with_collage
        except:
            raw_path = os.path.join(self.id_path_root,self.raw_files[0])
            hoi_img_path = os.path.join(raw_path,'hoi.jpg')
            sub_mask_path = os.path.join(raw_path,'human_mask.jpg')
            obj_mask_path = os.path.join(raw_path,'object_mask.jpg')
            
            hoi_img = cv2.imread(hoi_img_path)
            hoi_img = cv2.cvtColor(hoi_img, cv2.COLOR_BGR2RGB)
            
            sub_mask = (cv2.imread(sub_mask_path) > 128).astype(np.uint8)[:,:,0]
            obj_mask = (cv2.imread(obj_mask_path) > 128).astype(np.uint8)[:,:,0]
            
            h,w = hoi_img.shape[:-1]
            human_img_path = os.path.join(raw_path,'anydoor_human_0.jpg')
            human_mask_path = os.path.join(raw_path,'anydoor_human_mask_0.jpg')
            human_img = cv2.imread(human_img_path)
            human_img = cv2.cvtColor(human_img, cv2.COLOR_BGR2RGB)
            human_img = cv2.resize(human_img.astype(np.uint8),(w,h)).astype(np.uint8)
            human_mask = cv2.imread(human_mask_path)
            human_mask = cv2.resize(human_mask.astype(np.uint8),(w,h)).astype(np.uint8)
            human_mask = (human_mask>128).astype(np.uint8)[:,:,0]
            
            item_with_collage = self.hico_process_pairs(human_img,human_mask,hoi_img,sub_mask,obj_mask,max_ratio = 1.0)
            sampled_time_steps = self.sample_timestep()
            item_with_collage['time_steps'] = sampled_time_steps
            return item_with_collage   
        

        

    