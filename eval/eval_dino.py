import cv2
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image
import os
import random
import torch.nn as nn
import sys
sys.path.append("./dinov2")
import hubconf
from omegaconf import OmegaConf
config_path = '/network_space/server128/shared/zhuoying/AnyDoor-main/configs/anydoor.yaml'
config = OmegaConf.load(config_path)
DINOv2_weight_path = config.model.params.cond_stage_config.weight
class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
class FrozenDinoV2Encoder(AbstractEncoder):
    """
    Uses the DINOv2 encoder for image
    """
    def __init__(self, device="cuda", freeze=True):
        super().__init__()
        dinov2 = hubconf.dinov2_vitg14() 
        state_dict = torch.load(DINOv2_weight_path)
        dinov2.load_state_dict(state_dict, strict=False)
        self.model = dinov2.to(device)
        self.device = device
        if freeze:
            self.freeze()
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.image_std =  torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)        
        self.projector = nn.Linear(1536,1024)

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        if isinstance(image,list):
            image = torch.cat(image,0)

        image = (image.to(self.device)  - self.image_mean.to(self.device)) / self.image_std.to(self.device)
        features = self.model.forward_features(image)
        tokens = features["x_norm_patchtokens"]
        image_features  = features["x_norm_clstoken"]
        image_features = image_features.unsqueeze(1)
        hint = torch.cat([image_features,tokens],1) # 8,257,1024
        hint = self.projector(hint)
        return hint

    def encode(self, image):
        return self(image)


if __name__ == '__main__': 
    dino_model = FrozenDinoV2Encoder().to('cuda')
    gt_root_path = '/home/xuzhu/MoMA/data/process_data_test'
    root_path = '/home/xuzhu/MoMA/data/process_data_test'
    cat_dirs = sorted(os.listdir(root_path))
    all_hoi_sim_list = []
    all_h_sim_list = []
    all_o_sim_list = []
    for cat_name in tqdm(cat_dirs):
        
        cat_dir = os.path.join(root_path,cat_name)
        gt_cat_dir = os.path.join(gt_root_path,cat_name)
        if os.path.exists(os.path.join(cat_dir,'llava_judge_all_cat_mask_gen.png')):
            gt_image_path = os.path.join(gt_cat_dir,'hoi.jpg')
            gen_image_path = os.path.join(cat_dir,'llava_judge_all_cat_mask_gen.png')
            gt_human_mask_path = os.path.join(gt_cat_dir,'human_mask.jpg')
            gt_object_mask_path =  os.path.join(gt_cat_dir,'object_mask.jpg')
            gt_hoi_img = cv2.imread(gt_image_path)
            gt_hoi_img = cv2.cvtColor(gt_hoi_img, cv2.COLOR_BGR2RGB)
            
            gen_hoi_img = cv2.imread(gen_image_path)
            gen_hoi_img = cv2.cvtColor(gen_hoi_img, cv2.COLOR_BGR2RGB)
            sub_mask = (cv2.imread(gt_human_mask_path) > 128).astype(np.uint8)[:,:,0]
            obj_mask = (cv2.imread(gt_object_mask_path) > 128).astype(np.uint8)[:,:,0]
            union_mask = np.maximum(sub_mask,obj_mask)
            union_mask_image = np.stack([union_mask,union_mask,union_mask],-1)
            sub_mask_image = np.stack([sub_mask,sub_mask,sub_mask],-1)
            obj_mask_image = np.stack([obj_mask,obj_mask,obj_mask],-1)
            
            h,w = union_mask_image.shape[:-1]
            gen_hoi_img = cv2.resize(gen_hoi_img,(w,h))
            empty_back_image = gen_hoi_img.copy()
            empty_back_image[:,:,:] = 255
            gen_hoi_fore_image = union_mask_image*gen_hoi_img + (1-union_mask_image)*empty_back_image 
            gt_hoi_fore_image = union_mask_image*gt_hoi_img + (1-union_mask_image)*empty_back_image
            gen_h_fore_image = sub_mask_image*gen_hoi_img + (1-sub_mask_image)*empty_back_image
            gt_h_fore_image = sub_mask_image*gt_hoi_img + (1-sub_mask_image)*empty_back_image
            gen_o_fore_image = obj_mask_image*gen_hoi_img + (1-obj_mask_image)*empty_back_image
            gt_o_fore_image = obj_mask_image*gt_hoi_img + (1-obj_mask_image)*empty_back_image

            
            gt_dino_hoi_feature = dino_model(torch.tensor(cv2.resize(gt_hoi_fore_image,(224,224))/255,dtype=torch.float32).unsqueeze(0).permute(0,3,1,2))
            gen_dino_hoi_feature = dino_model(torch.tensor(cv2.resize(gen_hoi_fore_image,(224,224))/255,dtype=torch.float32).unsqueeze(0).permute(0,3,1,2))
            hoi_sim = torch.cosine_similarity(gt_dino_hoi_feature.reshape(1,-1),gen_dino_hoi_feature.reshape(1,-1),dim=1)
            all_hoi_sim_list.append(hoi_sim.item())
            gt_dino_h_feature = dino_model(torch.tensor(cv2.resize(gt_h_fore_image,(224,224))/255,dtype=torch.float32).unsqueeze(0).permute(0,3,1,2))
            gen_dino_h_feature = dino_model(torch.tensor(cv2.resize(gen_h_fore_image,(224,224))/255,dtype=torch.float32).unsqueeze(0).permute(0,3,1,2))
            h_sim = torch.cosine_similarity(gt_dino_h_feature.reshape(1,-1),gen_dino_h_feature.reshape(1,-1),dim=1)
            all_h_sim_list.append(h_sim.item())
            gt_dino_o_feature = dino_model(torch.tensor(cv2.resize(gt_o_fore_image,(224,224))/255,dtype=torch.float32).unsqueeze(0).permute(0,3,1,2))
            gen_dino_o_feature = dino_model(torch.tensor(cv2.resize(gen_o_fore_image,(224,224))/255,dtype=torch.float32).unsqueeze(0).permute(0,3,1,2))
            o_sim = torch.cosine_similarity(gt_dino_o_feature.reshape(1,-1),gen_dino_o_feature.reshape(1,-1),dim=1)
            all_o_sim_list.append(o_sim.item())
        else:
            continue
        
    print(f'DINO-score for ho pair is {sum(all_hoi_sim_list)/len(all_hoi_sim_list)}')
    print(f'DINO-score for human is {sum(all_h_sim_list)/len(all_h_sim_list)}')
    print(f'DINO-score of object is {sum(all_o_sim_list)/len(all_o_sim_list)}')    
 