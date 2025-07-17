import cv2
import einops
import numpy as np
import torch
import json
from tqdm import tqdm
import random
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image
import os

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


config = OmegaConf.load('./configs/inference_hoi.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file

model = create_model(model_config ).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)



def aug_data_mask(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ])
    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask


def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    # ========= Reference ===========
    # ref expand 
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask 
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]


    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image, (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    # ref aug 
    masked_ref_image_aug = masked_ref_image #aug_data(masked_ref_image) 

    # collage aug 
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask #aug_data_mask(masked_ref_image, ref_mask) 
    masked_ref_image_aug = masked_ref_image_compose.copy()
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)
    ref_image_collage_ = Image.fromarray(np.uint8(ref_image_collage))
    ref_image_collage_.save('/home/xuzhu/MoMA/hf_map.jpg')
    ref_mask_compose_ = Image.fromarray(np.uint8(ref_mask_compose))
    ref_mask_compose_.save('/home/xuzhu/MoMA/ref_mask.jpg')
    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2])

    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])    #1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage, (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = -1, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512,512)).astype(np.float32)
    collage = cv2.resize(collage, (512,512)).astype(np.float32)
    collage_mask  = (cv2.resize(collage_mask, (512,512)).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image_aug = masked_ref_image_aug  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

    item = dict(ref=masked_ref_image_aug.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), extra_sizes=np.array([H1, W1, H2, W2]), tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ) ) 
    return item


def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 5 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image
def aug_data_mask(image, mask):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            #A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,  value=(0,0,0)),
            ])

        transformed = transform(image=image.astype(np.uint8), mask = mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return transformed_image, transformed_mask
   

def hico_process_pairs( human_image, human_mask, ref_image, sub_mask, obj_mask,  max_ratio = 0.8):
    tar_image = ref_image  ##at this line, ref_image is origin hoi image, copy to tar, then we construct new ref_image by concat human and obj
    ##################### concat obj nad human condition into one image ################
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

    # Padding reference image to square and resize to 224
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]
    masked_ref_image_compose, ref_mask_compose =  aug_data_mask(masked_ref_image, ref_mask) 
    masked_ref_image_aug = masked_ref_image_compose.copy()

    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)
    

    # ========= Training Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2]) #1.1  1.3
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])   
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop
    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    cropped_tar_mask = tar_mask[y1:y2,x1:x2]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # Prepairing collage image
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
def inference_single_image(human_img,human_mask,hoi_img,sub_mask,obj_mask, guidance_scale = 5.0):
    item = hico_process_pairs(human_img,human_mask,hoi_img,sub_mask,obj_mask, max_ratio=1.0)
    ref = item['ref'] * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5

    hint_image = hint[:,:,:-1]
    hint_mask = item['hint'][:,:,-1] * 255
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512,512))
    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda() 
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()


    clip_input = torch.from_numpy(ref.copy()).float().cuda() 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    H,W = 512,512

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1 
    image_resolution = 512  
    strength = 1 
    guess_mode = False 
    ddim_steps = 50 
    scale = guidance_scale  
    seed = -1 
    eta = 0.0 

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    gen_image = crop_back(pred, hoi_img, sizes, tar_box_yyxx_crop) 
    return gen_image


if __name__ == '__main__': 
    with open('../data/dataset/annos/test_info.json','r') as f:
        all_sample = json.load(f)
    root_path = '../iamg/OUTPUT/eval'
    gt_root_path = '/home/xuzhu/MoMA/data/process_data_test/'
    for sample in tqdm(all_sample[::-1]):
        sample_root = os.path.join(root_path,sample)
        gt_sample_root = os.path.join(gt_root_path,sample)
        raw_path = sample_root
        hoi_img_path = os.path.join(gt_sample_root,'hoi.jpg')
        
        sub_mask_path = obj_mask_path = os.path.join(raw_path,'mask.png')
        hoi_img = cv2.imread(hoi_img_path)
        hoi_img = cv2.cvtColor(hoi_img, cv2.COLOR_BGR2RGB)
        hoi_img_copy = hoi_img.copy()

        one_stage_hoi_mask = cv2.imread(sub_mask_path)
        h,w = hoi_img.shape[:-1]
        sub_mask = (cv2.resize(cv2.imread(sub_mask_path),(w,h)) >128).astype(np.uint8)[:,:,0]
        obj_mask = (cv2.resize(cv2.imread(obj_mask_path),(w,h)) >128).astype(np.uint8)[:,:,0]
        human_img_path = os.path.join(gt_sample_root,'anydoor_human_0.jpg')
        human_mask_path = os.path.join(gt_sample_root,'anydoor_human_mask_0.jpg')
        human_img = cv2.imread(human_img_path)
        human_img = cv2.cvtColor(human_img, cv2.COLOR_BGR2RGB)
        human_img = cv2.resize(human_img.astype(np.uint8),(w,h)).astype(np.uint8)
        human_mask = cv2.imread(human_mask_path)
        human_mask = cv2.resize(human_mask.astype(np.uint8),(w,h)).astype(np.uint8)
        human_mask = (human_mask>128).astype(np.uint8)[:,:,0]
        
        gen_image = inference_single_image(human_img,human_mask,hoi_img,sub_mask,obj_mask)
        h,w = hoi_img.shape[0], hoi_img.shape[0]
        human_img = cv2.resize(human_img, (w,h))
        ##############
        one_stage_hoi_mask = cv2.resize(one_stage_hoi_mask,(w,h))
        #############
        vis_image = cv2.hconcat([human_img, hoi_img_copy,one_stage_hoi_mask,gen_image])

        save_path = os.path.join(raw_path,'hoi.png')
        cv2.imwrite(save_path, gen_image[:,:,::-1])

