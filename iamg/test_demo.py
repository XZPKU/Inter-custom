import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler_shape as PLMSSampler 
from ldm.util import instantiate_from_config
import numpy as np
import random
import time
from dataset.hoi import HICODataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
import os
import shutil
import torchvision
from convert_ckpt import add_additional_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from distributed import get_rank, synchronize, get_world_size, print_dist
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from copy import deepcopy
import cv2
from PIL import Image
import json
# = = = = = = = = = = = = = = = = = = useful functions = = = = = = = = = = = = = = = = = #

import numpy as np
import torch 
import cv2
import albumentations as A

def mask_score(mask):
    '''Scoring the mask according to connectivity.'''
    mask = mask.astype(np.uint8)
    if mask.sum() < 10:
        return 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    conc_score = np.max(cnt_area) / sum(cnt_area)
    return conc_score


def sobel(img, mask, thresh = 50):
    '''Calculating the high-frequency map.'''
    H,W = img.shape[0], img.shape[1]
    img = cv2.resize(img,(256,256))
    mask = (cv2.resize(mask,(256,256)) > 0.5).astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 2)
    
    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Ksize)
    sobel_X = cv2.convertScaleAbs(sobelx)
    sobel_Y = cv2.convertScaleAbs(sobely)
    scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    scharr = np.max(scharr,-1) * mask    
    
    scharr[scharr < thresh] = 0.0
    scharr = np.stack([scharr,scharr,scharr],-1)
    scharr = (scharr.astype(np.float32)/255 * img.astype(np.float32) ).astype(np.uint8)
    scharr = cv2.resize(scharr,(W,H))
    return scharr


def resize_and_pad(image, box):
    '''Fitting an image to the box region while keeping the aspect ratio.'''
    y1,y2,x1,x2 = box
    H,W = y2-y1, x2-x1
    h,w =  image.shape[0], image.shape[1]
    r_box = W / H 
    r_image = w / h
    if r_box >= r_image:
        h_target = H
        w_target = int(w * H / h) 
        image = cv2.resize(image, (w_target, h_target))

        w1 = (W - w_target) // 2
        w2 = W - w_target - w1
        pad_param = ((0,0),(w1,w2),(0,0))
        image = np.pad(image, pad_param, 'constant', constant_values=255)
    else:
        w_target = W 
        h_target = int(h * W / w)
        image = cv2.resize(image, (w_target, h_target))

        h1 = (H-h_target) // 2 
        h2 = H - h_target - h1
        pad_param =((h1,h2),(0,0),(0,0))
        image = np.pad(image, pad_param, 'constant', constant_values=255)
    return image



def expand_image_mask(image, mask, ratio=1.4):
    h,w = image.shape[0], image.shape[1]
    H,W = int(h * ratio), int(w * ratio) 
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W -w) // 2)
    w2 = W -w - w1

    pad_param_image = ((h1,h2),(w1,w2),(0,0))
    pad_param_mask = ((h1,h2),(w1,w2))
    image = np.pad(image, pad_param_image, 'constant', constant_values=255)
    mask = np.pad(mask, pad_param_mask, 'constant', constant_values=0)
    return image, mask


def resize_box(yyxx, H,W,h,w):
    y1,y2,x1,x2 = yyxx
    y1,y2 = int(y1/H * h), int(y2/H * h)
    x1,x2 = int(x1/W * w), int(x2/W * w)
    y1,y2 = min(y1,h), min(y2,h)
    x1,x2 = min(x1,w), min(x2,w)
    return (y1,y2,x1,x2)


def get_bbox_from_mask(mask):
    h,w = mask.shape[0],mask.shape[1]

    if mask.sum() < 10:
        return 0,h,0,w
    rows = np.any(mask,axis=1)
    cols = np.any(mask,axis=0)
    y1,y2 = np.where(rows)[0][[0,-1]]
    x1,x2 = np.where(cols)[0][[0,-1]]
    return (y1,y2,x1,x2)


def expand_bbox(mask,yyxx,ratio=[1.2,2.0], min_crop=0):
    y1,y2,x1,x2 = yyxx
    ratio = np.random.randint( ratio[0] * 10,  ratio[1] * 10 ) / 10
    H,W = mask.shape[0], mask.shape[1]
    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2-y1+1)
    w = ratio * (x2-x1+1)
    h = max(h,min_crop)
    w = max(w,min_crop)

    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)


def box2squre(image, box):
    H,W = image.shape[0], image.shape[1]
    y1,y2,x1,x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h,w = y2-y1, x2-x1

    if h >= w:
        x1 = cx - h//2
        x2 = cx + h//2
    else:
        y1 = cy - w//2
        y2 = cy + w//2
    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)


def pad_to_square(image, pad_value = 255, random = False):
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



def box_in_box(small_box, big_box):
    y1,y2,x1,x2 = small_box
    y1_b, _, x1_b, _ = big_box
    y1,y2,x1,x2 = y1 - y1_b ,y2 - y1_b, x1 - x1_b ,x2 - x1_b
    return (y1,y2,x1,x2 )



def shuffle_image(image, N):
    height, width = image.shape[:2]
    
    block_height = height // N
    block_width = width // N
    blocks = []
    
    for i in range(N):
        for j in range(N):
            block = image[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width]
            blocks.append(block)
    
    np.random.shuffle(blocks)
    shuffled_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(N):
        for j in range(N):
            shuffled_image[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = blocks[i*N+j]
    return shuffled_image


def get_mosaic_mask(image, fg_mask, N=16, ratio = 0.5):
    ids = [i for i in range(N * N)]
    masked_number = int(N * N * ratio)
    masked_id = np.random.choice(ids, masked_number, replace=False)
    

    
    height, width = image.shape[:2]
    mask = np.ones((height, width))
    
    block_height = height // N
    block_width = width // N
    
    b_id = 0
    for i in range(N):
        for j in range(N):
            if b_id in masked_id:
                mask[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = mask[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] * 0
            b_id += 1
    mask = mask * fg_mask
    mask3 = np.stack([mask,mask,mask],-1).copy().astype(np.uint8)
    noise = q_x(image)
    noise_mask = image * mask3 + noise * (1-mask3)
    return noise_mask

def extract_canney_noise(image, mask, dilate=True):
    h,w = image.shape[0],image.shape[1]
    mask = cv2.resize(mask.astype(np.uint8),(w,h)) > 0.5
    kernel = np.ones((8, 8), dtype=np.uint8)
    mask =  cv2.erode(mask.astype(np.uint8), kernel, 10)

    canny = cv2.Canny(image, 50,100) * mask
    kernel = np.ones((8, 8), dtype=np.uint8)
    mask = (cv2.dilate(canny, kernel, 5) > 128).astype(np.uint8)
    mask = np.stack([mask,mask,mask],-1)

    pure_noise = q_x(image, t=1) * 0 + 255
    canny_noise = mask * image + (1-mask) * pure_noise
    return canny_noise


def get_random_structure(size):
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)


def select_max_region(mask):
    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    background = 0
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            background = row
    stats_no_bg = np.delete(stats, background, axis=0)
    max_idx = stats_no_bg[:, 4].argmax()
    max_region = np.where(labels==max_idx+1, 1, 0)

    return max_region.astype(np.uint8)



def perturb_mask(gt, min_iou = 0.3,  max_iou = 0.99):
    iou_target = np.random.uniform(min_iou, max_iou)
    h, w = gt.shape
    gt = gt.astype(np.uint8)
    seg = gt.copy()
    
    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.1:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            # Dilate/erode
            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])
            
            seg = np.logical_or(seg, gt).astype(np.uint8)
            #seg = select_max_region(seg) 

        if compute_iou(seg, gt) < iou_target:
            break
    seg = select_max_region(seg.astype(np.uint8)) 
    return seg.astype(np.uint8)


def q_x(x_0,t=65):
    '''Adding noise for and given image.'''
    x_0 = torch.from_numpy(x_0).float() / 127.5 - 1
    num_steps = 100
    
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

    alphas = 1-betas
    alphas_prod = torch.cumprod(alphas,0)
    
    alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise).numpy()  * 127.5 + 127.5 


def extract_target_boundary(img, target_mask):
    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Ksize)

    # sobel-x
    sobel_X = cv2.convertScaleAbs(sobelx)
    # sobel-y
    sobel_Y = cv2.convertScaleAbs(sobely)
    # sobel-xy
    scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    scharr = np.max(scharr,-1).astype(np.float32)/255
    scharr = scharr *  target_mask.astype(np.float32)
    return scharr
class ImageCaptionSaver:
    def __init__(self, base_path, nrow=8, normalize=True, scale_each=True, range=(-1, 1)):
        self.base_path = base_path
        self.nrow = nrow
        self.normalize = normalize
        self.scale_each = scale_each
        self.range = range

    def __call__(self,origin_image, target, condition, generation, h,w, seen,ids=None):
        
        padd = abs(h - w)
        origin_bbox_size = max(h,w)
        target = target[0].permute(1,2,0).cpu().numpy()
        condition = condition[0].permute(1,2,0).cpu().numpy()
        generation = generation[0].permute(1,2,0).cpu().numpy()
        target_image = (cv2.resize(target.astype(np.uint8), (origin_bbox_size,origin_bbox_size) ).astype(np.uint8)+1)*127.5
        condition_image = (cv2.resize(condition.astype(np.uint8), (origin_bbox_size,origin_bbox_size) ).astype(np.uint8)+1)*127.5
        generation_image = (cv2.resize(generation.astype(np.uint8), (origin_bbox_size,origin_bbox_size) ).astype(np.uint8)+1)*127.5
        padd_1 = int(padd / 2)
        padd_2 = padd - padd_1
        if h > w:
            target_origin_shape = target_image[padd_1:origin_bbox_size-padd_2,:]
            condition_origin_hsape = condition_image[padd_1:origin_bbox_size-padd_2,:]
            result_orogin_shape = generation_image[padd_1:origin_bbox_size-padd_2,:]
        else:
            target_origin_shape = target_image[:,padd_1:origin_bbox_size-padd_2]
            condition_origin_shape = condition_image[:,padd_1:origin_bbox_size-padd_2]
            result_origin_shape = generation_image[:,padd_1:origin_bbox_size-padd_2]
         
        self.base_path = '/home/xuzhu/interactdiffusion/hoi_mask_eval/'
        save_path = os.path.join(self.base_path, str(seen).zfill(8) + '_target.png')
        Image.fromarray(target_origin_shape.astype(np.uint8)).save(save_path)
        save_path = os.path.join(self.base_path, str(seen).zfill(8) + '_condition.png')
        Image.fromarray(condition_origin_shape.astype(np.uint8)).save(save_path)
        save_path = os.path.join(self.base_path, str(seen).zfill(8) + '_generation.png')
        Image.fromarray(result_origin_shape.astype(np.uint8)).save(save_path)
    def resize_forward(self, target, condition, generation, h,w, seen,ids=None):
        
        padd = abs(h - w)
        origin_bbox_size = max(h,w)
        padd_1 = int(padd / 2)
        padd_2 = padd - padd_1
        ratio = 256/origin_bbox_size
        ratio_padd_1 = int(ratio*padd_1)
        ratio_padd_2 = int(padd_2*ratio)
        if h > w:
            target_origin_shape = target[:,:,ratio_padd_1:256-ratio_padd_2,:]
            condition_origin_shape = condition[:,:,ratio_padd_1:256-ratio_padd_2,:]
            result_origin_shape = generation[:,:,ratio_padd_1:256-ratio_padd_2,:]
        else:
            target_origin_shape = target[:,:,ratio_padd_1:256-ratio_padd_2,:]
            condition_origin_shape = condition[:,:,ratio_padd_1:256-ratio_padd_2,:]
            result_origin_shape = generation[:,:,ratio_padd_1:256-ratio_padd_2,:]
        
        self.base_path = './demo_result'
        os.makedirs(self.base_path,exist_ok=True)
        save_path = os.path.join(self.base_path, str(seen).zfill(8) + '_condition.png')
        torchvision.utils.save_image(condition_origin_shape, save_path, nrow=self.nrow)
        save_path = os.path.join(self.base_path, str(seen).zfill(8) + '_generation.png')
        torchvision.utils.save_image(result_origin_shape, save_path, nrow=self.nrow)

        


def read_official_ckpt(ckpt_path):
    "Read offical pretrained SD ckpt and convert into my style"
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    out = {}
    out["model"] = {}
    out["text_encoder"] = {}
    out["autoencoder"] = {}
    out["unexpected"] = {}
    out["diffusion"] = {}

    for k, v in state_dict.items():
        if k.startswith('model.diffusion_model'):
            out["model"][k.replace("model.diffusion_model.", "")] = v
        elif k.startswith('cond_stage_model'):
            out["text_encoder"][k.replace("cond_stage_model.", "")] = v
        elif k.startswith('first_stage_model'):
            out["autoencoder"][k.replace("first_stage_model.", "")] = v
        elif k in ["model_ema.decay", "model_ema.num_updates"]:
            out["unexpected"][k] = v
        else:
            out["diffusion"][k] = v
    return out


def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def sub_batch(batch, num=1):
    # choose first num in given batch 
    num = num if num > 1 else 1
    for k in batch:
        batch[k] = batch[k][0:num]
    return batch


def wrap_loader(loader):
    while True:
        for batch in loader:  # TODO: it seems each time you have the same order for all epoch?? 
            yield batch


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


def count_params(params):
    total_trainable_params_count = 0
    for p in params:
        total_trainable_params_count += p.numel()
    print_dist("total_trainable_params_count is: ", total_trainable_params_count)


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def create_expt_folder_with_auto_resuming(OUTPUT_ROOT, name):
    name = os.path.join(OUTPUT_ROOT, name)
    writer = None
    checkpoint = None

    if os.path.exists(name):
        all_tags = os.listdir(name)
        all_existing_tags = [tag for tag in all_tags if tag.startswith('tag')]
        all_existing_tags.sort()
        all_existing_tags = all_existing_tags[::-1]
        for previous_tag in all_existing_tags:
            potential_ckpt = os.path.join(name, previous_tag, 'checkpoint_latest.pth')
            if os.path.exists(potential_ckpt):
                checkpoint = potential_ckpt
                if get_rank() == 0:
                    print('auto-resuming ckpt found ' + potential_ckpt)
                break
        curr_tag = 'tag' + str(len(all_existing_tags)).zfill(2)
        name = os.path.join(name, curr_tag)  # output/name/tagxx
    else:
        name = os.path.join(name, 'tag00')  # output/name/tag00

    if get_rank() == 0:
        os.makedirs(name,exist_ok=True)
        os.makedirs(os.path.join(name, 'Log'),exist_ok=True)
        writer = SummaryWriter(os.path.join(name, 'Log'))
    checkpoint = '/home/xuzhu/interactdiffusion/OUTPUT/all_cat_tst_llava_judge_liantong_10_resume_from_few_cat_45000/test/tag00/checkpoint_00790001.pth'
    # print('ckpt is {checkpoint}')
    return name, writer, checkpoint


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 


class Trainer:
    def __init__(self, config, device="cuda"):

        self.config = config
        self.device = torch.device(device)

        self.l_simple_weight = 1
        self.name, self.writer, checkpoint = create_expt_folder_with_auto_resuming(config.OUTPUT_ROOT, config.name)
        if get_rank() == 0:
            shutil.copyfile(config.yaml_file, os.path.join(self.name, "train_config_file.yaml"))
            self.config_dict = vars(config)
            torch.save(self.config_dict, os.path.join(self.name, "config_dict.pth"))

        # = = = = = = = = = = = = = = = = = create model and diffusion = = = = = = = = = = = = = = = = = #
        self.model = instantiate_from_config(config.model).to(self.device)
        self.autoencoder = instantiate_from_config(config.autoencoder).to(self.device)
        self.text_encoder = instantiate_from_config(config.text_encoder).to(self.device)
        self.diffusion = instantiate_from_config(config.diffusion).to(self.device)
        state_dict = read_official_ckpt('/network_space/server128/shared/zhuoying/models/stable-diffusion-v1-4/sd-v1-4.ckpt')
        proj_out_list = [
            ("output_blocks.3.1.proj_out.weight"),
            ("output_blocks.4.1.proj_out.weight"),
            ("output_blocks.5.1.proj_out.weight"),
            ("output_blocks.6.1.proj_out.weight"),
            ("output_blocks.7.1.proj_out.weight"),
            ("output_blocks.8.1.proj_out.weight"),
            ("output_blocks.9.1.proj_out.weight"),
            ("output_blocks.10.1.proj_out.weight"),
            ("output_blocks.11.1.proj_out.weight"),
            ("input_blocks.1.1.proj_out.weight"),
            ("input_blocks.2.1.proj_out.weight"),
            ("input_blocks.4.1.proj_out.weight"),
            ("input_blocks.5.1.proj_out.weight"),
            ("input_blocks.7.1.proj_out.weight"),
            ("input_blocks.8.1.proj_out.weight"),
            ("middle_block.1.proj_out.weight")
        ]
        for weight_name in proj_out_list:
            if state_dict["model"][weight_name].ndim==4:
                temp = state_dict["model"][weight_name].squeeze(dim=3)
                state_dict["model"][weight_name] = temp.to(self.device)
            else:
                continue
        temp = state_dict["model"]["input_blocks.0.0.weight"].repeat(1,2,1,1)
        state_dict["model"]["input_blocks.0.0.weight"] = temp.to(self.device)
        
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict["model"], strict=False)
        original_params_names = list(state_dict["model"].keys())  # used for sanity check later
        self.autoencoder.load_state_dict(state_dict["autoencoder"])
        self.text_encoder.load_state_dict(state_dict["text_encoder"],strict=False)
        self.diffusion.load_state_dict(state_dict["diffusion"],strict=False)
        
        self.autoencoder.eval()
        self.text_encoder.eval()
        disable_grads(self.autoencoder)
        disable_grads(self.text_encoder)
        
        # = = = = = = = = = = = = = = = = = create opt = = = = = = = = = = = = = = = = = #
        params = []
        trainable_names = []
        all_params_name = []

        for name, p in self.model.named_parameters():
            params.append(p)
            trainable_names.append(name)
            all_params_name.append(name)

                
        self.opt = torch.optim.AdamW(params, lr=config.base_learning_rate, weight_decay=config.weight_decay)
        self.scaler = GradScaler(enabled=config.amp)
        print_dist(f"Using AMP: {self.config.amp}")

        count_params(params)

        #  = = = = EMA... It is worse than normal model in early experiments, thus never enabled later = = = = = = = #
        if config.enable_ema:
            self.master_params = list(self.model.parameters())
            self.ema = deepcopy(self.model)
            self.ema_params = list(self.ema.parameters())
            self.ema.eval()

        # = = = = = = = = = = = = = = = = = = = = create scheduler = = = = = = = = = = = = = = = = = = = = #
        if config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps,
                                                             num_training_steps=config.total_iters)
        elif config.scheduler_type == "constant":
            self.scheduler = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps)
        else:
            assert False

            # = = = = = = = = = = = = = = = = = = = = create data = = = = = = = = = = = = = = = = = = = = #
        train_dataset_repeats = config.train_dataset_repeats if 'train_dataset_repeats' in config else None
        dataset_train = HICODataset( os.path.join(config.DATA_ROOT,'hico_det_clip_instance'))
        sampler = DistributedSampler(dataset_train, seed=config.seed) if config.distributed else None
        loader_train = DataLoader(dataset_train, batch_size=config.batch_size,
                                  shuffle=(sampler is None),
                                  num_workers=config.workers,
                                  pin_memory=True,
                                  sampler=sampler)
        self.dataset_train = dataset_train
        self.loader_train = wrap_loader(loader_train)
        #self.loader_train = loader_train

        if get_rank() == 0:
            total_image = dataset_train.total_images()
            print("Total training images: ", total_image)

            # = = = = = = = = = = = = = = = = = load from autoresuming ckpt = = = = = = = = = = = = = = = = = = #
        self.starting_iter = 0
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            if config.enable_ema:
                self.ema.load_state_dict(checkpoint["ema"])
            self.opt.load_state_dict(checkpoint["opt"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            if config.amp:
                self.scaler.load_state_dict(checkpoint['scaler'])
            self.starting_iter = checkpoint["iters"]
        if get_rank() == 0:
            self.image_caption_saver = ImageCaptionSaver(self.name)

        if config.distributed:
            self.model = DDP(self.model, device_ids=[config.local_rank], output_device=config.local_rank,
                             broadcast_buffers=False)

    @torch.no_grad()
    def get_input(self, batch):

        target = self.autoencoder.encode(batch["target"].float())

        context = self.text_encoder.encode(batch["caption"])

        condition = self.autoencoder.encode(batch["condition"].float())
        _t = torch.rand(target.shape[0]).to(target.device)
        t = (torch.pow(_t, 1) * 1000).long()
        t = torch.where(t != 1000, t, 999)  # if 1000, then replace it with 999

        return target, t, context, condition

    def run_one_step(self, batch):
        x_start, t, context, condition = self.get_input(batch)
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)

        model_output = self.model(x_noisy,t,context,condition)

        loss = torch.nn.functional.mse_loss(model_output, noise) * self.l_simple_weight

        self.loss_dict = {"loss": loss.item()}

        return loss

    def start_training(self):

        iterator = tqdm(range(self.starting_iter, self.config.total_iters), desc='Training progress',
                        disable=get_rank() != 0, bar_format='{l_bar}{bar:25}{r_bar}')
        self.model.train()
        for iter_idx in iterator:  
            self.iter_idx = iter_idx

            batch = next(self.loader_train)
            batch_to_device(batch, self.device)

            if (iter_idx+1) % self.config.gradient_accumulation_step == 0 or \
                    (iter_idx == self.config.total_iters - 1):

                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=self.config.amp):
                    loss = self.run_one_step(batch)
                    loss = loss / self.config.gradient_accumulation_step

                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                self.scheduler.step()
                self.opt.zero_grad()
            else:
                with self.model.no_sync():
                    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=self.config.amp):
                        loss = self.run_one_step(batch)
                        loss = loss / self.config.gradient_accumulation_step
                    self.scaler.scale(loss).backward()
                    
            if self.config.enable_ema:
                update_ema(self.ema_params, self.master_params, self.config.ema_rate)

            if (get_rank() == 0):
                if (iter_idx % 10 == 0):
                    self.log_loss()
                if (iter_idx == 0) or (iter_idx % self.config.save_every_iters == 0) or (
                        iter_idx == self.config.total_iters - 1):
                    self.save_ckpt_and_result()
            synchronize()

        synchronize()
        print("Training finished. Start exiting")
        exit()

    def log_loss(self):
        for k, v in self.loss_dict.items():
            self.writer.add_scalar(k, v, self.iter_idx + 1)  # we add 1 as the actual name
    def aug_data_mask(self, image, mask):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            ])

        transformed = transform(image=image.astype(np.uint8), mask = mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return transformed_image, transformed_mask
    def demo_process(self, bg_image, h_s,h_e,w_s,w_e,max_ratio = 1.0):
        tar_image = ref_image = bg_image  
        h,w = bg_image.shape[:2]
        y1 = int(h*h_s)
        y2 = int(h*h_e)
        x1 = int(w*w_s)
        x2 = int(w*w_e)
        back_image = ref_image
        back_image[y1:y2,x1:x2] = 0
        back_image = pad_to_square(back_image, pad_value = 255, random = False)
        back_image = cv2.resize(back_image.astype(np.uint8), (256,256) ).astype(np.uint8)
        
        condition = back_image /127.5 -1.0
        condition = torch.tensor(condition).permute(2,0,1)
        target = back_image /127.5 -1.0
        target = torch.tensor(target).permute(2,0,1)
        item = dict(
                target=target, 
                condition = condition
                ) 
        return item   
    def hico_process_pairs(self, ref_image, sub_mask, obj_mask,  max_ratio = 0.8):
        
        tar_image = ref_image  
        tar_mask = ref_mask = np.maximum(sub_mask,obj_mask)
        
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)
        
        y1,y2,x1,x2 = ref_box_yyxx
        back_image = ref_image
        back_image[y1:y2,x1:x2] = 0
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (256,256) ).astype(np.uint8)

        ref_mask_3_pad = pad_to_square(ref_mask_3*255, pad_value = 0, random = False)
        ref_mask_3_pad_image = cv2.resize(ref_mask_3_pad.astype(np.uint8), (256,256) ).astype(np.uint8)
        
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
    def generate_batch_demo(self):
        bg_img = cv2.imread(self.config.demo_sample)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        h_s,h_e,w_s,w_e = json.loads(self.config.position)
        origin_h,origin_w = bg_img.shape[:2]
        item = self.demo_process(bg_img,h_s,h_e,w_s,w_e,max_ratio=1.0)
        item['caption'] = self.config.hoi_category
        item['origin_h'] = origin_h
        item['origin_w'] = origin_w
        return item

        
    @torch.no_grad()
    def save_ckpt_and_result(self):
        model_wo_wrapper = self.model.module if self.config.distributed else self.model

        iter_name = self.iter_idx + 1  # we add 1 as the actual name
        self.config.disable_inference_in_training = False
        if not self.config.disable_inference_in_training:
            batch_here = 1
            batch = generate_taobao_batch()
            real_target = []
            real_condition = []
            for i in range(batch_here):
                temp_data = {"target": batch["target"][i],"condition": batch["condition"][i]}
                real_target.append(batch["target"][i])
                real_condition.append(batch["condition"][i])
            real_target = torch.stack(real_target)
            real_condition = torch.stack(real_condition)


            uc = self.text_encoder.encode(batch_here * [""])
            context = self.text_encoder.encode(batch["caption"])
            condition = self.autoencoder.encode(batch["condition"].float())
            plms_sampler = PLMSSampler(self.diffusion, model_wo_wrapper)
            shape = (batch_here, int(model_wo_wrapper.in_channels/2), int(model_wo_wrapper.image_size/2), int(model_wo_wrapper.image_size/2))

            input = dict(x = None,
                timesteps = None,
                context = context,
                condition = condition.float()
            )
            samples = plms_sampler.sample(S=50, shape=shape, input=input, uc=uc, guidance_scale=5)

            autoencoder_wo_wrapper = self.autoencoder  # Note itself is without wrapper since we do not train that.

            samples = autoencoder_wo_wrapper.decode(samples).cpu()
            samples = torch.clamp(samples, min=-1, max=1)
            self.image_caption_saver(real_target,real_condition,generation=samples, seen = iter_name,ids=None)

        ckpt = dict(model=model_wo_wrapper.state_dict(),
                    text_encoder=self.text_encoder.state_dict(),
                    autoencoder=self.autoencoder.state_dict(),
                    diffusion=self.diffusion.state_dict(),
                    opt=self.opt.state_dict(),
                    scheduler=self.scheduler.state_dict(),
                    iters=self.iter_idx + 1,
                    config_dict=self.config_dict,
                    )
        if self.config.enable_ema:
            ckpt["ema"] = self.ema.state_dict()
        if self.config.amp:
            ckpt['scaler'] = self.scaler.state_dict()
        torch.save(ckpt, os.path.join(self.name, "checkpoint_" + str(iter_name).zfill(8) + ".pth"))
        torch.save(ckpt, os.path.join(self.name, "checkpoint_latest.pth"))
    @torch.no_grad()
    def evaluation(self):
        model_wo_wrapper = self.model.module if self.config.distributed else self.model

        self.config.disable_inference_in_training = False
        if not self.config.disable_inference_in_training:
            batch = self.generate_batch_demo()
            batch_to_device(batch, self.device)
            
            real_target = []
            real_condition = []
            batch_here = 1
            for i in range(batch_here):
                temp_data = {"target": batch["target"][i],"condition": batch["condition"][i]}
                real_target.append(batch["target"])
                real_condition.append(batch["condition"])
            real_target = torch.stack(real_target)
            real_condition = torch.stack(real_condition)


            uc = self.text_encoder.encode(batch_here * [""])
            context = self.text_encoder.encode(batch["caption"])
            # condition = self.autoencoder.encode(batch["condition"].float())
            condition = self.autoencoder.encode(real_condition.float())
            plms_sampler = PLMSSampler(self.diffusion, model_wo_wrapper)
            shape = (batch_here, int(model_wo_wrapper.in_channels/2), int(model_wo_wrapper.image_size/2), int(model_wo_wrapper.image_size/2))
           
            input = dict(x = None,
                timesteps = None,
                context = context,
                condition = condition.float()
            )
            samples = plms_sampler.sample(S=50, shape=shape, input=input, uc=uc, guidance_scale=5)

            autoencoder_wo_wrapper = self.autoencoder  # Note itself is without wrapper since we do not train that.

            samples = autoencoder_wo_wrapper.decode(samples).cpu()
            samples = torch.clamp(samples, min=-1, max=1)

            iter_name=0
            #origin_image  = Image.open(os.path.join(self.raw_path,'hoi.jpg'))
            self.image_caption_saver.resize_forward(real_target,real_condition,generation=samples, h= batch['origin_h'],w = batch['origin_w'],seen = iter_name,ids=None)
    