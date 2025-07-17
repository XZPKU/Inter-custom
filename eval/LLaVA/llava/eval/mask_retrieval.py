# import cv2
# import einops
# import numpy as np
# import torch
# import random
# from pytorch_lightning import seed_everything
# # from cldm.model import create_model, load_state_dict
# # from cldm.ddim_hacked import DDIMSampler
# # from cldm.hack import disable_verbosity, enable_sliced_attention
# # from datasets.data_utils import * 
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
# import albumentations as A
# from omegaconf import OmegaConf
# from PIL import Image
# import os
# import random
# save_memory = False
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,4,5'
import argparse
import torch
import json
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import cv2
from PIL import Image
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import re


import json


def select_mask(args):
   
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode
    obj_category = args.target_obj_category
    query = f'There is a person and a {obj_category}, please use three sentences to seperately describe (1) the haircut, body shape and appearance of person,(2) the shape and appearance of object. Following is a example: The person is a short hair, tall and thin man, and wearing a hat. The bicycle is a colorful mountain bicycle.'
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # image_files = [image_to_caption]
    # images = load_images(image_files)
    sub_img = Image.open(args.target_sub_img_path)
    obj_img = Image.open(args.target_obj_img_path)
    width = sub_img.width + obj_img.width
    height = max(sub_img.height, obj_img.height)
    result_img = Image.new('RGB', (width, height))
    result_img.paste(sub_img, (0, 0))
    result_img.paste(obj_img, (sub_img.width, 0))
    result_img.save('concat.jpg')
    images = [Image.fromarray(cv2.resize(np.array(result_img),(256,256)))]
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    
    
    with open(args.bank_path,'r') as f:
        mask_bank = json.load(f)
    stored_masks = mask_bank[str(args.target_hoi_category)]
    del mask_bank
    stored_ids = [id for id,_ in stored_masks]
    stored_captions = [cap for _,cap in stored_masks]
   
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/network_space/server128/shared/LLaVA/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default = '/network_space/server128/shared/zhuoying/AnyDoor-main/taobao_more_category/barbell/1/img_12.png')
    parser.add_argument("--query", type=str, default = 'There is a person and a barbell in the front of image, please describe their interaction with a single word verb, and form a complete sentence in the following format:a person is riding the horse.')
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--target_hoi_category",type = int,default=154,required=False)
    parser.add_argument("--target_obj_category",type=str,default='motorcycle',required=False)
    parser.add_argument('--target_sub_img_path',type=str,default = '/network_space/server128/shared/zhuoying/AnyDoor-main/hoi_samples/man.jpg',required=False)
    parser.add_argument('--target_obj_img_path',type=str,default = '/network_space/server128/shared/zhuoying/AnyDoor-main/hoi_samples/motorcycle.jpg',required=False)
    parser.add_argument('--bank_path',type=str,default= '/network_space/server128/shared/zhuoying/AnyDoor-main/datasets/mask_storage.json',required=False)
    args = parser.parse_args()
    selected_mask = select_mask(args)
    
    

        
    