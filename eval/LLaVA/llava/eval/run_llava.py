import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
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

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
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

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
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
    print(outputs)
    #return outputs



# def my_eval_model(model_path,model_base,model_name,query,conv_mode_in_arg,image_file,sep,temperature,top_p,num_beams,max_new_tokens):
#     # Model
#     disable_torch_init()

#     model_name = get_model_name_from_path(model_path)
#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path, model_base, model_name
#     )

#     qs = query
#     image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
#     if IMAGE_PLACEHOLDER in qs:
#         if model.config.mm_use_im_start_end:
#             qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
#         else:
#             qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
#     else:
#         if model.config.mm_use_im_start_end:
#             qs = image_token_se + "\n" + qs
#         else:
#             qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

#     if "llama-2" in model_name.lower():
#         conv_mode = "llava_llama_2"
#     elif "mistral" in model_name.lower():
#         conv_mode = "mistral_instruct"
#     elif "v1.6-34b" in model_name.lower():
#         conv_mode = "chatml_direct"
#     elif "v1" in model_name.lower():
#         conv_mode = "llava_v1"
#     elif "mpt" in model_name.lower():
#         conv_mode = "mpt"
#     else:
#         conv_mode = "llava_v0"

#     if conv_mode_in_arg is not None and conv_mode != conv_mode_in_arg:
#         print(
#             "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
#                 conv_mode, args.conv_mode, args.conv_mode
#             )
#         )
#     else:
#         conv_mode_in_arg = conv_mode

#     conv = conv_templates[conv_mode_in_arg].copy()
#     conv.append_message(conv.roles[0], qs)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt()

#     # image_files = image_parser(args)
#     image_files = image_file.split(sep)
#     images = load_images(image_files)
#     image_sizes = [x.size for x in images]
#     images_tensor = process_images(
#         images,
#         image_processor,
#         model.config
#     ).to(model.device, dtype=torch.float16)

#     input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())

#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids,
#             images=images_tensor,
#             image_sizes=image_sizes,
#             do_sample=True if temperature > 0 else False,
#             temperature=temperature,
#             top_p=top_p,
#             num_beams=num_beams,
#             max_new_tokens=max_new_tokens,
#             use_cache=True,
#         )

#     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#     #print(outputs)
#     return outputs
    
def generate_caption(args):
    # video_name = os.listdir('/network_space/server128/shared/zhuoying/AnyDoor-main/taobao_video')
    # #video_name = ['20240506_80eceb1003a29279_461308186617_117110754046215_published_mp4_264_hd_taobao.mp4',  '20240406_1ed6f9b52607832f_456963067167_107031058147134_published_mp4_264_hd_taobao.mp4',  '20230528_37dbae2531ee6cbb_412379190723_33291891623696_published_mp4_264_hd_taobao.mp4',  '20230711_75e5a074fb02c4c6_418075780020_39828218478142_published_mp4_264_hd_taobao.mp4',  '20230411_844db1b59bf18285_405231025625_mp4_264_hd_taobao.mp4',  '20220727_a98c3136cbcb5dd4_369646076727_mp4_264_hd_taobao.mp4',    'Zs5ZGkX0vnDPiXSjae8_325909274017_mp4_264_hd.mp4']
    # for name in video_name:
    #     valid_index = []
    #     if not name.endswith('.mp4'):
    #         continue
    #     folder_name = name[:-4]
    #     number_maximum = 352
    #     for i in range(number_maximum):
    #         folder_path = os.path.join('/network_space/server128/shared/zhuoying/AnyDoor-main/taobao_video',folder_name)
    #         if os.path.exists(os.path.join(folder_path,f'{i}_human_mask.jpg')) and os.path.exists(os.path.join(folder_path,f'{i}_human_mask.jpg')):
    #             valid_index.append(i)
    #     valid_num = len(valid_index)
    #     sample_num_in_one_video = 0
    #     for i in range(valid_num):
    #         for j in range(valid_num):
    #             if i<=j:
    #                 continue
    #             if i-j <=5:  #### delete pairs with similar pose
    #                     continue
    #             if sample_num_in_one_video > 5000:
    #                     break
    #             pair_info = {}
    #             pair_info['name']  = os.path.join('/network_space/server128/shared/zhuoying/AnyDoor-main/taobao_video',folder_name)
    #             pair_info['idx_1'] = valid_index[i]
    #             pair_info['idx_2'] = valid_index[j]
    #             sample_num_in_one_video+=1
    #             save_info.append(pair_info)
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


    video_name = os.listdir('/network_space/server128/shared/zhuoying/AnyDoor-main/youtube_dataset')      
    for cat_name in video_name:    
        cat_dir = os.path.join('/network_space/server128/shared/zhuoying/AnyDoor-main/youtube_dataset',f'{cat_name}')
        obj_cat = cat_name
        cat_idxs = os.listdir(cat_dir)
        for cat_idx in cat_idxs:
            if cat_idx.endswith('.mp4'):
                continue
            video_folder = os.path.join(cat_dir,f'{cat_idx}')  
            
            if obj_cat == 'ski':
                obj_cat = 'skiboard'
            elif obj_cat == 'baseball':
                obj_cat = 'baseball bat'
            image_to_caption = None
            max_img_num = len(os.listdir(video_folder))
            query = f'There is a person and a/an {obj_cat} in the front of image, please describe their interaction with a single word verb, and form a complete sentence with person and {obj_cat} in the following format:a person is riding the horse.'
            video_save_info = {}
            if os.path.exists(os.path.join(video_folder,'caption.json')):
                continue
            for i in range(max_img_num):
                #folder_path = os.path.join('/network_space/server128/shared/zhuoying/AnyDoor-main/taobao_video',folder_name)
                if os.path.exists(os.path.join(video_folder,f'{i}_human_mask.jpg')) and os.path.exists(os.path.join(video_folder,f'{i}_object_mask.jpg')):
                    image_to_caption = os.path.join(video_folder,f'img_{i}.png')
                    # args.query = query
                    # args.image-file = image_to_caption
                    # gen_caption = eval_model(args)
                    #gen_caption  = my_eval_model(model_path = args.model_path, model_base = args.model_base,model_name= args.model_name,query=query,conv_mode_in_arg = args.conv_mode,image_file = image_to_caption,sep,temperature=args.temperature,top_p=args.top_p,num_beams=args.num_beams,max_new_tokens=args.max_new_tokens)
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
                    
                    image_files = [image_to_caption]
                    images = load_images(image_files)
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
                    video_save_info[f'img_{i}.png'] = outputs
            
            with open(os.path.join(video_folder,'caption.json'),'w') as f:
                json.dump(video_save_info,f)  
            print(f'finish {video_folder}')    
    
    with open('/network_space/server128/shared/zhuoying/AnyDoor-main/datasets/youtube_dataset_info_more_category_v2.json','w') as f:
        json.dump(save_info,f)

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
    args = parser.parse_args()

    #eval_model(args)
    generate_caption(args)