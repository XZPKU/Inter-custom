import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import argparse
import torch
import json
# from llava.constants import (
#     IMAGE_TOKEN_INDEX,
#     DEFAULT_IMAGE_TOKEN,
#     DEFAULT_IM_START_TOKEN,
#     DEFAULT_IM_END_TOKEN,
#     IMAGE_PLACEHOLDER,
# )
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import (
#     process_images,
#     tokenizer_image_token,
#     get_model_name_from_path,
# )
import cv2
from PIL import Image
import numpy as np
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
    
    
    
    #print(outputs)
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
hico_text_label = {(4, 4): 'a photo of a person boarding an airplane',
                   (17, 4): 'a photo of a person directing an airplane',
                   (25, 4): 'a photo of a person exiting an airplane',
                   (30, 4): 'a photo of a person flying an airplane',
                   (41, 4): 'a photo of a person inspecting an airplane',
                   (52, 4): 'a photo of a person loading an airplane',
                   (76, 4): 'a photo of a person riding an airplane',
                   (87, 4): 'a photo of a person sitting on an airplane',
                   (111, 4): 'a photo of a person washing an airplane',
                   (57, 4): 'a photo of a person and an airplane', (8, 1): 'a photo of a person carrying a bicycle',
                   (36, 1): 'a photo of a person holding a bicycle',
                   (41, 1): 'a photo of a person inspecting a bicycle',
                   (43, 1): 'a photo of a person jumping a bicycle',
                   (37, 1): 'a photo of a person hopping on a bicycle',
                   (62, 1): 'a photo of a person parking a bicycle',
                   (71, 1): 'a photo of a person pushing a bicycle',
                   (75, 1): 'a photo of a person repairing a bicycle',
                   (76, 1): 'a photo of a person riding a bicycle',
                   (87, 1): 'a photo of a person sitting on a bicycle',
                   (98, 1): 'a photo of a person straddling a bicycle',
                   (110, 1): 'a photo of a person walking a bicycle',
                   (111, 1): 'a photo of a person washing a bicycle', (57, 1): 'a photo of a person and a bicycle',
                   (10, 14): 'a photo of a person chasing a bird', (26, 14): 'a photo of a person feeding a bird',
                   (36, 14): 'a photo of a person holding a bird', (65, 14): 'a photo of a person petting a bird',
                   (74, 14): 'a photo of a person releasing a bird',
                   (112, 14): 'a photo of a person watching a bird', (57, 14): 'a photo of a person and a bird',
                   (4, 8): 'a photo of a person boarding a boat', (21, 8): 'a photo of a person driving a boat',
                   (25, 8): 'a photo of a person exiting a boat', (41, 8): 'a photo of a person inspecting a boat',
                   (43, 8): 'a photo of a person jumping a boat', (47, 8): 'a photo of a person launching a boat',
                   (75, 8): 'a photo of a person repairing a boat', (76, 8): 'a photo of a person riding a boat',
                   (77, 8): 'a photo of a person rowing a boat', (79, 8): 'a photo of a person sailing a boat',
                   (87, 8): 'a photo of a person sitting on a boat',
                   (93, 8): 'a photo of a person standing on a boat', (105, 8): 'a photo of a person tying a boat',
                   (111, 8): 'a photo of a person washing a boat', (57, 8): 'a photo of a person and a boat',
                   (8, 39): 'a photo of a person carrying a bottle',
                   (20, 39): 'a photo of a person drinking with a bottle',
                   (36, 39): 'a photo of a person holding a bottle',
                   (41, 39): 'a photo of a person inspecting a bottle',
                   (48, 39): 'a photo of a person licking a bottle',
                   (58, 39): 'a photo of a person opening a bottle',
                   (69, 39): 'a photo of a person pouring a bottle', (57, 39): 'a photo of a person and a bottle',
                   (4, 5): 'a photo of a person boarding a bus', (17, 5): 'a photo of a person directing a bus',
                   (21, 5): 'a photo of a person driving a bus', (25, 5): 'a photo of a person exiting a bus',
                   (41, 5): 'a photo of a person inspecting a bus', (52, 5): 'a photo of a person loading a bus',
                   (76, 5): 'a photo of a person riding a bus', (87, 5): 'a photo of a person sitting on a bus',
                   (111, 5): 'a photo of a person washing a bus', (113, 5): 'a photo of a person waving a bus',
                   (57, 5): 'a photo of a person and a bus', (4, 2): 'a photo of a person boarding a car',
                   (17, 2): 'a photo of a person directing a car', (21, 2): 'a photo of a person driving a car',
                   (38, 2): 'a photo of a person hosing a car', (41, 2): 'a photo of a person inspecting a car',
                   (43, 2): 'a photo of a person jumping a car', (52, 2): 'a photo of a person loading a car',
                   (62, 2): 'a photo of a person parking a car', (76, 2): 'a photo of a person riding a car',
                   (111, 2): 'a photo of a person washing a car', (57, 2): 'a photo of a person and a car',
                   (22, 15): 'a photo of a person drying a cat', (26, 15): 'a photo of a person feeding a cat',
                   (36, 15): 'a photo of a person holding a cat', (39, 15): 'a photo of a person hugging a cat',
                   (45, 15): 'a photo of a person kissing a cat', (65, 15): 'a photo of a person petting a cat',
                   (80, 15): 'a photo of a person scratching a cat', (111, 15): 'a photo of a person washing a cat',
                   (10, 15): 'a photo of a person chasing a cat', (57, 15): 'a photo of a person and a cat',
                   (8, 56): 'a photo of a person carrying a chair', (36, 56): 'a photo of a person holding a chair',
                   (49, 56): 'a photo of a person lying on a chair',
                   (87, 56): 'a photo of a person sitting on a chair',
                   (93, 56): 'a photo of a person standing on a chair', (57, 56): 'a photo of a person and a chair',
                   (8, 57): 'a photo of a person carrying a couch',
                   (49, 57): 'a photo of a person lying on a couch',
                   (87, 57): 'a photo of a person sitting on a couch', (57, 57): 'a photo of a person and a couch',
                   (26, 19): 'a photo of a person feeding a cow', (34, 19): 'a photo of a person herding a cow',
                   (36, 19): 'a photo of a person holding a cow', (39, 19): 'a photo of a person hugging a cow',
                   (45, 19): 'a photo of a person kissing a cow', (46, 19): 'a photo of a person lassoing a cow',
                   (55, 19): 'a photo of a person milking a cow', (65, 19): 'a photo of a person petting a cow',
                   (76, 19): 'a photo of a person riding a cow', (110, 19): 'a photo of a person walking a cow',
                   (57, 19): 'a photo of a person and a cow',
                   (12, 60): 'a photo of a person cleaning a dining table',
                   (24, 60): 'a photo of a person eating at a dining table',
                   (86, 60): 'a photo of a person sitting at a dining table',
                   (57, 60): 'a photo of a person and a dining table',
                   (8, 16): 'a photo of a person carrying a dog', (22, 16): 'a photo of a person drying a dog',
                   (26, 16): 'a photo of a person feeding a dog', (33, 16): 'a photo of a person grooming a dog',
                   (36, 16): 'a photo of a person holding a dog', (38, 16): 'a photo of a person hosing a dog',
                   (39, 16): 'a photo of a person hugging a dog', (41, 16): 'a photo of a person inspecting a dog',
                   (45, 16): 'a photo of a person kissing a dog', (65, 16): 'a photo of a person petting a dog',
                   (78, 16): 'a photo of a person running a dog', (80, 16): 'a photo of a person scratching a dog',
                   (98, 16): 'a photo of a person straddling a dog',
                   (107, 16): 'a photo of a person training a dog', (110, 16): 'a photo of a person walking a dog',
                   (111, 16): 'a photo of a person washing a dog', (10, 16): 'a photo of a person chasing a dog',
                   (57, 16): 'a photo of a person and a dog', (26, 17): 'a photo of a person feeding a horse',
                   (33, 17): 'a photo of a person grooming a horse',
                   (36, 17): 'a photo of a person holding a horse', (39, 17): 'a photo of a person hugging a horse',
                   (43, 17): 'a photo of a person jumping a horse', (45, 17): 'a photo of a person kissing a horse',
                   (52, 17): 'a photo of a person loading a horse',
                   (37, 17): 'a photo of a person hopping on a horse',
                   (65, 17): 'a photo of a person petting a horse', (72, 17): 'a photo of a person racing a horse',
                   (76, 17): 'a photo of a person riding a horse', (78, 17): 'a photo of a person running a horse',
                   (98, 17): 'a photo of a person straddling a horse',
                   (107, 17): 'a photo of a person training a horse',
                   (110, 17): 'a photo of a person walking a horse',
                   (111, 17): 'a photo of a person washing a horse', (57, 17): 'a photo of a person and a horse',
                   (36, 3): 'a photo of a person holding a motorcycle',
                   (41, 3): 'a photo of a person inspecting a motorcycle',
                   (43, 3): 'a photo of a person jumping a motorcycle',
                   (37, 3): 'a photo of a person hopping on a motorcycle',
                   (62, 3): 'a photo of a person parking a motorcycle',
                   (71, 3): 'a photo of a person pushing a motorcycle',
                   (72, 3): 'a photo of a person racing a motorcycle',
                   (76, 3): 'a photo of a person riding a motorcycle',
                   (87, 3): 'a photo of a person sitting on a motorcycle',
                   (98, 3): 'a photo of a person straddling a motorcycle',
                   (108, 3): 'a photo of a person turning a motorcycle',
                   (110, 3): 'a photo of a person walking a motorcycle',
                   (111, 3): 'a photo of a person washing a motorcycle',
                   (57, 3): 'a photo of a person and a motorcycle', (8, 0): 'a photo of a person carrying a person',
                   (31, 0): 'a photo of a person greeting a person',
                   (36, 0): 'a photo of a person holding a person', (39, 0): 'a photo of a person hugging a person',
                   (45, 0): 'a photo of a person kissing a person',
                   (92, 0): 'a photo of a person stabbing a person',
                   (100, 0): 'a photo of a person tagging a person',
                   (102, 0): 'a photo of a person teaching a person',
                   (48, 0): 'a photo of a person licking a person', (57, 0): 'a photo of a person and a person',
                   (8, 58): 'a photo of a person carrying a potted plant',
                   (36, 58): 'a photo of a person holding a potted plant',
                   (38, 58): 'a photo of a person hosing a potted plant',
                   (57, 58): 'a photo of a person and a potted plant',
                   (8, 18): 'a photo of a person carrying a sheep', (26, 18): 'a photo of a person feeding a sheep',
                   (34, 18): 'a photo of a person herding a sheep', (36, 18): 'a photo of a person holding a sheep',
                   (39, 18): 'a photo of a person hugging a sheep', (45, 18): 'a photo of a person kissing a sheep',
                   (65, 18): 'a photo of a person petting a sheep', (76, 18): 'a photo of a person riding a sheep',
                   (83, 18): 'a photo of a person shearing a sheep',
                   (110, 18): 'a photo of a person walking a sheep',
                   (111, 18): 'a photo of a person washing a sheep', (57, 18): 'a photo of a person and a sheep',
                   (4, 6): 'a photo of a person boarding a train', (21, 6): 'a photo of a person driving a train',
                   (25, 6): 'a photo of a person exiting a train', (52, 6): 'a photo of a person loading a train',
                   (76, 6): 'a photo of a person riding a train', (87, 6): 'a photo of a person sitting on a train',
                   (111, 6): 'a photo of a person washing a train', (57, 6): 'a photo of a person and a train',
                   (13, 62): 'a photo of a person controlling a tv', (75, 62): 'a photo of a person repairing a tv',
                   (112, 62): 'a photo of a person watching a tv', (57, 62): 'a photo of a person and a tv',
                   (7, 47): 'a photo of a person buying an apple', (15, 47): 'a photo of a person cutting an apple',
                   (23, 47): 'a photo of a person eating an apple',
                   (36, 47): 'a photo of a person holding an apple',
                   (41, 47): 'a photo of a person inspecting an apple',
                   (64, 47): 'a photo of a person peeling an apple',
                   (66, 47): 'a photo of a person picking an apple',
                   (89, 47): 'a photo of a person smelling an apple',
                   (111, 47): 'a photo of a person washing an apple', (57, 47): 'a photo of a person and an apple',
                   (8, 24): 'a photo of a person carrying a backpack',
                   (36, 24): 'a photo of a person holding a backpack',
                   (41, 24): 'a photo of a person inspecting a backpack',
                   (58, 24): 'a photo of a person opening a backpack',
                   (114, 24): 'a photo of a person wearing a backpack',
                   (57, 24): 'a photo of a person and a backpack', (7, 46): 'a photo of a person buying a banana',
                   (8, 46): 'a photo of a person carrying a banana',
                   (15, 46): 'a photo of a person cutting a banana',
                   (23, 46): 'a photo of a person eating a banana',
                   (36, 46): 'a photo of a person holding a banana',
                   (41, 46): 'a photo of a person inspecting a banana',
                   (64, 46): 'a photo of a person peeling a banana',
                   (66, 46): 'a photo of a person picking a banana',
                   (89, 46): 'a photo of a person smelling a banana', (57, 46): 'a photo of a person and a banana',
                   (5, 34): 'a photo of a person breaking a baseball bat',
                   (8, 34): 'a photo of a person carrying a baseball bat',
                   (36, 34): 'a photo of a person holding a baseball bat',
                   (84, 34): 'a photo of a person signing a baseball bat',
                   (99, 34): 'a photo of a person swinging a baseball bat',
                   (104, 34): 'a photo of a person throwing a baseball bat',
                   (115, 34): 'a photo of a person wielding a baseball bat',
                   (57, 34): 'a photo of a person and a baseball bat',
                   (36, 35): 'a photo of a person holding a baseball glove',
                   (114, 35): 'a photo of a person wearing a baseball glove',
                   (57, 35): 'a photo of a person and a baseball glove',
                   (26, 21): 'a photo of a person feeding a bear', (40, 21): 'a photo of a person hunting a bear',
                   (112, 21): 'a photo of a person watching a bear', (57, 21): 'a photo of a person and a bear',
                   (12, 59): 'a photo of a person cleaning a bed', (49, 59): 'a photo of a person lying on a bed',
                   (87, 59): 'a photo of a person sitting on a bed', (57, 59): 'a photo of a person and a bed',
                   (41, 13): 'a photo of a person inspecting a bench',
                   (49, 13): 'a photo of a person lying on a bench',
                   (87, 13): 'a photo of a person sitting on a bench', (57, 13): 'a photo of a person and a bench',
                   (8, 73): 'a photo of a person carrying a book', (36, 73): 'a photo of a person holding a book',
                   (58, 73): 'a photo of a person opening a book', (73, 73): 'a photo of a person reading a book',
                   (57, 73): 'a photo of a person and a book', (36, 45): 'a photo of a person holding a bowl',
                   (96, 45): 'a photo of a person stirring a bowl', (111, 45): 'a photo of a person washing a bowl',
                   (48, 45): 'a photo of a person licking a bowl', (57, 45): 'a photo of a person and a bowl',
                   (15, 50): 'a photo of a person cutting a broccoli',
                   (23, 50): 'a photo of a person eating a broccoli',
                   (36, 50): 'a photo of a person holding a broccoli',
                   (89, 50): 'a photo of a person smelling a broccoli',
                   (96, 50): 'a photo of a person stirring a broccoli',
                   (111, 50): 'a photo of a person washing a broccoli',
                   (57, 50): 'a photo of a person and a broccoli', (3, 55): 'a photo of a person blowing a cake',
                   (8, 55): 'a photo of a person carrying a cake', (15, 55): 'a photo of a person cutting a cake',
                   (23, 55): 'a photo of a person eating a cake', (36, 55): 'a photo of a person holding a cake',
                   (51, 55): 'a photo of a person lighting a cake', (54, 55): 'a photo of a person making a cake',
                   (67, 55): 'a photo of a person picking up a cake', (57, 55): 'a photo of a person and a cake',
                   (8, 51): 'a photo of a person carrying a carrot',
                   (14, 51): 'a photo of a person cooking a carrot',
                   (15, 51): 'a photo of a person cutting a carrot',
                   (23, 51): 'a photo of a person eating a carrot',
                   (36, 51): 'a photo of a person holding a carrot',
                   (64, 51): 'a photo of a person peeling a carrot',
                   (89, 51): 'a photo of a person smelling a carrot',
                   (96, 51): 'a photo of a person stirring a carrot',
                   (111, 51): 'a photo of a person washing a carrot', (57, 51): 'a photo of a person and a carrot',
                   (8, 67): 'a photo of a person carrying a cell phone',
                   (36, 67): 'a photo of a person holding a cell phone',
                   (73, 67): 'a photo of a person reading a cell phone',
                   (75, 67): 'a photo of a person repairing a cell phone',
                   (101, 67): 'a photo of a person talking on a cell phone',
                   (103, 67): 'a photo of a person texting on a cell phone',
                   (57, 67): 'a photo of a person and a cell phone',
                   (11, 74): 'a photo of a person checking a clock',
                   (36, 74): 'a photo of a person holding a clock',
                   (75, 74): 'a photo of a person repairing a clock',
                   (82, 74): 'a photo of a person setting a clock', (57, 74): 'a photo of a person and a clock',
                   (8, 41): 'a photo of a person carrying a cup',
                   (20, 41): 'a photo of a person drinking with a cup',
                   (36, 41): 'a photo of a person holding a cup', (41, 41): 'a photo of a person inspecting a cup',
                   (69, 41): 'a photo of a person pouring a cup', (85, 41): 'a photo of a person sipping a cup',
                   (89, 41): 'a photo of a person smelling a cup', (27, 41): 'a photo of a person filling a cup',
                   (111, 41): 'a photo of a person washing a cup', (57, 41): 'a photo of a person and a cup',
                   (7, 54): 'a photo of a person buying a donut', (8, 54): 'a photo of a person carrying a donut',
                   (23, 54): 'a photo of a person eating a donut', (36, 54): 'a photo of a person holding a donut',
                   (54, 54): 'a photo of a person making a donut',
                   (67, 54): 'a photo of a person picking up a donut',
                   (89, 54): 'a photo of a person smelling a donut', (57, 54): 'a photo of a person and a donut',
                   (26, 20): 'a photo of a person feeding an elephant',
                   (36, 20): 'a photo of a person holding an elephant',
                   (38, 20): 'a photo of a person hosing an elephant',
                   (39, 20): 'a photo of a person hugging an elephant',
                   (45, 20): 'a photo of a person kissing an elephant',
                   (37, 20): 'a photo of a person hopping on an elephant',
                   (65, 20): 'a photo of a person petting an elephant',
                   (76, 20): 'a photo of a person riding an elephant',
                   (110, 20): 'a photo of a person walking an elephant',
                   (111, 20): 'a photo of a person washing an elephant',
                   (112, 20): 'a photo of a person watching an elephant',
                   (57, 20): 'a photo of a person and an elephant',
                   (39, 10): 'a photo of a person hugging a fire hydrant',
                   (41, 10): 'a photo of a person inspecting a fire hydrant',
                   (58, 10): 'a photo of a person opening a fire hydrant',
                   (61, 10): 'a photo of a person painting a fire hydrant',
                   (57, 10): 'a photo of a person and a fire hydrant',
                   (36, 42): 'a photo of a person holding a fork', (50, 42): 'a photo of a person lifting a fork',
                   (95, 42): 'a photo of a person sticking a fork', (48, 42): 'a photo of a person licking a fork',
                   (111, 42): 'a photo of a person washing a fork', (57, 42): 'a photo of a person and a fork',
                   (2, 29): 'a photo of a person blocking a frisbee',
                   (9, 29): 'a photo of a person catching a frisbee',
                   (36, 29): 'a photo of a person holding a frisbee',
                   (90, 29): 'a photo of a person spinning a frisbee',
                   (104, 29): 'a photo of a person throwing a frisbee',
                   (57, 29): 'a photo of a person and a frisbee', (26, 23): 'a photo of a person feeding a giraffe',
                   (45, 23): 'a photo of a person kissing a giraffe',
                   (65, 23): 'a photo of a person petting a giraffe',
                   (76, 23): 'a photo of a person riding a giraffe',
                   (112, 23): 'a photo of a person watching a giraffe',
                   (57, 23): 'a photo of a person and a giraffe',
                   (36, 78): 'a photo of a person holding a hair drier',
                   (59, 78): 'a photo of a person operating a hair drier',
                   (75, 78): 'a photo of a person repairing a hair drier',
                   (57, 78): 'a photo of a person and a hair drier',
                   (8, 26): 'a photo of a person carrying a handbag',
                   (36, 26): 'a photo of a person holding a handbag',
                   (41, 26): 'a photo of a person inspecting a handbag',
                   (57, 26): 'a photo of a person and a handbag', (8, 52): 'a photo of a person carrying a hot dog',
                   (14, 52): 'a photo of a person cooking a hot dog',
                   (15, 52): 'a photo of a person cutting a hot dog',
                   (23, 52): 'a photo of a person eating a hot dog',
                   (36, 52): 'a photo of a person holding a hot dog',
                   (54, 52): 'a photo of a person making a hot dog', (57, 52): 'a photo of a person and a hot dog',
                   (8, 66): 'a photo of a person carrying a keyboard',
                   (12, 66): 'a photo of a person cleaning a keyboard',
                   (36, 66): 'a photo of a person holding a keyboard',
                   (109, 66): 'a photo of a person typing on a keyboard',
                   (57, 66): 'a photo of a person and a keyboard', (1, 33): 'a photo of a person assembling a kite',
                   (8, 33): 'a photo of a person carrying a kite', (30, 33): 'a photo of a person flying a kite',
                   (36, 33): 'a photo of a person holding a kite',
                   (41, 33): 'a photo of a person inspecting a kite',
                   (47, 33): 'a photo of a person launching a kite', (70, 33): 'a photo of a person pulling a kite',
                   (57, 33): 'a photo of a person and a kite', (16, 43): 'a photo of a person cutting with a knife',
                   (36, 43): 'a photo of a person holding a knife',
                   (95, 43): 'a photo of a person sticking a knife',
                   (111, 43): 'a photo of a person washing a knife',
                   (115, 43): 'a photo of a person wielding a knife',
                   (48, 43): 'a photo of a person licking a knife', (57, 43): 'a photo of a person and a knife',
                   (36, 63): 'a photo of a person holding a laptop',
                   (58, 63): 'a photo of a person opening a laptop',
                   (73, 63): 'a photo of a person reading a laptop',
                   (75, 63): 'a photo of a person repairing a laptop',
                   (109, 63): 'a photo of a person typing on a laptop',
                   (57, 63): 'a photo of a person and a laptop',
                   (12, 68): 'a photo of a person cleaning a microwave',
                   (58, 68): 'a photo of a person opening a microwave',
                   (59, 68): 'a photo of a person operating a microwave',
                   (57, 68): 'a photo of a person and a microwave',
                   (13, 64): 'a photo of a person controlling a mouse',
                   (36, 64): 'a photo of a person holding a mouse',
                   (75, 64): 'a photo of a person repairing a mouse', (57, 64): 'a photo of a person and a mouse',
                   (7, 49): 'a photo of a person buying an orange',
                   (15, 49): 'a photo of a person cutting an orange',
                   (23, 49): 'a photo of a person eating an orange',
                   (36, 49): 'a photo of a person holding an orange',
                   (41, 49): 'a photo of a person inspecting an orange',
                   (64, 49): 'a photo of a person peeling an orange',
                   (66, 49): 'a photo of a person picking an orange',
                   (91, 49): 'a photo of a person squeezing an orange',
                   (111, 49): 'a photo of a person washing an orange',
                   (57, 49): 'a photo of a person and an orange', (12, 69): 'a photo of a person cleaning an oven',
                   (36, 69): 'a photo of a person holding an oven',
                   (41, 69): 'a photo of a person inspecting an oven',
                   (58, 69): 'a photo of a person opening an oven',
                   (75, 69): 'a photo of a person repairing an oven',
                   (59, 69): 'a photo of a person operating an oven', (57, 69): 'a photo of a person and an oven',
                   (11, 12): 'a photo of a person checking a parking meter',
                   (63, 12): 'a photo of a person paying a parking meter',
                   (75, 12): 'a photo of a person repairing a parking meter',
                   (57, 12): 'a photo of a person and a parking meter',
                   (7, 53): 'a photo of a person buying a pizza', (8, 53): 'a photo of a person carrying a pizza',
                   (14, 53): 'a photo of a person cooking a pizza', (15, 53): 'a photo of a person cutting a pizza',
                   (23, 53): 'a photo of a person eating a pizza', (36, 53): 'a photo of a person holding a pizza',
                   (54, 53): 'a photo of a person making a pizza',
                   (67, 53): 'a photo of a person picking up a pizza',
                   (88, 53): 'a photo of a person sliding a pizza',
                   (89, 53): 'a photo of a person smelling a pizza', (57, 53): 'a photo of a person and a pizza',
                   (12, 72): 'a photo of a person cleaning a refrigerator',
                   (36, 72): 'a photo of a person holding a refrigerator',
                   (56, 72): 'a photo of a person moving a refrigerator',
                   (58, 72): 'a photo of a person opening a refrigerator',
                   (57, 72): 'a photo of a person and a refrigerator',
                   (36, 65): 'a photo of a person holding a remote',
                   (68, 65): 'a photo of a person pointing a remote',
                   (99, 65): 'a photo of a person swinging a remote', (57, 65): 'a photo of a person and a remote',
                   (8, 48): 'a photo of a person carrying a sandwich',
                   (14, 48): 'a photo of a person cooking a sandwich',
                   (15, 48): 'a photo of a person cutting a sandwich',
                   (23, 48): 'a photo of a person eating a sandwich',
                   (36, 48): 'a photo of a person holding a sandwich',
                   (54, 48): 'a photo of a person making a sandwich',
                   (57, 48): 'a photo of a person and a sandwich',
                   (16, 76): 'a photo of a person cutting with a scissors',
                   (36, 76): 'a photo of a person holding a scissors',
                   (58, 76): 'a photo of a person opening a scissors',
                   (57, 76): 'a photo of a person and a scissors', (12, 71): 'a photo of a person cleaning a sink',
                   (75, 71): 'a photo of a person repairing a sink',
                   (111, 71): 'a photo of a person washing a sink', (57, 71): 'a photo of a person and a sink',
                   (8, 36): 'a photo of a person carrying a skateboard',
                   (28, 36): 'a photo of a person flipping a skateboard',
                   (32, 36): 'a photo of a person grinding a skateboard',
                   (36, 36): 'a photo of a person holding a skateboard',
                   (43, 36): 'a photo of a person jumping a skateboard',
                   (67, 36): 'a photo of a person picking up a skateboard',
                   (76, 36): 'a photo of a person riding a skateboard',
                   (87, 36): 'a photo of a person sitting on a skateboard',
                   (93, 36): 'a photo of a person standing on a skateboard',
                   (57, 36): 'a photo of a person and a skateboard',
                   (0, 30): 'a photo of a person adjusting a skis', (8, 30): 'a photo of a person carrying a skis',
                   (36, 30): 'a photo of a person holding a skis',
                   (41, 30): 'a photo of a person inspecting a skis',
                   (43, 30): 'a photo of a person jumping a skis',
                   (67, 30): 'a photo of a person picking up a skis',
                   (75, 30): 'a photo of a person repairing a skis', (76, 30): 'a photo of a person riding a skis',
                   (93, 30): 'a photo of a person standing on a skis',
                   (114, 30): 'a photo of a person wearing a skis', (57, 30): 'a photo of a person and a skis',
                   (0, 31): 'a photo of a person adjusting a snowboard',
                   (8, 31): 'a photo of a person carrying a snowboard',
                   (32, 31): 'a photo of a person grinding a snowboard',
                   (36, 31): 'a photo of a person holding a snowboard',
                   (43, 31): 'a photo of a person jumping a snowboard',
                   (76, 31): 'a photo of a person riding a snowboard',
                   (93, 31): 'a photo of a person standing on a snowboard',
                   (114, 31): 'a photo of a person wearing a snowboard',
                   (57, 31): 'a photo of a person and a snowboard', (36, 44): 'a photo of a person holding a spoon',
                   (48, 44): 'a photo of a person licking a spoon',
                   (111, 44): 'a photo of a person washing a spoon',
                   (85, 44): 'a photo of a person sipping a spoon', (57, 44): 'a photo of a person and a spoon',
                   (2, 32): 'a photo of a person blocking a sports ball',
                   (8, 32): 'a photo of a person carrying a sports ball',
                   (9, 32): 'a photo of a person catching a sports ball',
                   (19, 32): 'a photo of a person dribbling a sports ball',
                   (35, 32): 'a photo of a person hitting a sports ball',
                   (36, 32): 'a photo of a person holding a sports ball',
                   (41, 32): 'a photo of a person inspecting a sports ball',
                   (44, 32): 'a photo of a person kicking a sports ball',
                   (67, 32): 'a photo of a person picking up a sports ball',
                   (81, 32): 'a photo of a person serving a sports ball',
                   (84, 32): 'a photo of a person signing a sports ball',
                   (90, 32): 'a photo of a person spinning a sports ball',
                   (104, 32): 'a photo of a person throwing a sports ball',
                   (57, 32): 'a photo of a person and a sports ball',
                   (36, 11): 'a photo of a person holding a stop sign',
                   (94, 11): 'a photo of a person standing under a stop sign',
                   (97, 11): 'a photo of a person stopping at a stop sign',
                   (57, 11): 'a photo of a person and a stop sign',
                   (8, 28): 'a photo of a person carrying a suitcase',
                   (18, 28): 'a photo of a person dragging a suitcase',
                   (36, 28): 'a photo of a person holding a suitcase',
                   (39, 28): 'a photo of a person hugging a suitcase',
                   (52, 28): 'a photo of a person loading a suitcase',
                   (58, 28): 'a photo of a person opening a suitcase',
                   (60, 28): 'a photo of a person packing a suitcase',
                   (67, 28): 'a photo of a person picking up a suitcase',
                   (116, 28): 'a photo of a person zipping a suitcase',
                   (57, 28): 'a photo of a person and a suitcase',
                   (8, 37): 'a photo of a person carrying a surfboard',
                   (18, 37): 'a photo of a person dragging a surfboard',
                   (36, 37): 'a photo of a person holding a surfboard',
                   (41, 37): 'a photo of a person inspecting a surfboard',
                   (43, 37): 'a photo of a person jumping a surfboard',
                   (49, 37): 'a photo of a person lying on a surfboard',
                   (52, 37): 'a photo of a person loading a surfboard',
                   (76, 37): 'a photo of a person riding a surfboard',
                   (93, 37): 'a photo of a person standing on a surfboard',
                   (87, 37): 'a photo of a person sitting on a surfboard',
                   (111, 37): 'a photo of a person washing a surfboard',
                   (57, 37): 'a photo of a person and a surfboard',
                   (8, 77): 'a photo of a person carrying a teddy bear',
                   (36, 77): 'a photo of a person holding a teddy bear',
                   (39, 77): 'a photo of a person hugging a teddy bear',
                   (45, 77): 'a photo of a person kissing a teddy bear',
                   (57, 77): 'a photo of a person and a teddy bear',
                   (8, 38): 'a photo of a person carrying a tennis racket',
                   (36, 38): 'a photo of a person holding a tennis racket',
                   (41, 38): 'a photo of a person inspecting a tennis racket',
                   (99, 38): 'a photo of a person swinging a tennis racket',
                   (57, 38): 'a photo of a person and a tennis racket',
                   (0, 27): 'a photo of a person adjusting a tie', (15, 27): 'a photo of a person cutting a tie',
                   (36, 27): 'a photo of a person holding a tie', (41, 27): 'a photo of a person inspecting a tie',
                   (70, 27): 'a photo of a person pulling a tie', (105, 27): 'a photo of a person tying a tie',
                   (114, 27): 'a photo of a person wearing a tie', (57, 27): 'a photo of a person and a tie',
                   (36, 70): 'a photo of a person holding a toaster',
                   (59, 70): 'a photo of a person operating a toaster',
                   (75, 70): 'a photo of a person repairing a toaster',
                   (57, 70): 'a photo of a person and a toaster', (12, 61): 'a photo of a person cleaning a toilet',
                   (29, 61): 'a photo of a person flushing a toilet',
                   (58, 61): 'a photo of a person opening a toilet',
                   (75, 61): 'a photo of a person repairing a toilet',
                   (87, 61): 'a photo of a person sitting on a toilet',
                   (93, 61): 'a photo of a person standing on a toilet',
                   (111, 61): 'a photo of a person washing a toilet', (57, 61): 'a photo of a person and a toilet',
                   (6, 79): 'a photo of a person brushing with a toothbrush',
                   (36, 79): 'a photo of a person holding a toothbrush',
                   (111, 79): 'a photo of a person washing a toothbrush',
                   (57, 79): 'a photo of a person and a toothbrush',
                   (42, 9): 'a photo of a person installing a traffic light',
                   (75, 9): 'a photo of a person repairing a traffic light',
                   (94, 9): 'a photo of a person standing under a traffic light',
                   (97, 9): 'a photo of a person stopping at a traffic light',
                   (57, 9): 'a photo of a person and a traffic light',
                   (17, 7): 'a photo of a person directing a truck', (21, 7): 'a photo of a person driving a truck',
                   (41, 7): 'a photo of a person inspecting a truck',
                   (52, 7): 'a photo of a person loading a truck', (75, 7): 'a photo of a person repairing a truck',
                   (76, 7): 'a photo of a person riding a truck', (87, 7): 'a photo of a person sitting on a truck',
                   (111, 7): 'a photo of a person washing a truck', (57, 7): 'a photo of a person and a truck',
                   (8, 25): 'a photo of a person carrying a umbrella',
                   (36, 25): 'a photo of a person holding a umbrella',
                   (53, 25): 'a photo of a person losing a umbrella',
                   (58, 25): 'a photo of a person opening a umbrella',
                   (75, 25): 'a photo of a person repairing a umbrella',
                   (82, 25): 'a photo of a person setting a umbrella',
                   (94, 25): 'a photo of a person standing under a umbrella',
                   (57, 25): 'a photo of a person and a umbrella', (36, 75): 'a photo of a person holding a vase',
                   (54, 75): 'a photo of a person making a vase', (61, 75): 'a photo of a person painting a vase',
                   (57, 75): 'a photo of a person and a vase', (27, 40): 'a photo of a person filling a wine glass',
                   (36, 40): 'a photo of a person holding a wine glass',
                   (85, 40): 'a photo of a person sipping a wine glass',
                   (106, 40): 'a photo of a person toasting a wine glass',
                   (48, 40): 'a photo of a person licking a wine glass',
                   (111, 40): 'a photo of a person washing a wine glass',
                   (57, 40): 'a photo of a person and a wine glass',
                   (26, 22): 'a photo of a person feeding a zebra', (36, 22): 'a photo of a person holding a zebra',
                   (65, 22): 'a photo of a person petting a zebra',
                   (112, 22): 'a photo of a person watching a zebra', (57, 22): 'a photo of a person and a zebra'}
 
def generate_caption(args):
    
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


    
        
                
    sub_img = Image.open('/network_space/server128/shared/zhuoying/AnyDoor-main/hoi_samples/man.jpg')
    obj_img = Image.open('/network_space/server128/shared/zhuoying/AnyDoor-main/hoi_samples/motorcycle.jpg')
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
    query = f'There is a person and a motorcycle, please use three sentences to seperately describe (1) the haircut, body shape and appearance of person,(2) the shape and appearance of object. Following is a example: The person is a short hair, tall and thin man, and wearing a hat. The bicycle is a colorful mountain bicycle.'
    
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
    # images = [crop_hoi_image]
    # image_sizes = [x.size for x in images]
    # images_tensor = process_images(
    #     images,
    #     image_processor,
    #     model.config
    # ).to(model.device, dtype=torch.float16)

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
    with open(args.bank_path,'r') as f:
        mask_bank = json.load(f)
    stored_masks = mask_bank[str(args.target_hoi_category)]
    del mask_bank
    stored_ids = [item['image_name'] for item in stored_masks]
    stored_captions = [f"{item['llava_caption']}" for item in stored_masks]
    
    import clip
    device = 'cuda'
    model, preprocess = clip.load('/network_space/server129/lttt/hoid/checkpoints/pretrained_clip/ViT-L-14.pt', device=device)
    # stored_text = clip.tokenize(stored_captions).to(device)
    ans = []
    for cap in stored_captions:
        try:
            ans.append(clip.tokenize(cap))
        except:
            ans.append(clip.tokenize('a man and a object'))
    stored_captions = torch.stack(ans,dim=0).to(device).squeeze(1)
    #stored_text = torch.cat([clip.tokenize(c) for c in stored_captions]).to(device)
    image_text = clip.tokenize([outputs]).to(device)
    with torch.no_grad():
        stored_captions = model.encode_text(stored_captions)
        text_features = model.encode_text(image_text)
        sim = stored_captions@text_features.T
        most_sim = torch.argmax(sim,dim=0).item() 
        image_name = stored_ids[most_sim]       
    return Image.open(os.path.join(f'/network_space/server128/shared/xuzhu/process_data/{image_name}','human_mask.jpg')), Image.open(os.path.join(f'/network_space/server128/shared/xuzhu/process_data/{image_name}','object_mask.jpg'))
def test_hico(args):
    
    # disable_torch_init()

    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     args.model_path, args.model_base, model_name
    # )

    

    # if "llama-2" in model_name.lower():
    #     conv_mode = "llava_llama_2"
    # elif "mistral" in model_name.lower():
    #     conv_mode = "mistral_instruct"
    # elif "v1.6-34b" in model_name.lower():
    #     conv_mode = "chatml_direct"
    # elif "v1" in model_name.lower():
    #     conv_mode = "llava_v1"
    # elif "mpt" in model_name.lower():
    #     conv_mode = "mpt"
    # else:
    #     conv_mode = "llava_v0"

    # if args.conv_mode is not None and conv_mode != args.conv_mode:
    #     print(
    #         "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
    #             conv_mode, args.conv_mode, args.conv_mode
    #         )
    #     )
    # else:
    #     args.conv_mode = conv_mode


    from tqdm import tqdm
    # all_hoi_instances = []
    root_path = '/home/xuzhu/MoMA/data/process_data'
    hico_image_path = '/home/xuzhu/MoMA/data/hico_20160224_det/images/train2015'
    _valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
    _valid_verb_ids = list(range(1, 118))
    
    with open(os.path.join('/home/xuzhu/MoMA/data/hico_20160224_det/annotations/trainval_hico_ann.json')) as f:
        anno_info = json.load(f)
    all_mask_sotrage = {}
    with open(args.bank_path,'r') as f:
        mask_bank = json.load(f)
    
    for i in range(1,601):
        all_mask_sotrage[i] = []
    import clip
    device = 'cuda'
    model, preprocess = clip.load('/network_space/server129/lttt/hoid/checkpoints/pretrained_clip/ViT-L-14.pt', device=device)
    all_num = 0
    ans_1 = 0
    ans_3 = 0
    ans_5 = 0
    ans_10 = 0
    for anno_item in tqdm(anno_info):
        image_name = anno_item['file_name']
        instance_anno = anno_item['annotations']
        hoi_annotation = anno_item['hoi_annotation']
        
            
            
        hoi_instance_num = -1
        for hoi_instance in hoi_annotation:
            if all_num > 100:
                break
            hoi_instance_num += 1
            image_info_path = os.path.join(root_path,f'{image_name}_{hoi_instance_num}')
            query_info_path = os.path.join(image_info_path,'llava_caption.txt')
            if os.path.exists(query_info_path) and os.path.getsize(query_info_path) > 0:
                all_num +=1
                
                with open(query_info_path,'r') as f:
                    query_info  = f.read()
                #query_info = query_info_.split('.')[0] +  query_info_.split('.')[1]
                stored_masks = mask_bank[str(args.target_hoi_category)]
                #del mask_bank
                stored_ids = [item['image_name'] for item in stored_masks]
                #stored_captions = [f"{item['llava_caption'].split('.')[0]+item['llava_caption'].split('.')[1]}" for item in stored_masks]
                stored_captions = [f"{item['llava_caption']}" for item in stored_masks]
                stored_imagenames = [f"{item['image_name']}" for item in stored_masks]
                ans = []
                for cap in stored_captions:
                    try:
                        ans.append(clip.tokenize(cap))
                    except:
                        ans.append(clip.tokenize('a man and an object'))
                stored_captions = torch.stack(ans,dim=0).to(device).squeeze(1)
                #stored_text = torch.cat([clip.tokenize(c) for c in stored_captions]).to(device)
                try:
                    image_text = clip.tokenize([query_info]).to(device)
                except:
                    tokens = query_info.split()
                    truncated_tokens = tokens[:70]
                    query_info = ' '.join(truncated_tokens)
                    image_text = clip.tokenize(['a man and an object']).to(device)
                with torch.no_grad():
                    stored_captions = model.encode_text(stored_captions)
                    text_features = model.encode_text(image_text)
                    sim = stored_captions@text_features.T
                    arg_1 = torch.topk(sim,10,dim=0).indices
                    arg_3  =torch.topk(sim,30,dim=0).indices
                    arg_5  =torch.topk(sim,50,dim = 0).indices
                    arg_10 = torch.topk(sim,1000,dim=0).indices
                    
                    acc_1 = get_val(arg_1,stored_ids,image_name)
                    acc_3 = get_val(arg_3,stored_ids,image_name)
                    acc_5 = get_val(arg_5,stored_ids,image_name)
                    acc_10 = get_val(arg_10,stored_ids,image_name)
                    
                    ans_1 +=acc_1
                    ans_3 +=acc_3
                    ans_5 +=acc_5
                    ans_10 +=acc_10
                    #idx_i = 
                    # most_sim = torch.argmax(sim,dim=0).item() 
                    # image_name = stored_ids[most_sim]   
    print(f'all_num is {all_num}')
    print(f'acc_1 is {ans_1}, ratio = {ans_1/all_num}')           
    print(f'acc_3 is {ans_3}, ratio = {ans_3/all_num}')   
    print(f'acc_5 is {ans_5}, ratio = {ans_5/all_num}')   
    print(f'acc_10 is {ans_10}, ratio = {ans_1/all_num}')        
            
def get_val(tensor, stored_ids,image_name):
    flg=False
    for i in range(tensor.size(0)):
        value = tensor[i, 0].item() 
        stored_val = stored_ids[value]
        if image_name in stored_val:
            flg  = True
    if flg:
        return 1
    return 0
                  
    # for item in             
    # sub_img = Image.open('/network_space/server128/shared/zhuoying/AnyDoor-main/hoi_samples/man.jpg')
    # obj_img = Image.open('/network_space/server128/shared/zhuoying/AnyDoor-main/hoi_samples/motorcycle.jpg')
    # width = sub_img.width + obj_img.width
    # height = max(sub_img.height, obj_img.height)
    # result_img = Image.new('RGB', (width, height))
    # result_img.paste(sub_img, (0, 0))
    # result_img.paste(obj_img, (sub_img.width, 0))
    # result_img.save('concat.jpg')
    # images = [Image.fromarray(cv2.resize(np.array(result_img),(256,256)))]
    # image_sizes = [x.size for x in images]
    # images_tensor = process_images(
    #     images,
    #     image_processor,
    #     model.config
    # ).to(model.device, dtype=torch.float16)
    # query = f'There is a person and a motorcycle, please use three sentences to seperately describe (1) the haircut, body shape and appearance of person,(2) the shape and appearance of object. Following is a example: The person is a short hair, tall and thin man, and wearing a hat. The bicycle is a colorful mountain bicycle.'
    
    # qs = query
    # image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    # if IMAGE_PLACEHOLDER in qs:
    #     if model.config.mm_use_im_start_end:
    #         qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
    #     else:
    #         qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    # else:
    #     if model.config.mm_use_im_start_end:
    #         qs = image_token_se + "\n" + qs
    #     else:
    #         qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    # conv = conv_templates[args.conv_mode].copy()
    # conv.append_message(conv.roles[0], qs)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()
    
    # # image_files = [image_to_caption]
    # # images = load_images(image_files)
    # # images = [crop_hoi_image]
    # # image_sizes = [x.size for x in images]
    # # images_tensor = process_images(
    # #     images,
    # #     image_processor,
    # #     model.config
    # # ).to(model.device, dtype=torch.float16)

    # input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())

    # with torch.inference_mode():
    #     output_ids = model.generate(
    #         input_ids,
    #         images=images_tensor,
    #         image_sizes=image_sizes,
    #         do_sample=True if args.temperature > 0 else False,
    #         temperature=args.temperature,
    #         top_p=args.top_p,
    #         num_beams=args.num_beams,
    #         max_new_tokens=args.max_new_tokens,
    #         use_cache=True,
    #     )
    
    # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(outputs)
    # with open(args.bank_path,'r') as f:
    #     mask_bank = json.load(f)
    # stored_masks = mask_bank[str(args.target_hoi_category)]
    # del mask_bank
    # stored_ids = [item['image_name'] for item in stored_masks]
    # stored_captions = [f"{item['llava_caption']}" for item in stored_masks]
    
    # import clip
    # device = 'cuda'
    # model, preprocess = clip.load('/network_space/server129/lttt/hoid/checkpoints/pretrained_clip/ViT-L-14.pt', device=device)
    # # stored_text = clip.tokenize(stored_captions).to(device)
    
    #return Image.open(os.path.join(f'/network_space/server128/shared/xuzhu/process_data/{image_name}','human_mask.jpg')), Image.open(os.path.join(f'/network_space/server128/shared/xuzhu/process_data/{image_name}','object_mask.jpg'))

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
    parser.add_argument("--target_hoi_category",type = int,default=155,required=False)
    parser.add_argument("--target_obj_category",type=str,default='motorcycle',required=False)
    parser.add_argument('--target_sub_img_path',type=str,default = '/network_space/server128/shared/zhuoying/AnyDoor-main/hoi_samples/man.jpg',required=False)
    parser.add_argument('--target_obj_img_path',type=str,default = '/network_space/server128/shared/zhuoying/AnyDoor-main/hoi_samples/motorcycle.jpg',required=False)
    parser.add_argument('--bank_path',type=str,default= '/network_space/server128/shared/zhuoying/AnyDoor-main/datasets/mask_storage.json',required=False)

    args = parser.parse_args()
    test_hico(args)
    # #eval_model(args)
    # h_mask, o_mask = generate_caption(args)
    # union_mask = np.maximum(np.array(h_mask)[:,:,0],np.array(o_mask)[:,:,0])
    # union_mask_3 = np.stack([union_mask,union_mask,union_mask],-1)
    # Image.fromarray(union_mask_3).save('./retrieved.jpg')