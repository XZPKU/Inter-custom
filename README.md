# Interact-custom:Customized Human Object Interaction Image Generation
[![arXiv](https://img.shields.io/badge/arXiv-2508.19575-b31b1b.svg)](https://arxiv.org/abs/2508.19575)
 <a href="https://sites.google.com/view/interact-custom"> <img src="https://img.shields.io/badge/&#8459-homepage-3875B7.svg?labelColor=21438A&style=plastic" height="25px">



> By Zhu Xu, Zhaowen Wang, Yuxin Peng, Yang Liu*
> 
> Accepted by ACM-MM 2025 

<div align="left">
<img src=./asset/method.png width=70% />
</div>


## Quick start
our pre-trained model for iamg and mgig are stored in [LINK](https://huggingface.co/sdcdavcsvc/Interact-custom), download and place them at ./ckpts.
### 1. Mask Generation
For mask generation, first follow ./iamg/environment.yml to build the virtual environment and activate it.
```
cd ./iamg
python main_demo.py --hoi_category 'a person is riding a bicycle' --demo_sample ./demo_data/1.jpg --position [0.3,0.8,0.3,0.8]
#### demo_sample and position are used to specified the background image and union location of human-object.
```

### 2. HOI Image Generation
For hoi image generation, alter the virtual environment to following environment, which is constructed via
```
pip install -r ./mgig/requirements.txt
pip install git+https://github.com/cocodataset/panopticapi.git
pip install pycocotools -i https://pypi.douban.com/simple
pip install lvis
```
Then generate with demo sample
```
cd ./mgig
python run_inference_demo.py
```

## Data prepration
Our data is stored in [LINK](https://huggingface.co/sdcdavcsvc/Interact-custom), download and place the data like:
```
-data
   |--train
        |--image
        |--video
        |--video_2
   |--test
   |--annos
   |--hico_det_clip_instance
```
## Evaluation
Firstly, use our model to generate hoi samples on our testset.
```
### for mask generation ####
cd ./iamg
python main_eval.py --OUTPUT_ROOT ./OUTPUT/eval
```
The generated masks will be saved in ./OUTPUT/eval folder. Then use the masks to generate HOI images
```
cd ../mgig
python run_inference_hoi_w_one_stage_mask_eval.py
```
Generated HOI images will be saved in ./iamg/OUTPUT/eval folder.
Then we separately evaluate the quality of generated image in terms of interaction semantic control and subject customization.
### 1.Interaction Semantic Control
For spatial-sensitive semantic evaluation, you should additionally follow [FGAHOI](https://github.com/xiaomabufei/FGAHOI) to construct the environment for evaluation. 
```
cd ./FGAHOI
python main.py
```

For Holistic Semantic, need to install LLaVA follow [LLaVA](https://github.com/haotian-liu/LLaVA)
```
cd ./eval/LLaVA/llava/eval
python hico_test_hoid_w_llava.py
```
| Model  | Holistic Semantic|Interaction Semantic|Interaction Semantic|Interaction Semantic|
| --- | ----------- |----- |----- |----- |
|    |   | Full | Rare | Non-rare|
|AnyDoor   | 82.04|17.54 |10.63 |19.18|
| Ours  | 86.02 |22.07|11.87|23.87|
### 2.Subject Customization
```
cd ./eval
python eval_dino.py
python eval_clip.py
```
| Model  | CLIP-I|DINO-human|DINO-object|DINO-pair|
| --- | ----------- |----- |----- |----- |
|AnyDoor   | 82.31 |70.08 |72.27 |74.14|
| Ours  |  87.60 |78.90 |81.39 |83.27|

## Training
### 1.iamg training
```
cd ./iamg
python main.py --OUTPUT_ROOT ./OUTPUT/train
```
### 2. mgig training

```
cd ../mgig
python run_train_anydoor.py
```

