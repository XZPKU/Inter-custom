# Inter-custom


## Quick start
our pre-trained model for iamg and mgig is stored in [LINK]() and [LINK](), download and place them at xxx.
### Mask Generation
```
## for virtual environment construction, follow ~/iamg/environment.yml to build and activate it.
cd ~/iamg
python main_demo.py --hoi_category 'a person is riding a bicycle' --demo_sample ./demo_data/1.jpg --position [0,3,0.8,0.3,0.8]
```

### HOI Image Generation
```
## for final hoi image generation, alter the virtual environment to following environment, which is constructed via

pip install -r requirements.txt
pip install git+https://github.com/cocodataset/panopticapi.git
pip install pycocotools -i https://pypi.douban.com/simple
pip install lvis

### then generate with demo sample
cd ~/mgig
python run_inference_demo.py
```

## Data prepration
Our data is stored in [LINK](https://www.alipan.com/t/KFYVLE2H3mJRLEnHcOb0), download and place the data like:
```
-data
   |--train
        |--image
        |--video
        |--video_2
   |--test
        |--
   |--annos
        |--
```
## Evaluation
Firstly, use our model to generate hoi samples on our testset.
```
### for mask generation ####
cd ~/iamg
python main_eval.py
```
Then use the mask to generate HOI image
```
### hoi image generation ###
cd ~/mgig
python run_inference_hoi_w_one_stage_mask_eval.py
```
Then we separately evaluate the quality of generated image in terms of interaction semantic control and subject customization.
### Interaction Semantic Control
For spatial-sensitive semantic evaluation, you should additionally follow [FGAHOI](https://github.com/xiaomabufei/FGAHOI) to construct the environment for evaluation. 
```
cd ~/FGAHOI
python main.py
```

For Holistic Semantic, need to install LLaVA follow [LLaVA](https://github.com/haotian-liu/LLaVA)
```
cd ~/eval/LLaVA/llava/eval
python llava_judge.py
```

### Subject Customization
```
cd ~/eval
python evaluate_hico_test.py
python evaluate_hico_test_clip.py
```

## Training
