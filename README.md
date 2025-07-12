# Inter-custom


## Quick start

### Mask Generation
```
## for virtual environment construction, follow ~/iamg/environment.yml to build and activate it.
python test_demo.py
```

### HOI Image Generation
```
## for virtual environment construction
cd ~/mgig
pip install -r requirements.txt
pip install git+https://github.com/cocodataset/panopticapi.git
pip install pycocotools -i https://pypi.douban.com/simple
pip install lvis

### then generate with demo sample
python inference.py
```

## Data prepration
Our data is stored in [LINK](), download and place the data like:
```
-data
   |--

```
## Evaluation
our pre-trained model for iamg and mgig is stored in [LINK]() and [LINK](), download and place them at xxx.
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
