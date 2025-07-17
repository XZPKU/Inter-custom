import argparse
import torch
from omegaconf import OmegaConf
import numpy as np
import random
from test_demo import Trainer
from distributed import synchronize
import os 
import torch.multiprocessing as multiprocessing


if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_ROOT", type=str,  default="DATA", help="path to DATA")
    parser.add_argument("--OUTPUT_ROOT", type=str,  default="OUTPUT/temp", help="path to OUTPUT")

    parser.add_argument("--name", type=str,  default="test", help="experiment will be stored in OUTPUT_ROOT/name")
    parser.add_argument("--seed", type=int,  default=123, help="used in sampler")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--yaml_file", type=str,  default="configs/hoi_hico_shape.yaml", help="paths to base configs.")


    parser.add_argument("--base_learning_rate", type=float,  default=5e-5, help="")
    parser.add_argument("--weight_decay", type=float,  default=0.0, help="")
    parser.add_argument("--warmup_steps", type=int,  default=10000, help="")
    parser.add_argument("--scheduler_type", type=str,  default='constant', help="cosine or constant")
    parser.add_argument("--batch_size", type=int,  default=4, help="")
    parser.add_argument("--workers", type=int,  default=1, help="")
    parser.add_argument("--official_ckpt_name", type=str,  default="sd-v1-4.ckpt", help="SD ckpt name and it is expected in DATA_ROOT, thus DATA_ROOT/official_ckpt_name must exists")
    parser.add_argument("--ckpt", type=lambda x:x if type(x) == str and x.lower() != "none" else None,  default='./ckpts/iamg.pth', 
        help=("If given, then it will start training from this ckpt"
              "It has higher prioty than official_ckpt_name, but lower than the ckpt found in autoresuming (see trainer.py) ")
    )
    parser.add_argument('--enable_ema', default=False, type=lambda x:x.lower() == "true")
    parser.add_argument("--ema_rate", type=float,  default=0.9999, help="")
    parser.add_argument("--total_iters", type=int,  default=500000, help="")
    parser.add_argument("--save_every_iters", type=int,  default=5000, help="")
    parser.add_argument("--disable_inference_in_training", type=lambda x:x.lower() == "false",  default=True, help="Do not do inference, thus it is faster to run first a few iters. It may be useful for debugging ")
    parser.add_argument("--gradient_accumulation_step", type=int, default=1, help="")
    parser.add_argument('--fix_interaction_embedding', default=False, type=lambda x: x.lower() == "true",
                        help="Do not train interaction embedding")
    parser.add_argument('--amp', default=False, type=lambda x:x.lower() == "true")
    parser.add_argument('--demo_sample',default = './demo_data/1.jpg')
    parser.add_argument('--hoi_category',default = 'a person is riging a bycicle')
    parser.add_argument('--position',default = [0.3,0.8,0.3,0.8])
    args = parser.parse_args()
    assert args.scheduler_type in ['cosine', 'constant']

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1
    args.inpaint_mode = False
    args.randomize_fg_mask = False
    args.random_add_bg_mask = False

    assert (args.gradient_accumulation_step == 1) != args.distributed, "gradient accumulation only for ddp"

    if args.distributed:
        if "LOCAL_RANK" in os.environ:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        else:
            torch.cuda.set_device(args.local_rank)
            print("Depreciated --local-rank")
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()



    config = OmegaConf.load(args.yaml_file) 
    config.update( vars(args) )
    config.total_batch_size = config.batch_size * n_gpu
    if "LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["LOCAL_RANK"])
    if args.inpaint_mode:
        config.model.params.inpaint_mode = True


    trainer = Trainer(config)
    synchronize()
    trainer.evaluation()











