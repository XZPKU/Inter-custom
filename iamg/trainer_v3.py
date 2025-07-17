import torch
from torch.cuda.amp import GradScaler

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

# = = = = = = = = = = = = = = = = = = useful functions = = = = = = = = = = = = = = = = = #


class ImageCaptionSaver:
    def __init__(self, base_path, nrow=8, normalize=True, scale_each=True, range=(-1, 1)):
        self.base_path = base_path
        self.nrow = nrow
        self.normalize = normalize
        self.scale_each = scale_each
        self.range = range

    def __call__(self, target, condition, generation,  seen,ids=None):

        save_path = os.path.join(self.base_path, str(seen).zfill(8) + '_target.png')
        target = target.to(torch.float32)
        torchvision.utils.save_image(target, save_path, nrow=self.nrow)

        save_path = os.path.join(self.base_path, str(seen).zfill(8) + '_condition.png')
        condition = condition.to(torch.float32)
        torchvision.utils.save_image(condition, save_path, nrow=self.nrow)
        
        save_path = os.path.join(self.base_path, str(seen).zfill(8) + '_generation.png')
        torchvision.utils.save_image(generation, save_path, nrow=self.nrow)
 


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
        state_dict = read_official_ckpt('/home/xuzhu/interactdiffusion/768-v-ema.ckpt')
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
            elif state_dict["model"][weight_name].ndim==2:
                temp = state_dict["model"][weight_name].unsqueeze(-1)
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
            if self.starting_iter >= config.total_iters:
                synchronize()
                print("Training finished. Start exiting")
                exit()
        
        if get_rank() == 0:
            self.image_caption_saver = ImageCaptionSaver(self.name)

        if config.distributed:
            self.model = DDP(self.model, device_ids=[config.local_rank], output_device=config.local_rank,
                             broadcast_buffers=False)

    @torch.no_grad()
    def get_input(self, batch):

        # target = self.autoencoder.encode(batch["target"].type(torch.float16))
        target = self.autoencoder.encode(batch["target"].float())

        context = self.text_encoder.encode(batch["caption"])
        # condition = self.autoencoder.encode(batch["condition"].type(torch.float16))
        condition = self.autoencoder.encode(batch["condition"].float())
        _t = torch.rand(target.shape[0]).to(target.device)
        t = (torch.pow(_t, 1) * 1000).long()
        t = torch.where(t != 1000, t, 999)  # if 1000, then replace it with 999

        return target, t, context, condition
    def decode_loss(self,decoded_image,target):
        return torch.nn.functional.mse_loss(decoded_image.float(),target.float())
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
        for iter_idx in iterator:  # note: iter_idx is not from 0 if resume training
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

    @torch.no_grad()
    def save_ckpt_and_result(self):
        model_wo_wrapper = self.model.module if self.config.distributed else self.model

        iter_name = self.iter_idx + 1  # we add 1 as the actual name
        self.config.disable_inference_in_training = False
        if not self.config.disable_inference_in_training:
            # Do an inference on one training batch
            batch_here = self.config.batch_size
            batch = sub_batch(next(self.loader_train), batch_here)
            batch_to_device(batch, self.device)
            
            ##### get real input image and taregt image##########
            real_target = []
            real_condition = []
            for i in range(batch_here):
                temp_data = {"target": batch["target"][i],"condition": batch["condition"][i]}
                # target, condition  = self.dataset_train.vis_get_item(temp_data['target'],temp_data['condition'])
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
        