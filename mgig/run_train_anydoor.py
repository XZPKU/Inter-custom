import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.hico_dataset import HICODataset_concat_h_o

import os
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.utils.data import ConcatDataset
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf

def main():
    save_memory = False
    disable_verbosity()
    if save_memory:
        enable_sliced_attention()
    resume_path = '/network_space/server128/shared/zhuoying/AnyDoor-main/ckpt/epoch=1-step=8687.ckpt'
    batch_size = 1
    logger_freq = 1000
    learning_rate = 1e-5
    sd_locked = False
    only_mid_control = False
    n_gpus = 2
    accumulate_grad_batches=1

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./configs/anydoor.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Datasets
    DConf = OmegaConf.load('./configs/hico_data.yaml')
    
    dataset3 = HICODataset_concat_h_o(**DConf.Train.HICODataset_concat_h_o)

    image_data = [dataset3]

    dataset = ConcatDataset( image_data )
    dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=n_gpus,strategy="ddp_sharded",precision=16, accelerator="gpu",callbacks=[logger,pl.callbacks.ModelCheckpoint(every_n_train_steps=40000, save_top_k=-1,save_weights_only=True)], progress_bar_refresh_rate=1, weights_save_path='/network_space/server128/shared/zhuoying/AnyDoor-main/inv_sr_hico_ckpt',accumulate_grad_batches=accumulate_grad_batches)
    trainer.fit(model, dataloader)

if __name__ == '__main__': 
    # __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # __spec__.name = 'builtins'
    main()
    