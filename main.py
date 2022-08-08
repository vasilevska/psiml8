import argparse
import bisect
from email.mime import audio
import logging
import pickle
import pytorch_lightning as pl
import dataset.prepare_scripts.config as config
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
import torch
import os
from dataset.prepare_scripts.download import create_folder
from utils import d_prime, dump_config
from dataset.dataset import Audioset
import torch.distributed as dist
import torch.optim as optim
import dataset.prepare_scripts.config as config
from model.at import CSWAT
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
#import torch.nn as nn
#import torch.nn.functional as F

class DataLoaderModule(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
    

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset = self.train_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size,
            shuffle = False,
            sampler = None
        )
        return train_loader
    
    def val_dataloader(self):
        eval_loader = DataLoader(
            dataset = self.eval_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size,
            shuffle = False,
            sampler = None
        )
        return eval_loader
    
    def test_dataloader(self):
        test_loader = DataLoader(
            dataset = self.eval_dataset,
            num_workers = config.num_workers,
            batch_size = config.batch_size,
            shuffle = False,
            sampler = None
        )
        return test_loader

class SEDWrapper(pl.LightningModule):
    def __init__(self, config, sed_model, dataset):
        super().__init__()
        self.sed_model = sed_model
        self.config = config
        self.dataset = dataset
    
    def evaluate_metric(self, pred, ans):
        ap = []
        if self.config.dataset_type == "audioset":
            mAP = np.mean(average_precision_score(ans, pred, average = None))
            mAUC = np.mean(roc_auc_score(ans, pred, average = None))
            dprime = d_prime(mAUC)
            return {"mAP": mAP, "mAUC": mAUC, "dprime": dprime}
        else:
            acc = accuracy_score(ans, np.argmax(pred, 1))
            return {"acc": acc}  

    def forward(self, x):
        output_dict = self.sed_model(x)
        return output_dict["clipwise_output"], output_dict["framewise_output"]

    def inference(self, x):
        self.eval()
        x = torch.from_numpy(x).float().to(self.device_type)
        output_dict = self.sed_model(x, None, True)
        for key in output_dict.keys():
            output_dict[key] = output_dict[key].detach().cpu().numpy()
        return output_dict

    def training_step(self, batch, batch_idx):
        pred, _ = self(batch["waveform"])
        loss = self.loss_func(pred, batch["target"])
        self.log("loss", loss, on_epoch= True, prog_bar=True)
        return loss
    
    def training_epoch_end(self, outputs):
        self.dataset.generate_queue()
    
    def validation_step(self, batch, batch_idx):
        pred, _ = self(batch["waveform"])
        return [pred.detach(), batch["target"].detach()]

    def validation_epoch_end(self, validation_step_outputs):
        self.device_type = next(self.parameters()).device
        pred = torch.cat([d[0] for d in validation_step_outputs], dim = 0)
        target = torch.cat([d[1] for d in validation_step_outputs], dim = 0)

        metric_dict = {
            "mAP": 0.,
            "mAUC": 0.,
            "dprime": 0.
        }
        
        gather_pred = pred.cpu().numpy()
        gather_target = target.cpu().numpy()
        if self.config.dataset_type == "scv2":
            gather_target = np.argmax(gather_target, 1)
        metric_dict = self.evaluate_metric(gather_pred, gather_target)
        print(self.device_type, metric_dict, flush = True)
    
        if self.config.dataset_type == "audioset":
            self.log("mAP", metric_dict["mAP"], on_epoch = True, prog_bar=True, sync_dist=False)
            self.log("mAUC", metric_dict["mAUC"], on_epoch = True, prog_bar=True, sync_dist=False)
            self.log("dprime", metric_dict["dprime"], on_epoch = True, prog_bar=True, sync_dist=False)
        else:
            self.log("acc", metric_dict["acc"], on_epoch = True, prog_bar=True, sync_dist=False)

    def test_step(self, batch, batch_idx):
        preds = []
        pred, pred_map = self(batch["waveform"])
        preds.append(pred.unsqueeze(0))
        preds = torch.cat(preds, dim=0)
        pred = preds.mean(dim = 0)
        if self.config.fl_local:
            return [
                pred.detach().cpu().numpy(), 
                pred_map.detach().cpu().numpy(),
                batch["audio_name"],
                batch["real_len"].cpu().numpy()
            ]
        else:
            return [pred.detach(), batch["target"].detach()]      

    def test_epoch_end(self, test_step_outputs):
        if self.config.fl_local:
            pred = np.concatenate([d[0] for d in test_step_outputs], axis = 0)
            pred_map = np.concatenate([d[1] for d in test_step_outputs], axis = 0)
            audio_name = np.concatenate([d[2] for d in test_step_outputs], axis = 0)
            real_len = np.concatenate([d[3] for d in test_step_outputs], axis = 0)
            heatmap_file = os.path.join(self.config.heatmap_dir, self.config.test_file + "_" + str(self.device_type) + ".npy")
            save_npy = [
                {
                    "audio_name": audio_name[i],
                    "heatmap": pred_map[i],
                    "pred": pred[i],
                    "real_len":real_len[i]
                }
                for i in range(len(pred))
            ]
            np.save(heatmap_file, save_npy)
        else:
            self.device_type = next(self.parameters()).device
            pred = torch.cat([d[0] for d in test_step_outputs], dim = 0)
            target = torch.cat([d[1] for d in test_step_outputs], dim = 0)
            gather_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
            gather_target = [torch.zeros_like(target) for _ in range(dist.get_world_size())]
            dist.barrier()
            metric_dict = {
            "mAP": 0.,
            "mAUC": 0.,
            "dprime": 0.
            }
            dist.all_gather(gather_pred, pred)
            dist.all_gather(gather_target, target)
            if dist.get_rank() == 0:
                gather_pred = torch.cat(gather_pred, dim = 0).cpu().numpy()
                gather_target = torch.cat(gather_target, dim = 0).cpu().numpy()
                if self.config.dataset_type == "scv2":
                    gather_target = np.argmax(gather_target, 1)
                metric_dict = self.evaluate_metric(gather_pred, gather_target)
                print(self.device_type, dist.get_world_size(), metric_dict, flush = True)
            
            self.log("mAP", metric_dict["mAP"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            self.log("mAUC", metric_dict["mAUC"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            self.log("dprime", metric_dict["dprime"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            dist.barrier()  

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr = self.config.learning_rate, 
            betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0.05, 
        )
        # Change: SWA, deprecated
        # optimizer = SWA(optimizer, swa_start=10, swa_freq=5)
        def lr_foo(epoch):       
            if epoch < 3:
                # warm up lr
                lr_scale = self.config.lr_rate[epoch]
            else:
                # warmup schedule
                lr_pos = int(-1 - bisect.bisect_left(self.config.lr_scheduler_epoch, epoch))
                if lr_pos < -3:
                    lr_scale = max(self.config.lr_rate[0] * (0.98 ** epoch), 0.03 )
                else:
                    lr_scale = self.config.lr_rate[lr_pos]
            return lr_scale
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )
        
        return [optimizer], [scheduler]    



def train():
    train_index_path = os.path.join(config.dataset_path, "hdf5s","indexes", config.index_type + ".h5")
    eval_index_path = os.path.join(config.dataset_path,"hdf5s", "indexes", "eval.h5")
    train_idc = np.load(config.index_type + "_idc.npy", allow_pickle = True)
    eval_idc = np.load("eval_idc.npy", allow_pickle = True)
    exp_dir = os.path.join(config.workspace, "results", config.exp_name)
    checkpoint_dir = os.path.join(config.workspace, "results", config.exp_name, "checkpoint")
    if not config.debug:
        create_folder(os.path.join(config.workspace, "results"))
        create_folder(exp_dir)
        create_folder(checkpoint_dir)
        #dump_config(config, os.path.join(exp_dir, config.exp_name), False)
    dataset = Audioset(
        index_path=train_index_path,
        idc = train_idc,
        config = config
    )
    eval_dataset = Audioset(
        index_path=eval_index_path,
        idc = eval_idc,
        config = config,
        eval_mode = True
    )
    audioset_data = DataLoaderModule(dataset, eval_dataset)
    checkpoint_callback = ModelCheckpoint(
        monitor = "mAP",
        filename='l-{epoch:d}-{mAP:.3f}-{mAUC:.3f}',
        save_top_k = 20,
        mode = "max"
    )
    trainer = pl.Trainer(
        deterministic=True,
        default_root_dir = checkpoint_dir,
        gpus = 1, 
        accelerator = "gpu",
        val_check_interval = 0.1,
        max_epochs = config.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        callbacks = [checkpoint_callback],
        num_sanity_val_steps = 0,
        resume_from_checkpoint = None, 
        replace_sampler_ddp = False,
        gradient_clip_val=1.0
    )
    sed_model = CSWAT(
        img_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans = 1,
        num_classes=config.classes_num,
        #window_size=config.htsat_window_size,
        #config = config,
        depth = config.htsat_depth,
        embed_dim = config.htsat_dim,
        #patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head
    )
    
    model = SEDWrapper(
        sed_model = sed_model, 
        config = config,
        dataset = dataset
    )
    if config.resume_checkpoint is not None:
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
        ckpt["state_dict"].pop("sed_model.head.weight")
        ckpt["state_dict"].pop("sed_model.head.bias")
        # finetune on the esc and spv2 dataset
        ckpt["state_dict"].pop("sed_model.tscam_conv.weight")
        ckpt["state_dict"].pop("sed_model.tscam_conv.bias")
        model.load_state_dict(ckpt["state_dict"], strict=False)
    elif config.cswin_pretrain_path is not None: # train with pretrained model
        ckpt = torch.load(config.cswin_pretrain_path, map_location="cpu")
        # load pretrain model
        ckpt = ckpt["model"]
        found_parameters = []
        unfound_parameters = []
        model_params = dict(model.state_dict())

        for key in model_params:
            m_key = key.replace("sed_model.", "")
            if m_key in ckpt:
                if m_key == "patch_embed.proj.weight":
                    ckpt[m_key] = torch.mean(ckpt[m_key], dim = 1, keepdim = True)
                if m_key == "head.weight" or m_key == "head.bias":
                    ckpt.pop(m_key)
                    unfound_parameters.append(key)
                    continue
                assert model_params[key].shape==ckpt[m_key].shape, "%s is not match, %s vs. %s" %(key, str(model_params[key].shape), str(ckpt[m_key].shape))
                found_parameters.append(key)
                ckpt[key] = ckpt.pop(m_key)
            else:
                unfound_parameters.append(key)
        print("pretrain param num: %d \t wrapper param num: %d"%(len(found_parameters), len(ckpt.keys())))
        print("unfound parameters: ", unfound_parameters)
        model.load_state_dict(ckpt, strict = False)
        model_params = dict(model.named_parameters())
    
    trainer.fit(model, audioset_data)


def test():
    # dataset file pathes
    eval_index_path = os.path.join(config.dataset_path,"hdf5s", "indexes", "eval.h5")
    eval_idc = np.load("eval_idc.npy", allow_pickle = True)
    eval_dataset = Audioset(
        index_path=eval_index_path,
        idc = eval_idc,
        config = config,
        eval_mode = True
    )
        
    audioset_data = DataLoaderModule(eval_dataset, eval_dataset)
    trainer = pl.Trainer(
        deterministic=True,
        gpus = 1, 
        max_epochs = config.max_epoch,
        auto_lr_find = True,    
        sync_batchnorm = True,
        checkpoint_callback = False,
        accelerator = "qpu",
        num_sanity_val_steps = 0,
        # resume_from_checkpoint = config.resume_checkpoint,
        replace_sampler_ddp = False,
        gradient_clip_val=1.0
    )
    sed_model = CSWAT(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config = config,
        depths = config.htsat_depth,
        embed_dim = config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head
    )
    
    model = SEDWrapper(
        sed_model = sed_model, 
        config = config,
        dataset = eval_dataset
    )
    if config.resume_checkpoint is not None:
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
        ckpt["state_dict"].pop("sed_model.head.weight")
        ckpt["state_dict"].pop("sed_model.head.bias")
        model.load_state_dict(ckpt["state_dict"], strict=False)
    trainer.test(model, datamodule=audioset_data)   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #subparsers = parser.add_subparsers(dest = "mode")
    
    #parser_esm_test = subparsers.add_parser("esm_test")
    #parser_saveidc = subparsers.add_parser("save_idc")
    #parser_wa = subparsers.add_parser("weight_average")
    
    args = parser.parse_args()
    
    # default settings
    args.mode = "train"
    logging.basicConfig(level=logging.INFO) 
    pl.utilities.seed.seed_everything(seed = config.random_seed)


    
    
    if args.mode == "train":
        train()

    elif args.mode == "test":
        test()
       