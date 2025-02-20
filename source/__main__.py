from datetime import datetime
import wandb
import hydra
from omegaconf import DictConfig, open_dict
from .dataset import dataset_factory
from .models import model_factory
from .components import lr_scheduler_factory, optimizers_factory, logger_factory
from .training import training_factory
from datetime import datetime
import os
import torch
import numpy as np
import pandas as pd
def model_training(cfg: DictConfig):

    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")

    dataloaders = dataset_factory(cfg)
    logger = logger_factory(cfg)
    model = model_factory(cfg)
    optimizers = optimizers_factory(
        model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer,
                                         cfg=cfg)
    training = training_factory(cfg, model, optimizers,
                                lr_schedulers, dataloaders, logger)

    t_acc,t_auc,t_sen,t_spec, t_rec,t_pre,attn = training.train()
    return t_acc,t_auc,t_sen,t_spec, t_rec,t_pre,attn


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    group_name = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.datasz.percentage}_{cfg.preprocess.name}"
    # _{cfg.training.name}\
    # _{cfg.optimizer[0].lr_scheduler.mode}"
    acc_list = []
    auc_list = []
    sen_list = []
    spec_list = []
    rec_list = []
    pre_list = []
    attn_list=[]

    for _ in range(cfg.repeat_time):
        run = wandb.init(project=cfg.project, entity=cfg.wandb_entity, reinit=True,
                         group=f"{group_name}", tags=[f"{cfg.dataset.name}"])
        t_acc,t_auc,t_sen,t_spec,t_rec,t_pre , attn= model_training(cfg)
        acc_list.append(t_acc)
        auc_list.append(t_auc)
        sen_list.append(t_sen)
        spec_list.append(t_spec)
        rec_list.append(t_rec)
        pre_list.append(t_pre)
        attn_list.append(attn)
        # print("test acc mean {}".format(t_acc))
        # print("test auc mean {}".format(t_auc))
        # print("test sensitivity mean {} ".format(t_sen))
        # print("test specficity mean {} ".format(t_spec))
        # print("test recall mean {} ".format(t_rec))
        # print("test precision mean {} ".format(t_pre))
        run.finish()
    print("test acc mean {} std {}".format(np.mean(acc_list),np.std(acc_list)))
    print("test auc mean {} std {}".format(np.mean(auc_list)*100,np.std(auc_list)*100))
    print("test sensitivity mean {} std {}".format(np.mean(sen_list)*100,np.std(sen_list)*100))
    print("test specficity mean {} std {}".format(np.mean(spec_list)*100,np.std(spec_list)*100))
    print("test recall mean {} std {}".format(np.mean(rec_list)*100,np.std(rec_list)*100))
    print("test precision mean {} std {}".format(np.mean(pre_list)*100,np.std(pre_list)*100))

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # device_ids = [0]
    # device = torch.device('cuda:{}'.format(device_ids[0]))
    main()
