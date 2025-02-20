from omegaconf import DictConfig, open_dict
from .abcd import load_abcd_data
from .abide import load_abide_data
from .adni import load_adni_data
from .dataloader import init_dataloader, init_stratified_dataloader
from typing import List
import torch.utils as utils


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    assert cfg.dataset.name in ['abcd', 'abide','adni']
    # print(cfg.dataset)
    datasets = eval(
        f"load_{cfg.dataset.name}_data")(cfg)
    # print(cfg.dataset.stratified)
    dataloaders = init_stratified_dataloader(cfg, *datasets) \
        if cfg.dataset.stratified \
        else init_dataloader(cfg, *datasets)

    return dataloaders
