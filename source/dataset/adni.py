import numpy as np
import torch
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_adni_data(cfg: DictConfig):

    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    final_timeseires = data["timeseries"]
    # print(final_timeseires)
    if np.any(np.isnan(final_timeseires)):
        print("NaN detected in inputs.")
    final_pearson = data["corr"]
    labels = data["label"]
    site = data['sex']
    scaler = StandardScaler(mean=np.mean(
        final_timeseires), std=np.std(final_timeseires))

    final_timeseires = scaler.transform(final_timeseires)
    
    final_timeseires, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, labels)]

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]

    return final_timeseires, final_pearson, labels, site