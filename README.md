# CP-SSM
Official implementation for MICCAI 2025 paper ["Core-Periphery Principle Guided State Space Model for Functional Connectome Classification"](https://papers.miccai.org/miccai-2025/paper/1080_paper.pdf)
## Usage

Abide dataset available [here](https://drive.google.com/file/d/1rTmBuLbMNu-vW7g43eSu21ur1Sc4oVHh/view?usp=sharing).

1. Update *path* in the file *source/conf/dataset/ABIDE.yaml* to the path of your dataset.
2. Creat your conda environment

```bash
conda env create -f cpssm.yml
```
4. Run the following command to train the model.

```bash
python -m source --multirun datasz=100p model=cpssm dataset=ABIDE repeat_time=10 preprocess=mixup
```
- **datasz**, default=(10p, 20p, 30p, 40p, 50p, 60p, 70p, 80p, 90p, 100p). Percentage of the total number of samples in the dataset to use for training.

- **model**, default=(comtf,fbnetgen,brainnetcnn). Model to be used.

- **dataset**, default=(ABIDE). Dataset to be used.

- **repeat_time**, default=5. Number of times to repeat the experiment.

- **preprocess**, default=(mixup, non_mixup). Data pre-processing.
## Citation
If you find this work useful in your research, please cite the appropriate papers:
```
@inproceedings{chen2025core,
  title={Core-periphery principle guided state space model for functional connectome classification},
  author={Chen, Minheng and Yu, Xiaowei and Zhang, Jing and Chen, Tong and Cao, Chao and Zhuang, Yan and Lyu, Yanjun and Zhang, Lu and Liu, Tianming and Zhu, Dajiang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={236--246},
  year={2025},
  organization={Springer}
}
```
