# CP-SSM
Official implementation of paper "Core-Periphery Principle Guided State Space Model for Functional Connectome Classification"
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
