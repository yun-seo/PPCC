# PPCC: Syn-to-Real Domain Adaptation for Point Cloud Completion via Part-based Approach (ECCV 2024)
Official repository for ECCV 2024 paper:
**"Syn-to-Real Domain Adaptation for Point Cloud Completion via Part-based Approach"**


## Setup
We recommend using `conda` to create a virtual environment.
```bash
conda create -n ppcc python=3.8
conda activate ppcc
pip install -r requirements.txt
```
## Install
To enable CUDA-based acceleration, install the required extensions:
```bash
cd extensions/chamfer_dist/ && python setup.py install
cd extensions/emd/ && python setup.py install
cd utils/pointnet2_ops_lib/ && python setup.py install
```

## Datasets
The preprocessed datasets are available [here](https://drive.google.com/drive/folders/1Ckn-8O3GU1HTIMv57zigWo6F9uUbqSkq?usp=sharing).

- **PartNet**: A synthetic point cloud dataset with per-point part labels.
- **USSPA**: A real-world point cloud dataset with per-point part labels and confidence values.

The original USSPA dataset can be found at [USSPA](https://github.com/murcherful/USSPA).  
We reprocessed the data by adding part labels and confidence values for use in our training and evaluation pipelines.

Please organize the dataset directory as follows:

```text
data/
├── chair/
│   └── partnet
│       └── train
│       └── val
│       └── test
│   └── usspa
│       └── ...
├── table/
│   └── ...
└── ...
```
Make sure to update dataset paths in `configs/PPCC.yaml` if needed.

## Train
To train the model, run the following command:

```bash
CUDA_VISIBLE_DEVICES={GPU_ID} python main.py --category {category} --exp_name {name}
```
- `--category`: Object category to train on. Options are:
chair, table, lamp, bed
- `--exp_name`: A custom name for the experiment (used for logging and checkpoints).

Example:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --category chair --exp_name chair_exp1
```

## Evalutation
To evaluate the model, use the following command:

```bash
CUDA_VISIBLE_DEVICES={GPU_ID} python main.py --evaluation --category {category} --ckpts {checkpoint_path}
```
- `--evaluation`: Flag to indicate evaluation mode.
- `--category`: Object category to evaluate on. Options are:
chair, table, lamp, bed
- `--ckpts`: Path to the saved model checkpoint for evaluation.

## Citation
If you find our work useful for your research, please consider citing:
```bibtex
@inproceedings{yang2024syn,
  title={Syn-to-Real Domain Adaptation for Point Cloud Completion via Part-Based Approach},
  author={Yang, Yunseo and Kim, Jihun and Yoon, Kuk-Jin},
  booktitle={European Conference on Computer Vision},
  pages={179--197},
  year={2024},
  organization={Springer}
}
```
