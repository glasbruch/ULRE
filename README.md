# Uncertainty-Aware Likelihood Ratio Estimation for Pixel-Wise Out-of-Distribution Detection

This repository contains the official implementation of our paper "Uncertainty-Aware Likelihood Ratio Estimation for Pixel-Wise Out-of-Distribution Detection".

![Teaser](teaser.PNG)

## Installation
### Install Dependencies
```sh
conda env create -f environment.yml
```
### Prepare Datasets
Follow instructions at https://github.com/yyliu01/RPL/blob/main/docs/installation.md

## Training
Run training by executing:
```sh
sh code/train.sh
```

## Citation
```
@inproceedings{holle2025uncertainty,
  title={Uncertainty-Aware Likelihood Ratio Estimation for Pixel-Wise Out-of-Distribution Detection},
  author={H{\"o}lle, Marc and Kellermann, Walter and Belagiannis, Vasileios},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={772--782},
  year={2025}
}
```

## Acknowledgement
We used and modified code parts from [PEBAL](https://github.com/tianyu0207/PEBAL) and [RPL](https://github.com/yyliu01/RPL). We like to thank the authors for making their code publicly available.
