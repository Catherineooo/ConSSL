# ConSSL
This is the code in Pytorch for Class and Instance Aware Semi-supervised Learning with Contrastive Learning under Class Distribution Mismatch.

<!-- # FixMatch
The code is changed from https://github.com/kekmodel/FixMatch-pytorch -->
<!-- # FixMatchCCSSL
The code is from https://github.com/TencentYoutuResearch/Classification-SemiCLS -->
**Supported algorithms**
- FixMatch (NeurIPS 2020)[1]
- CoMatch (ICCV 2021)[2]
- FixMatch+CCSSL(CVPR 2022)[3]
- ConSSL

**Supported dataset**   
In Distribution dataset
- CIFAR10
- CIFAR100
- STL-10   

Out of Distribution dataset
- semi-iNat-2021


## Results
### Top-1 Accuracy for in-distribution datasets.
(in-distribution tab)

|Method |       |CIFAR100  |       |       |CIFAR10|       |STL10  |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| **labels**    | 400      | 2500|10000| 40      | 250 | 4000|        |
| FixMatch      | 24.55    | 53.90|65.93| 91.18   | 93.15| 93.99| 87.40  |
| CoMatch       | 47.27    | 65.04|73.78| **93.01**| 93.54| 95.44| 88.59  |
| FixmatchCCSSL | **54.06**| 75.1 |76.53| 92.16   | 93.92| **95.77**| 86.94|
| ConSSL(Ours)  | 50.80    | **75.20**| **78.02**| 92.41   | **94.00**| 95.49| **88.64** |


### Out-of-distribution datasets
(out-of-distribution tab)

| Method        | Semi-iNat 2021 Top-1 | Semi-iNat 2021 Top-5 |
|:-------------:|:---------------------:|:---------------------:|
|   FixMatch    |         18.40%        |         32.15%        |
|   CoMatch     |         20.42%        |         38.94%        |
| FixMatchCCSSL |         23.48%        |         40.15%        |
| ConSSL(ours)  |       **24.86%**      |       **43.58%**      |



## Usage
<!-- ### Install
Clone this repo to your machine and install dependencies:  
We use torch==1.6.0 and torchvision==0.12.0 for CUDA 10.1  
You may have to adapt for your own CUDA and install corresponding mmcv-full version. (Make sure your mmcv-full version is later than 1.3.2)
>
or you can just:
```
pip install -r requirements.txt
``` -->
### Run
1. **prepare environment**  
  We use torch 1.8.1+cu111, the dependencies are as requirements.txt. You may have to adapt for your own CUDA and install corresponding mmcv-full version. (Make sure your mmcv-full version is later than 1.3.2). You can just run：
```
pip install -r requirements.txt
```

2. **prepare datasets**  
   Organize your datasets as the following form:
```
data
└── CIFAR
│   └── cifar-10-batches-py # cifar10
│   └── cifar-100-python # cifar100
├── stl10
│   └── stl10_binary
└── semi-inat2021
│   ├── annotation_v2.json
│   ├── l_train
│   │   ├──anno.txt
│   │   └──l_train
│   │   │   ├──0
│   │   │   ├──1
│   │   │   │  └──0.jpg
│   │   │      ....
│   ├── u_train
│   │   ├──anno.txt
│   │   └──u_train
│   ├── val
│   │   ├──anno.txt
│   │   └──val
  ```

*Note: anno.txt contains data path and label(if have) for each image*, e.g.:

```python
# prepare for semi-inat 2021, will print three txt path needed in config,
# like in configs/ccssl/fixmatchccssl_exp512_cifar100_wres_x8_b4x16_l2500_soft.py
python tools/data/prepare_semi_inat.py ./data/semi-inat2021

# anno.txt under l_train
your/dataste/semi-inat-2021/l_train/l_train/1/0.jpg 1

# anno.txt under u_train
your/dataste/semi-inat-2021/l_train/u_train/xxxxx.jpg
```


3. Run the experiments with different SSL as:
 ```python
 ## Single-GPU
 # to train the model by 40 labeled data of CIFAR-10 dataset by FixMatch:
 python train_semi.py --cfg ./configs/fixmatch/fm_cifar10_wres_b1x64_l250.py --out your/output/path   --seed 5 --gpu-id 0

 # to train the model by 10000 labeled data of CIFAR-100 dataset by ConSSL:
 python train_semi.py --cfg ./configs/conssl/conssl_exp512_cifar100_wres_x8_b4x16_l10000.py --out out/Semi-iNat2021   --seed 5 --gpu-id 0

## Multi-GPU
# to train the model by CIFAR100 dataset by FixMatch+CCSSL with 4GPUs:
 python -m torch.distributed.launch --nproc_per_node 4 train_semi.py --cfg ./configs/ccssl/fixmatchccssl_exp512_cifar100_wres_x8_b4x16_l2500_soft.py --out /your/output/path --use_BN True  --seed 5

# to train the model by Semi-iNat2021 dataset by ConSSL with 4GPUs:
 python -m torch.distributed.launch --nproc_per_node 4 train_semi.py --cfg ./configs/conssl/conssl_exp512_seminat_b4x16_soft06_push09_mu7_lc2.py --out /your/output/path --use_BN True  --seed 5
 ```
## Acknowledgement
This code  is based on [FixMatchCCSSL](https://github.com/TencentYoutuResearch/Classification-SemiCLS).
## Reference
[1] Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A Raf-fel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li.  Fixmatch:  Simplifying semi-supervised learning with consistency and confidence.NeurIPS, 33, 2020.  
[2] Li, Junnan, Caiming Xiong, and Steven CH Hoi. "Comatch: Semi-supervised learning with contrastive graph regularization." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.  
[3] Yang, Fan, et al. "Class-Aware Contrastive Semi-Supervised Learning." arXiv preprint arXiv:2203.02261 (2022).

