# A Comparative Study for Effective OOD Detection

This is the source code for our paper:
[A Comparative Study for Effective OOD Detection]

Code is modified from [GradNorm](https://github.com/deeplearning-wisc/gradnorm_ood)

We investigate the effectiveness of the max-logit-based approach for OOD detection. The primary objective is to demonstrate that by utilising the model logits alone, we can achieve comparable performance to more complex OOD detection techniques. 

## Usage

### 1. Dataset Preparation

#### In-distribution dataset

Please download CIFAR10, CIFAR100 and [ImageNet-1k](https://www.image-net.org/challenges/LSVRC/2012/index) and place the validation data in `./dataset/id_data/`.  

#### Out-of-distribution dataset

We use the following OOD datasets for evaluation:
[Places](http://places2.csail.mit.edu/download.html), 
[LSUN_Crop](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz), 
[LSUN_resize](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz), 
[iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz),
[Texture](https://www.robots.ox.ac.uk/~vgg/data/dtd/), 
SVHN,
[SUN](https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/), and
[iNaturalist](https://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/)

Please put all downloaded OOD datasets into `./dataset/ood_data/`.



### 2. Pre-trained Model Preparation

For the ease of reproduction, we provide our pre-trained models below:
[cifar10_resnet50_pretrained_epoch_199.pt](https://drive.google.com/uc?export=download&id=1j8A22fdxFFgLTFZHDEH7eaL9PlLAvuff), and 
[cifar100_resnet50_pretrained.pt](https://drive.google.com/uc?export=download&id=1AVBgzLLuUE-tUcOJuKfaJB9_3hAoJ3Gi)

Following [Energy-based Out-of-distribution Detection](https://arxiv.org/pdf/2010.03759.pdf), we utilise their pre-trained WideResNet models

[cifar10_wrn_pretrained_epoch_99.pt](https://github.com/wetliu/energy_ood/tree/master/CIFAR/snapshots/pretrained), and
[cifar100_wrn_pretrained_epoch_99.pt](https://github.com/wetliu/energy_ood/tree/master/CIFAR/snapshots/pretrained)

Please put the downloaded models in `./checkpoints/pretrained_models`.



### 3. OOD Detection Evaluation

To reproduce our results, please run:
```
./scripts/test.sh MSP(/ODIN/Energy/Max-logit) SVHN(/iSUN/Places/dtd/images/LSUN/LSUN_resize/SUN/iNaturalist)        
```

## OOD Detection Results

The max-logit technique provides a practical and efficient solution for distinguishing between in-distribution and OOD data. It has demonstrated superior performance compared to the MSP baseline and comparable results to the state-of-the-art ODIN and energy techniques.

Performance among different baselines

CIFAR10:
|   Method         |     FPR95 (WideResNet)      | FPR95 (ResNet50)  | 
| ------------------ |---------------- | ---------------- |
| MSP |     50.11%      | 47.29% |
| ODIN |     31.94%      |26.09%|
| Energy score |     31.75%       | 28.52%|
| Max-logit |     32.2%       | 29.63%|

CIFAR100:
|   Method         |     FPR95 (WideResNet)    |  FPR95 (ResNet50)  | 
| ------------------ |---------------- | ---------------- |
| MSP |     79.83%      | 86.53%|
| ODIN |     70.25%      | 83.13%|
| Energy score |     72.43%       |  85.02%|
| Max-logit |     72.39%       |85.34%|

ImageNet:
|   Method         |     FPR95 (MobileNet)    |  FPR95 (ResNet50)  | 
| ------------------ |---------------- | ---------------- |
| MSP |     70.83%      | 65.48%|
| ODIN |    56.71%      | 52.95%|
| Energy score |      59.28%       |  58.04%|
| Max-logit |     60.93%       | 58.35%|


## Citation
