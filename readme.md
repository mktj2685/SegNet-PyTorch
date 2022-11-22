# SegNet-Pytorch

## Description

Pytoch Implementation of SegNet and train script with VOC2012 dataset.

## Install

```
git clone https://github.com/mktj2685/SegNet-PyTorch.git
cd SegNet-Pytorch
pip install -r requirements.txt
pip install -e .
```

## Usage

1. Please download and unzip [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) at `SegNet-Pytorch/data/`.

```
SegNet-Pytorch/
    └ data/
        └ VOC2012/
            ├ Annotations/
            ├ ImageSets/
            ├ JPEGImages/
            ├ SegmantationClass/
            └ SegmantationObject/
```

2. execute train script.
```
python tools/train.py --epoch 100 --batch_size 8
```

## Reference

- Badrinarayanan, Vijay, Alex Kendall, and Roberto Cipolla. "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." arXiv preprint arXiv:1511.00561, 2015.