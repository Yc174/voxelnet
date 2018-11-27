
# VoxelNet

## Introduction

This is a reproduction of voxelnet based on PyTorch.  The original can be referred to [link](https://arxiv.org/abs/1711.06396).

### Major features
- **Pytorch**

- **Multi GPUs**


## Updates


## Benchmark and model zoo


## Installation

### Requirements

- Linux (tested on Ubuntu 16.04 )
- Python 3.6
- PyTorch 0.4.0 and torchvision
- Cython
- tensorboard
- [mayavi](https://docs.enthought.com/mayavi/mayavi/)

### Install VoxelNet

a. Install PyTorch 0.4.0 and torchvision following the [official instructions](https://pytorch.org/).

b. Clone the VoxelNet repository.

```shell
git clone https://github.com/Yc174/voxelnet.git
```

c. Compile cuda extensions.

```shell
cd lib/extensions/_nms
sh build.sh
```


### Prepare Kitti dataset.

It is recommended to symlink the dataset root to `$voxelnet/datasets`.

```
voxelnet
├── lib
├── tools
├── experiments
├── datasets
│   ├── KITTI
│   │   ├── object
│   │   │     ├── training
│   │   │     ├── testing
```


## Train a model
- [x] Multi GPUs training
- [x] use tensorboard to visualize loss
- [x] use validate dataset


To train kitti dataset and save the model.

```shell
cd voxelnet
./experiments/run_trainval.sh
```


## Inference with pretrained models

### Test a dataset

- [x] single GPU testing
- [x] visualize detection results


To test kitti dataset and visualize the results.

```shell
cd voxelnet
./experiments/run_test.sh
```



