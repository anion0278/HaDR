
# HaDR: Applying Domain Randomization for Generating Synthetic Multimodal Dataset for Hand Instance Segmentation in Cluttered Industrial Environments

## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+
- PyTorch 1.1 or higher (>=1.5 is not tested)
- CUDA 9.0 or higher
- NCCL 2
- GCC 4.9 or higher
- [mmcv 0.2.16](https://github.com/open-mmlab/mmcv/tree/v0.2.16)

### Install

a. Create a conda virtual environment and activate it.

```
conda create -n mmdet python=3.7 -y
conda activate mmdet
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```
conda install pytorch torchvision -c pytorch
```

c. Clone the repository.

```shell
git clone https://github.com/anion0278/HaDR.git
cd HaDR
```

d. Install build requirements and then install.

```shell
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e .  # or "python setup.py develop"
```

## Dataset and pretrained models
The dataset and pretrained SOLOv2 and Mask R-CNN models can be found on [Kaggle](https://www.kaggle.com/datasets/alevysock/hadr-dataset-for-hands-instance-segmentation).

## Demo
![demo](https://github.com/anion0278/HaDR/assets/23017063/6a5750f8-84c7-48af-a382-a2bc446ad975)


