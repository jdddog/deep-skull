# Deep Skull
A DICOM skull stripping command line tool, using the model and weights from [CT_BET](https://github.com/aqqush/CT_BET).

## Installation
Clone project:
```bash
git clone git@github.com:jdddog/deep-skull.git
```

Enter deep-skull folder:
```bash
cd deep-skull
```

Create a virtual environment.
```
virtualenv -p python3.8 venv
```

Activate your virtual environment.
```
source venv/bin/activate
```

Install:
```bash
pip install -e .
```

## How to Use
Set the CUDA device you want to use:
```bash
export CUDA_VISIBLE_DEVICES=0
```

To extract brain for CTs:
```bash
deep-skull extract-brain /path/to/nifti ax_CT --num-workers 1
```

To extract brain for CTAs:
```bash
deep-skull extract-brain /path/to/nifti ax_A --num-workers 1
```

To print help text for the command:
```bash
deep-skull extract-brain --help
```