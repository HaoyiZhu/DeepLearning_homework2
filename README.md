# 深度学习第二次小作业

## Requirements

- Paddle==2.2.1
- python>=3.8
- tqdm
- pyyaml
- easydict

## Quick Start

- To run baseline (default is ResNet-50) on original MNIST:

```bash
python main.py (recommended)--sync
```

- To run baseline (default is ResNet-50) on reduced MNIST (will reserve only 10% of 0-4):

```
python main.py (recommended)--sync --reduce_dataset
```

- To use my improvement algorithms, you can flexibly set their flags:

```
python main.py --sync --reduce_dataset (optional)--label_weighted (optional)--ssp (optional)--label_smooth
```

- Note that `label_smooth` may harm the performance.

- For more flags and details, or if you want to use other models or customized settings, please refer to [opt.py](./opt.py) and [./configs/](./configs/).

## Experiment Logs

- You can see my experiment logs in [./exp/](./exp/).

