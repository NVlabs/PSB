
# PSB: Parallelized Spatiotemporal Binding
*ICML 2024*

#### [[arXiv](https://arxiv.org/abs/2402.17077)] [[project](https://parallel-st-binder.github.io/)]

This is the **official PyTorch implementation** of the PSB model.

### Authors
Gautam Singh, Yue Wang, Jiawei Yang, Boris Ivanovic, Sungjin Ahn*, Marco Pavone* and Tong Che (* denotes equal advising)

<img src="https://parallel-st-binder.github.io/figure_one.png">

### Dataset
The MOVi-A/B datasets can be downloaded using [this script](https://github.com/singhgautam/steve/blob/master/download_movi.py). 

### Training
To train the model, simply execute:
```bash
python train.py
```
Check `train.py` to see the full list of training arguments. You can use the `--data_path` argument to point to the path of your dataset.

### Outputs
The training code produces Tensorboard logs. To see these logs, run Tensorboard on the logging directory that was provided in the training argument `--log_path`. These logs contain the training loss curves and visualizations of reconstructions and object attention maps.

### Packages Required
The following packages may need to be installed first.
- [PyTorch](https://pytorch.org/)
- [TensorBoard](https://pypi.org/project/tensorboard/) for logging.
- [MoviePy](https://pypi.org/project/moviepy/) to produce video visualizations in the tensorboard logs.

### Code Files
This repository provides the following files.
- `train.py` contains the training script.
- `psb.py` provides the implementation of PSB.
- `data.py` contains the dataset class.
- `model.py` provides the model classes for the PSB-based video auto-encoder.
- `utils.py` provides helper classes and functions.

### Citation
```
@inproceedings{
  singh2024psb,
  title={Parallelized Spatiotemporal Binding},
  author={Gautam Singh and Yue Wang and Jiawei Yang and Boris Ivanovic and Sungjin Ahn and Marco Pavone and Tong Che},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```