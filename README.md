# pytorch_DeepMimic

This repository is an implementation of DeepMimic, a Reinforcement Learning scheme designed to learn policies for imitating a wide range of example motions. The original DeepMimic implementation was in TensorFlow 1.15, and this repository contains a translated version to PyTorch 1.12.

## Translation Work

This repository focuses on translating the original DeepMimic implementation from TensorFlow 1.15 to PyTorch 1.12. The goal is to maintain the functionality of the original while providing users with the flexibility and advantages of PyTorch.


## Installation

To install the repo, you might want to install the following packages beforehand:

- gym==0.23.1
- pybullet==3.2.5
- mpi4py==3.1.4
- pytorch==1.12.1
- numpy==1.25.2

```bash
git clone https://github.com/myiKim/pytorch_DeepMimic.git
cd pytorch_DeepMimic
```

## Training and Validation

To train and validate, you want to run the followings on the project directory:

```bash
cd deepmimic
python DeepMimic_Optimizer.py --arg_file args/run_humanoid3d_spinkick_args.txt
```

For additional details regarding arg files, kindly consult the original DeepMimic repository created by Peng (https://github.com/xbpeng/DeepMimic).

Please note that I did the training on local PC (CPU).
Model files will be tracked on the folder "output" 
The RL algorithm is based on Proximal Policy Optimization (https://arxiv.org/abs/1707.06347)
- agent0_model_anet.pth for actor network
- agent0_model_cnet.pth for critic network


## Inference
For inference, 

```bash
cd deepmimic
python testrl.py
```


## Bug Reporting

If you encounter any bugs or issues while using this repository, please help us by reporting them. Your feedback is valuable and contributes to the improvement of this project.

To report a bug:

- Create a new issue with a clear and detailed description of the bug, including:
  - Steps to reproduce the issue.
  - Expected behavior.
  - Actual behavior.
  - Any error messages or stack traces.

For sensitive or private bug reports, you can contact us directly by emailing xxx@gmail.com.

Thank you for your assistance in enhancing this project!


## Acknowledgements
This implementation builds upon the DeepMimic project (https://github.com/xbpeng/DeepMimic), crafted by Xubin Peng. I extend my appreciation to the original authors for generously sharing their code and models, which played an instrumental role in shaping this project. Additionally, for the environment code, I utilized code adapted from pybullet (https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/deep_mimic/gym_env/deep_mimic_env.py).