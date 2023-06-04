# IR-BPP: Learning Physically Realizable Skills for Online Packing of General 3D Shapes


We develop a learning-based solver for packing arbitrarily-shaped objects in a physically realizable problem setting, which is arguably the most challenging setting of bin packing problems. This work is newly been accpted by **ACM Transactions on Graphics (TOG)**. 
See these links for video demonstration: [YouTube](https://www.youtube.com/watch?v=z4Q05EGcW64&t=56s), [bilibili](https://www.bilibili.com/video/BV1ho4y1M7gG/)


We release our source code and build well-established benchmark datasets. Our datasets consisting of training and testing objects, separated into discrete cuboidal sets, continuous cuboidal sets, and large-scale irregular sets with various geometric characteristics, and specifying a container size. 

As our TOG reviewers suggested, although there have been numerous packing papers using RL for higher packing density, there is a lack of a **common** dataset to benchmark performance. We believe that having such a common benchmark would greatly facilitate research and comparison of different techniques.


**This repo is being continuously updated, please stay tuned!**


If you are interested, please star this repo! 

![PCT](images/teaser.png)

## Paper
For more details, please see our paper [Learning Physically Realizable Skills for Online Packing of General 3D Shapes](https://openreview.net/forum?id=bfuGjlCwAq). If this code is useful for your work, please cite our paper:

```
@article{zhao2022learning,
  title={Learning Physically Realizable Skills for Online Packing of General 3D Shapes},
  author={Zhao, Hang and Pan, Zherong and Yu, Yang and Xu, Kai},
  journal={arXiv preprint arXiv:2212.02094},
  year={2022}
}
``` 


[//]: # (## Dependencies)

[//]: # (* NumPy)

[//]: # (* gym)

[//]: # (* Python>=3.7)

[//]: # (* [PyTorch]&#40;http://pytorch.org/&#41; >=1.7)

[//]: # (* My suggestion: Python == 3.7, gym==0.13.0, torch == 1.10, OS: Ubuntu 16.04)

[//]: # (## Quick start)

[//]: # ()
[//]: # (For training online 3D-BPP on setting 2 &#40;mentioned in our paper&#41; with our PCT method and the default arguments:)

[//]: # (```bash)

[//]: # (python main.py )

[//]: # (```)

[//]: # (The training data is generated on the fly. The training logs &#40;tensorboard&#41; are saved in './logs/runs'. Related file backups are saved in './logs/experiment'.)

[//]: # ()
[//]: # (## Usage)

[//]: # ()
[//]: # (### Data description)

[//]: # ()
[//]: # (Describe your 3D container size and 3D item size in 'givenData.py')

[//]: # (```)

[//]: # (container_size: A vector of length 3 describing the size of the container in the x, y, z dimension.)

[//]: # (item_size_set:  A list records the size of each item. The size of each item is also described by a vector of length 3.)

[//]: # (```)

[//]: # (If you need to )

[//]: # (### Dataset)

[//]: # (You can download the prepared dataset from [here]&#40;https://drive.google.com/drive/folders/1QLaLLnpVySt_nNv0c6YetriHh0Ni-yXY?usp=sharing&#41;.)

[//]: # (The dataset consists of 3000 randomly generated trajectories, each with 150 items. The item is a vector of length 3 or 4, the first three numbers of the item represent the size of the item, the fourth number &#40;if any&#41; represents the density of the item.)

[//]: # ()
[//]: # (### Model)

[//]: # (We provide [pretrained models]&#40;https://drive.google.com/drive/folders/14PC3aVGiYZU5AaGdNM9YOVdp8pPiZ3fe?usp=sharing&#41; trained using the EMS scheme in a discrete environment, where the bin size is &#40;10,10,10&#41; and the item size range from 1 to 5.)

[//]: # ()
[//]: # (### Training)

[//]: # ()
[//]: # (For training online 3D BPP instances on setting 1 &#40;80 internal nodes and 50 leaf nodes&#41; nodes:)

[//]: # (```bash)

[//]: # (python main.py --setting 1 --internal-node-holder 80 --leaf-node-holder 50)

[//]: # (```)

[//]: # (If you want to train a model that works on the **continuous** domain, add '--continuous', don't forget to change your problem in 'givenData.py':)

[//]: # (```bash)

[//]: # (python main.py --continuous --setting 1 --internal-node-holder 80 --leaf-node-holder 50)

[//]: # (```)

[//]: # (#### Warm start)

[//]: # (You can initialize a run using a pretrained model:)

[//]: # (```bash)

[//]: # (python main.py --load-model --model-path path/to/your/model)

[//]: # (```)

[//]: # ()
[//]: # (### Evaluation)

[//]: # (To evaluate a model, you can add the `--evaluate` flag to `evaluation.py`:)

[//]: # (```bash)

[//]: # (python evaluation.py --evaluate --load-model --model-path path/to/your/model --load-dataset --dataset-path path/to/your/dataset)

[//]: # (```)

[//]: # (### Heuristic)

[//]: # (Running heuristic.py for test heuristic baselines, the source of the heuristic algorithm has been marked in the code:)

[//]: # ()
[//]: # (Running heuristic on setting 1 （discrete） with LASH method:)

[//]: # (```)

[//]: # (python heuristic.py --setting 1 --heuristic LSAH --load-dataset  --dataset-path setting123_discrete.pt)

[//]: # (```)

[//]: # ()
[//]: # (Running heuristic on setting 2 （continuous） with OnlineBPH method:)

[//]: # (```)

[//]: # (python heuristic.py --continuous --setting 2 --heuristic OnlineBPH --load-dataset  --dataset-path setting2_continuous.pt)

[//]: # (```)

[//]: # ()
[//]: # (### Help)

[//]: # (```bash)

[//]: # (python main.py -h)

[//]: # (python evaluation.py -h)

[//]: # (python heuristic.py -h)

[//]: # (```)

[//]: # ()
### License
```
This source code is released only for academic use. Please do not use it for commercial purpose without authorization of the author.
```
