# Few-Shot learning with Prototypes Networks
Simple implementation of prototypical networks in few shot learning

<details>
<summary>Installation</summary>

Create a conda/virtualenv with all necessary packages:

### Conda

`conda create --name fs-learn --file ./requirements.txt`

`conda activate fs-learn`

### Venv

`python -m pip install virtualenv`

`virtualenv venv-fs-learn`

`source venv/bin/activate`

`python -m pip install ./requirements.txt`

</details>

<details>
<summary>Datasets</summary>

We used 3 main classification datasets:
- **mini_imagenet**: a collection of 100 real-world objects classes as rgb images.
  - total: 60,000
  - splits: 64 train, 16 val, 20 test (according to Vinyals et al)
  - Used in paper
- **omniglot**: a collection of 1623 classes of handwritted characters. Each image is then rotated 3 more times by 90 degrees.
  - total: 32460 real, plus 4 rotations per image
  - splits: 1032 train, 172 val, 464 test (according to Vinyals et al)
  - Used in paper
- **flowers102**: a collection of 102 real-world flowers classes as rgb images.
  - total: 32460 real, plus 4 rotations per image
  - splits: 64 train, 16 val, 22 test (random seed for splits)
  - **NOT** Used in paper
</details>

<details open>
<summary>Usage</summary>

The starter script is **train.py** that has all necessary params to train and validate on a dataset.

To replicate the results, launch this training:

```bash
python train.py --dataset mini_imagenet \
                --epochs 200 \
                --train-num-class 30 \
                --test-num-class 5 \
                --number-support 5 \
                --train-num-query 15 \
                --episodes-per-epoch 100 \
                --adam-lr 0.001 \
                --opt-step-size 20 \
                --opt-gamma 0.5 \
                --save-each 5
```

Implemented datasets are [omniglot, mini_imagenet, flowers102]:

</details>

<details open>
<summary>Experiments</summary>

Lots of experiments were done using basic paper's data by replicating the training.
All of these uses **epochs=200** and **iterations_per_epoch=100**.

| Dataset       | Paper res<br><sup>(Acc) | Our res<br><sup>(Acc)                                                                                                  | Images<br><sup>(shape) | Prototype<br><sup>(shape) | Duration     |
|---------------|-------------------------|------------------------------------------------------------------------------------------------------------------------|------------------------|---------------------------|--------------|
| mini_imagenet | 68.20                   | [65.24](https://github.com/fabian57fabian/fewshot-learning-prototypical-networks/results/mini_imagenet/train_5way.png) | (84, 84, 3)            | (batch, 1600)             | cpu / 13h20m |
| omniglot      | 98.80                   | [97.69](https://github.com/fabian57fabian/fewshot-learning-prototypical-networks/results/omniglot/train_5way.png)       | (28, 28, 1)            | (batch, 60)               | cpu / 2h35m  |
| flowers102    | /                       | [73.81](https://github.com/fabian57fabian/fewshot-learning-prototypical-networks/results/flowers102/training_.png)     | (64, 64, 3)            | (batch, TODO)             | cpu / 5h48m  |

</details>