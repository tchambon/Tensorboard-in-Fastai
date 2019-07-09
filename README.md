# Tensorboard in fastai

Tensorboard is now supported by pytorch natively (since pytorch 1.1).

This repository shows how to create a custom callback to use tensorboard with the fastai library.

# How to use it

[The demo notebook](https://github.com/tchambon/Tensorboard-in-Fastai/blob/master/Tensorboard%20for%20fastai%20DEMO.ipynb) shows how to use the callback. It's only a few lines of codes.

By default, the logs used by tensorboard will be in the local dir "./runs".
To launch tensorboard, you have to use the following command (from the repository containing the runs dir):

`tensorboard --logdir=./run`

# Prerequisites

- Pytorch 1.1 (important: it doesn't work before v1.1)
- Fastai v1
- Tensorboard

# Tensorboard visualization

![Demo](https://github.com/tchambon/Tensorboard-in-Fastai/blob/master/demo.png "Example tensorboard")

