# Tensorboard in fastai

Tensorboard is now supported by pytorch natively (since pytorch 1.1).

This repository shows how to create a custom callback to use tensorboard with the fastai library.

# How to use it

[The demo notebook](https://github.com/tchambon/Tensorboard-in-Fastai/blob/master/Tensorboard%20for%20fastai%20DEMO.ipynb) shows how to use the callback. It's only a few lines of code:

```
# Tensorboard object: used to write the logs
writer = SummaryWriter(comment='Demo')

# Track_weight and track_grad are used to decide if weights and gradients will be logged in TensorBoard.
# Metric names are names to be displayed in Tensorboard. The first is always validation loss
# The order of metric names has to be the same than in learn.metrics
mycallback = partial(TensorBoardFastAI, writer, track_weight=True, track_grad=True, metric_names=['val loss', 'accuracy'])

# Add the callback to the learn object
learn.callback_fns.append(mycallback)
```

By default, the logs used by tensorboard will be in the local dir "./runs".
To launch tensorboard, you have to use the following command (from the repository containing the runs dir):

`tensorboard --logdir=./runs`

To use it in your own project, you can import the TensorBoardCallback directory (for instance with a git submodule) and then, use this import line:

`from TensorBoardCallback import *`

# Prerequisites

- Pytorch 1.1 (important: it doesn't work before v1.1)
- Fastai v1
- Tensorboard

# Tensorboard visualization

![Demo](https://github.com/tchambon/Tensorboard-in-Fastai/blob/master/demo.png "Example tensorboard")

