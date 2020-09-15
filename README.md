<p align="center"><img src="https://user-images.githubusercontent.com/16495490/89590797-bf379000-d859-11ea-8414-1e08aee3a95c.png" width="300"></p>

# DeepTrain

[![Build Status](https://travis-ci.com/OverLordGoldDragon/deeptrain.svg?branch=master)](https://travis-ci.com/OverLordGoldDragon/deeptrain)
[![Coverage Status](https://coveralls.io/repos/github/OverLordGoldDragon/deeptrain/badge.svg?branch=master&service=github)](https://coveralls.io/github/OverLordGoldDragon/deeptrain)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/b3ddf578cd674c268004b0c445c2d695)](https://www.codacy.com/manual/OverLordGoldDragon/deeptrain?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=OverLordGoldDragon/deeptrain&amp;utm_campaign=Badge_Grade)
[![PyPI version](https://badge.fury.io/py/deeptrain.svg)](https://badge.fury.io/py/keras-adamw)
[![Documentation Status](https://readthedocs.org/projects/deeptrain/badge/?version=latest)](https://deeptrain.readthedocs.io/en/latest/?badge=latest)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Full knowledge and control of the train state.


## Features

DeepTrain is founded on **control** and **introspection**: full knowledge and manipulation of the train state.

### Train Loop

  - **Resumability**: interrupt-protection, can pause mid-training
  - **Tracking & reproducibility**: save & load model, train state, random seeds, and hyperparameter info

### Data Pipeline

  - **Flexible batch_size**: can differ from that of loaded files, will split/combine ([ex](https://deeptrain.readthedocs.io/en/latest/examples/misc/flexible_batch_size.html))
  - **Faster SSD loading**: load larger batches to maximize read speed utility
  - **Stateful timeseries**: splits up a batch into windows, and `reset_states()` (RNNs) at end ([ex](https://deeptrain.readthedocs.io/en/latest/examples/misc/timeseries.html))
  
### Introspection & Utilities

  - **Model**: auto descriptive naming ([ex](https://deeptrain.readthedocs.io/en/latest/examples/misc/model_auto_naming.html)); gradients, weights, activations visuals ([ex](https://deeptrain.readthedocs.io/en/latest/examples/callbacks/mnist.html#Init-&-train))
  - **Train state**: image log of key attributes for easy reference ([ex](https://deeptrain.readthedocs.io/en/latest/examples/advanced.html#Inspect-generated-logs)); batches marked w/ "set nums" - know what's being fit and when
  - **Algorithms, preprocesing, calibration**: tools for inspecting & manipulating data and models

[Complete list](https://deeptrain.readthedocs.io/en/latest/why_deeptrain.html)

## When is DeepTrain suitable (and not)?

Training _few_ models _thoroughly_: closely tracking model and train attributes to debug performance and inform next steps.

DeepTrain is _not_ for models that take under an hour to train, or for training hundreds of models at once.

## What does DeepTrain do?

Abstract away boilerplate train loop and data loading code, *without* making it into a black box. Code is written intuitively and fully documented.
Everything about the train state can be seen via *dedicated attributes*; which batch is being fit and when, how long until an epoch ends, intermediate metrics, etc.

DeepTrain is *not* a "wrapper" around TF; while currently only supporting TF, fitting and data logic is framework-agnostic.

## How it works

<p align="center"><img src="https://raw.githubusercontent.com/OverLordGoldDragon/deeptrain/master/docs/source/_images/train_loop.png" width="700"></p>

<img src="https://raw.githubusercontent.com/OverLordGoldDragon/deeptrain/master/docs/source/_images/train_val.gif" width="450" align="right">

 1. We define `tg = TrainGenerator(**configs)`,
 2. call `tg.train()`.<br>
 3. `get_data()` is called, returning data & labels,<br>
 4. fed to `model.fit()`, returning `metrics`,<br>
 5. which are then printed, recorded.<br>
 6. The loop repeats, or `validate()` is called.<br>

Once `validate()` finishes, training may checkpoint, and `train()` is called again. Internally, data loads with `DataGenerator.load_data()` (using e.g. `np.load`).

That's the high-level overview; details [here](https://deeptrain.readthedocs.io/en/latest/how_works.html). Callbacks & other behavior can be configured for every stage of training.

## Examples

<a href="https://deeptrain.readthedocs.io/en/latest/examples/advanced.html">MNIST AutoEncoder</a> | <a href="https://deeptrain.readthedocs.io/en/latest/examples/misc/timeseries.html">Timeseries Classification</a> | <a href="https://deeptrain.readthedocs.io/en/latest/examples/misc/model_health.html">Health Monitoring</a>
:----------------:|:-----------------:|:-----------------:
<a href="https://deeptrain.readthedocs.io/en/latest/examples/advanced.html"><img src="https://raw.githubusercontent.com/OverLordGoldDragon/deeptrain/master/docs/source/_images/mnist.gif" width="210" height="210"><a>|<a href="https://deeptrain.readthedocs.io/en/latest/examples/misc/timeseries.html"><img src="https://raw.githubusercontent.com/OverLordGoldDragon/deeptrain/master/docs/source/_images/ecg2.png" width="210" height="210"></a>|<a href="https://deeptrain.readthedocs.io/en/latest/examples/misc/model_health.html"><img src="https://raw.githubusercontent.com/OverLordGoldDragon/deeptrain/master/docs/source/_images/model_health.png" width="210" height="210"></a>
  
<a href="https://deeptrain.readthedocs.io/en/latest/examples/callbacks/mnist.html">Tracking Weights</a> | <a href="https://deeptrain.readthedocs.io/en/latest/examples/misc/reproducibility.html">Reproducibility</a> | <a href="https://deeptrain.readthedocs.io/en/latest/examples/misc/flexible_batch_size.html">Flexible batch_size</a>
:----------------:|:----------------:|:----------------:|
<a href="https://deeptrain.readthedocs.io/en/latest/examples/callbacks/mnist.html"><img src="https://raw.githubusercontent.com/OverLordGoldDragon/deeptrain/master/docs/source/_images/gradients.gif" width="210" height="210"></a>|<a href="https://deeptrain.readthedocs.io/en/latest/examples/misc/reproducibility.html"><img src="https://raw.githubusercontent.com/OverLordGoldDragon/deeptrain/master/docs/source/_images/reproducibility.png" width="210" height="210"></a>|<a href="https://deeptrain.readthedocs.io/en/latest/examples/misc/flexible_batch_size.html"><img src="https://raw.githubusercontent.com/OverLordGoldDragon/deeptrain/master/docs/source/_images/flexible_batch_size.png" width="210" height="210"></a>


## Installation

`pip install deeptrain` (without data; see [how to run examples](https://deeptrain.readthedocs.io/en/latest/how_to.html#run-examples)), or clone repository

## Quickstart

To run, DeepTrain requires (1) a compiled model; (2) data directories (train & val). Below is a minimalistic example.

Checkpointing, visualizing, callbacks & more can be accomplished via additional arguments; see [Basic](https://deeptrain.readthedocs.io/en/latest/examples/basic.html) and [Advanced](https://deeptrain.readthedocs.io/en/latest/examples/advanced.html) examples. 
Also see [Recommended Usage](https://deeptrain.readthedocs.io/en/latest/recommended_usage.html).

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from deeptrain import TrainGenerator, DataGenerator

ipt = Input((16,))
out = Dense(10, 'softmax')(ipt)
model = Model(ipt, out)
model.compile('adam', 'categorical_crossentropy')

dg  = DataGenerator(data_path="data/train", labels_path="data/train/labels.npy")
vdg = DataGenerator(data_path="data/val",   labels_path="data/val/labels.npy")
tg  = TrainGenerator(model, dg, vdg, epochs=3, logs_dir="logs/")

tg.train()
```

## In future releases

 - `MetaTrainer`: direct support for dynamic model recompiling with changing hyperparameters, and optimizing thereof
 - PyTorch support
