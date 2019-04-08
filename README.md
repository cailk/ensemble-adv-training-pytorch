# Ensemble Adversarial Training With Pytorch

This repository contains pytorch code to reproduce results from the paper:

**Ensemble Adversarial Training: Attacks and Defenses** <br>
*Florian Tramèr, Alexey Kurakin, Nicolas Papernot, Dan Boneh and Patrick McDaniel* <br>
ArXiv report: https://arxiv.org/abs/1705.07204

<br>

###### REQUIREMENTS

The code was tested with Python 3.6.7 and Pytorch 1.0.1.

###### EXPERIMENTS

Training a few simple MNIST models. These are described in _mnist.py_.

```
python -m train models/modelA --type=0
python -m train models/modelB --type=1
python -m train models/modelC --type=2
python -m train models/modelD --type=3
```

(standard) Adversarial Training:

```
python -m train_adv models/modelA_adv --type=0 --epochs=12
```
Ensemble Adversarial Training:
```
python -m train_adv models/modelA_ens models/modelA models/modelC models/modelD --type=0 --epochs=12
```

The accuracy of the models on the MNIST test set can be computed using

```
python -m simple_eval test [model(s)]
```

To evaluate robustness to various attacks

```
python -m simple_eval [attack] [source_model] [target_model(s)] [--parameters (opt)]
```

###### REFERENCE
1. Author's code: [ftramer/ensemble-adv-training](https://github.com/ftramer/ensemble-adv-training)
