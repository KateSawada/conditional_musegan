# Conditional MuseGAN

## Environment setup

```bash
$ cd conditional_musegan
$ pip install -e .
```

## Folder architecture
coming soon

## Run

In this repo, hyperparameters are managed using [Hydra](https://hydra.cc/docs/intro/).<br>
Hydra provides an easy way to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

### Dataset preparation
coming soon

### Training

```bash
# Train a model customizing the hyperparameters as you like
$ conditional_musegan-train data=lod_small out_dir=exp/conditional_musegan
```

### Inference

```bash
# Decode with several F0 scaling factors
$ conditional_musegan-decode data=lpd_small out_dir=exp/conditional_musegan checkpoint_steps=400000
```

### Analysis-Synthesis

coming soon

### Monitor training progress

```bash
$ tensorboard --logdir exp
```
