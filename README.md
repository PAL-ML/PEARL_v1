# Pretrained Encoders are All You Need

Mina Khan, P Srivatsa, Advait Rane, Shriram Chenniappa, Rishabh Anand, Sherjil
Ozair, Pattie Maes

This repo provides code for the benchmark and techniques from the paper [Pretrained Encoders are All You Need](?)

* [📦 Install ](#install) -- Install relevant dependencies and the project
* [🔧 Usage ](#usage) -- Learn how to use PEAYN in colab


## Install
```shell
$ git clone https://github.com/PAL-ML/PEARL_v1.git pearl
$ cd pearl
$ pip install -r requirements.txt
```

Complete installation to run on colab can be found in any of the notebooks in `notebooks/experiments`.

## Usage

1. In your Google Drive, add a shortcut to our processed clip embeddings [drive folder](https://drive.google.com/drive/folders/1WBE9nsfDURndHh73WfaPC9rwrAqfe_GT?usp=sharing)
2. Open any of the jupyter notebooks in `notebooks/experiments` in Google Colab and update the section `Initialization & constants`. Make sure the paths point to where the clip embeddings are saved as in step 1 and where saved probe and encoder checkpoints should be saved in your drive. 

To run a different game using the same parameters, change the `env_name` in `Initialization & constants`. Refer to `game_names.txt` for complete list of games supported.

### Acknowledgements

A significant part of the code in this repo was adapted from the codebase of
[AtariARI](https://github.com/mila-iqia/atari-representation-learning)

### Citation

```
@article{TODO}
}
```
