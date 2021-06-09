# Pretrained Encoders are All You Need

Mina Khan, P Srivatsa, Advait Rane, Shriram Chenniappa, Rishabh Anand, Sherjil
Ozair, Pattie Maes

This repo provides code for the benchmark and techniques from the paper [Pretrained Encoders are All You Need](?)

* [📦 Install ](#install) -- Install relevant dependencies and the project
* [🏃 Usage ](#usage) -- Learn how to use PEARL in colab
* [🔧 Change Configurations ](#change-configurations) -- Change configuration to run different experiments
* [💾 Save Embeddings ](#save-embeddings) -- Generate clip embeddings for selected games


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
3. Run the notebook

> To run without using our saved clip embeddings, change `training_input` in `Initialization & constants` from `embeddings` to `images`. Note that making this change would require you to make several changes to the notebooks we provide, inclduing the encoder used (to CLIPEncoder).

## Change configurations

#### Change game

To run a different game using the same parameters, change the `env_name` in `Initialization & constants`. Refer to `game_names.txt` for complete list of supported games.

#### Change parameters

To change parameters, refer to relevant section in `Initialization & constants`.

#### Change training methods for encoder/probe

To change the training methods for encoders, refer to template notebooks in `notebooks/experiments`.

#### Change probe used

Change `probe_type` in `Initialization & constants` to match any of the available probes in `src/benchmark/probe.py`

## Save embeddings

To generate and save the CLIP embeddings we used in our experiments, refer to the notebooks in `notebooks/save_embeddings`. These would save the embeddings to Google Drive.

### Acknowledgements

A significant part of the code in this repo was adapted from the codebase of
[AtariARI](https://github.com/mila-iqia/atari-representation-learning)

### Citation

```
@article{TODO}
}
```
