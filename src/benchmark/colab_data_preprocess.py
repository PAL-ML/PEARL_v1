import numpy as np
import os
import torch

from .episodes import get_episodes

def save_npy(filepath, data):
	np.savez_compressed(filepath, data=data)
	print("Data saved to {}".format(filepath))

def load_npy(filepath, file_name='arr_0'):
	loaded_data = np.load(filepath, allow_pickle=True, mmap_mode="r")
	print("Data loaded from {}".format(filepath))
	return loaded_data[file_name]

def get_episode_data(images_n_labels_dir, env_name, steps, collect_mode, color=True):
  try:
    tr_episodes = load_npy(os.path.join(images_n_labels_dir, "train_eps.npz"))
    tr_labels = load_npy(os.path.join(images_n_labels_dir, "train_labels.npz"))
    val_episodes = load_npy(os.path.join(images_n_labels_dir, "val_eps.npz"))
    val_labels = load_npy(os.path.join(images_n_labels_dir, "val_labels.npz"))
    test_episodes = load_npy(os.path.join(images_n_labels_dir, "test_eps.npz"))
    test_labels =load_npy(os.path.join(images_n_labels_dir, "test_labels.npz"))
  except:
    print("Unable to load data from drive...")
    tr_episodes, val_episodes,\
    tr_labels, val_labels,\
    test_episodes, test_labels = get_episodes(env_name=env_name, 
                                        steps=steps, 
                                        collect_mode=collect_mode,
                                        color=color)
  return tr_episodes, val_episodes, tr_labels, val_labels, test_episodes, test_labels
  
def get_file_names(input_resolution):

  if input_resolution == "full-image":
    return "clip_embeddings_train", "clip_embeddings_val", "clip_embeddings_test"
  elif input_resolution == "2x2patch":
    return "clip_embeddings_2x2_patches_train", "clip_embeddings_2x2_patches_val", "clip_embeddings_2x2_patches_test"
  elif input_resolution == "4x4patch":
    return "clip_embeddings_4x4_patches_train", "clip_embeddings_4x4_patches_val", "clip_embeddings_4x4_patches_test"
  elif input_resolution == "gaussian_blur":
    return "gaussian_blur_clip_embeddings_train", "gaussian_blur_clip_embeddings_val", "gaussian_blur_clip_embeddings_test"
  elif input_resolution == "colour_jitter":
    return "colour_jitter_clip_embeddings_train", "colour_jitter_clip_embeddings_val", "colour_jitter_clip_embeddings_test"
  elif input_resolution == "random_crop":
    return "random_crop_clip_embeddings_train", "random_crop_clip_embeddings_val", "random_crop_clip_embeddings_test"
  elif input_resolution == "image-diff":
    return "clip_embeddings_image_diff_train_new","clip_embeddings_image_diff_val_new", "clip_embeddings_image_diff_test_new"
  elif input_resolution == "6x6patch":
    return "clip_embeddings_6x6_patches_train", "clip_embeddings_6x6_patches_val", "clip_embeddings_6x6_patches_test"
  else:
    raise Exception("Invalid input resolution... choose among 'full-image', '2x2patch' & '4x4patch'")
    
def get_embedding_data(embeddings_dir, input_resolution="full-image"):

  tr, val, test = get_file_names(input_resolution)

  try:
    tr_episodes = torch.load(os.path.join(embeddings_dir, tr))
    tr_labels = load_npy(os.path.join(embeddings_dir, "train_labels.npz"))
    val_episodes = torch.load(os.path.join(embeddings_dir, val))
    val_labels = load_npy(os.path.join(embeddings_dir, "val_labels.npz"))
    test_episodes = torch.load(os.path.join(embeddings_dir, test))
    test_labels = load_npy(os.path.join(embeddings_dir, "test_labels.npz"))

  except:
    raise Exception("Unable to load embedding data from drive...")

  return tr_episodes, val_episodes, tr_labels, val_labels, test_episodes, test_labels

def get_data(data_type, data_dir, env_name, steps, collect_mode, color=True, input_resolution="full-image"):
  if data_type == "embeddings":
    tr_episodes, val_episodes,\
    tr_labels, val_labels,\
    test_episodes, test_labels = get_embedding_data(data_dir, input_resolution=input_resolution)
  elif data_type == "images":
    tr_episodes, val_episodes,\
    tr_labels, val_labels,\
    test_episodes, test_labels = get_episode_data(data_dir, env_name=env_name, steps=steps, collect_mode=collect_mode, color=True)
  else:
    raise Exception("Invalid data type... choose between 'embeddings' & 'images'")
  
  return tr_episodes, val_episodes, tr_labels, val_labels, test_episodes, test_labels


def get_split_embedding_data(embeddings_dir, input_resolution="full-image", split="test"):

  tr, val, test = get_file_names(input_resolution)

  try:
    if split == "test":
      episodes = torch.load(os.path.join(embeddings_dir, test))
      labels = load_npy(os.path.join(embeddings_dir, "test_labels.npz"))
    elif split == "train":
      episodes = torch.load(os.path.join(embeddings_dir, tr))
      labels = load_npy(os.path.join(embeddings_dir, "train_labels.npz"))
    elif split == "val":
      episodes = torch.load(os.path.join(embeddings_dir, val))
      labels = load_npy(os.path.join(embeddings_dir, "val_labels.npz"))

    return episodes, labels
  except:
    raise Exception("Unable to load embedding data from drive...")


def get_split_img_data(data_dir, split="test"):
  try:
    if split == "train":
      episodes = load_npy(os.path.join(data_dir, "train_eps.npz"))
      labels = load_npy(os.path.join(data_dir, "train_labels.npz"))
    elif split == "val":
      episodes = load_npy(os.path.join(data_dir, "val_eps.npz"))
      labels = load_npy(os.path.join(data_dir, "val_labels.npz"))
    elif split == "test":
      episodes = load_npy(os.path.join(data_dir, "test_eps.npz"))
      labels =load_npy(os.path.join(data_dir, "test_labels.npz"))
    
    return episodes, labels
  except:
    print("Unable to load data from drive...")


def get_data_split(data_type, data_dir, input_resolution="full-image", split="test"):
  if split not in ["train", "val", "test"]:
    raise Exception("Invalid split type...choose between 'train', 'val', 'test'")
  
  if data_type == "embeddings":
    episodes, labels = get_split_embedding_data(data_dir, input_resolution=input_resolution, split=split)
  elif data_type == "images":
    episodes, labels = get_split_img_data(data_dir, split=split)
  else:
    raise Exception("Invalid data type... choose between 'embeddings' & 'images'")
  
  return episodes, labels

'''
def np_to_tensor(tr_eps, val_eps, test_eps):
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  tr_eps_tensors = [torch.from_numpy(np.array(x)).to(device) for x in tr_eps]
  val_eps_tensors = [torch.from_numpy(np.array(x)).to(device) for x in val_eps]
  test_eps_tensors = [torch.from_numpy(np.array(x)).to(device) for x in test_eps]
  return tr_eps_tensors, val_eps_tensors, test_eps_tensors
'''

def concatcat_patch_embeddings(eps, num_patches=4):
  processed_eps = []
  for i, ep in enumerate(eps):
    processed_ep = []
    for j, s in enumerate(ep):
      if j % num_patches == 0:
        processed_emb = torch.cat(eps[i][j:j+num_patches], dim=0)
        processed_ep.append(processed_emb)
    processed_eps.append(processed_ep)
  return processed_eps

def concat_patch_embeddings_with_full_img(patch_eps, full_eps, num_patches=4):
  processed_eps = []
  for i, ep in enumerate(patch_eps):
    processed_ep = []
    for j, s in enumerate(ep):
      if j % num_patches == 0:
        processed_emb = torch.cat(patch_eps[i][j:j+num_patches], dim=0)
        index_full =  j // num_patches
        processed_emb = torch.cat([full_eps[i][index_full], processed_emb], dim=0)
        processed_ep.append(processed_emb)
    processed_eps.append(processed_ep)
  return processed_eps

def concat_multiple_patch_embeddings_with_full_img(patch_eps1, patch_eps2, full_eps, num_patches1=16, num_patches2=4):
  processed_eps = []
  for i, ep in enumerate(patch_eps1):
    processed_ep = []
    for j, s in enumerate(ep):
      if j % num_patches1 == 0:
        processed_emb_4x4 = torch.cat(patch_eps1[i][j:j+num_patches1], dim=0)
        index_full =  j // num_patches1
        index_2x2 = j // num_patches2
        processed_emb_2x2 = torch.cat(patch_eps2[i][index_2x2:index_2x2+num_patches1], dim=0)
        processed_emb = torch.cat([processed_emb_4x4, processed_emb_2x2, full_eps[i][index_full]], dim=0)
        processed_ep.append(processed_emb)
    processed_eps.append(processed_ep)
  return processed_eps

def squeeze_tensors(eps):
  for i, ep in enumerate(eps):
    for j, s in enumerate(ep):
      eps[i][j] = torch.squeeze(eps[i][j])
  return eps