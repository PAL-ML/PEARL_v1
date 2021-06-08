import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, Normalize
from a2c_ppo_acktr.utils import init
import time
from atariari.benchmark.utils import download_run
from atariari.benchmark.episodes import checkpointed_steps_full_sorted
import os
import clip
from PIL import Image

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            Conv2dSame(in_channels, out_channels, 3)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class ImpalaCNN(nn.Module):
    def __init__(self, input_channels, args):
        super(ImpalaCNN, self).__init__()
        self.hidden_size = args.feature_size
        self.depths = [16, 32, 32, 32]
        self.downsample = not args.no_downsample
        self.layer1 = self._make_layer(input_channels, self.depths[0])
        self.layer2 = self._make_layer(self.depths[0], self.depths[1])
        self.layer3 = self._make_layer(self.depths[1], self.depths[2])
        self.layer4 = self._make_layer(self.depths[2], self.depths[3])
        if self.downsample:
            self.final_conv_size = 32 * 9 * 9
        else:
            self.final_conv_size = 32 * 12 * 9
        self.final_linear = nn.Linear(self.final_conv_size, self.hidden_size)
        self.flatten = Flatten()
        self.train()

    def _make_layer(self, in_channels, depth):
        return nn.Sequential(
            Conv2dSame(in_channels, depth, 3),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(),
            ResidualBlock(depth, depth),
            nn.ReLU(),
            ResidualBlock(depth, depth)
        )

    @property
    def local_layer_depth(self):
        return self.depths[-2]

    def forward(self, inputs, fmaps=False):
        f5 = self.layer3(self.layer2(self.layer1(inputs)))

        if not self.downsample:
            out = self.layer4(f5)
        else:
            out = f5

        out = F.relu(self.final_linear(self.flatten(out)))

        if fmaps:
            return {
                'f5': f5.permute(0, 2, 3, 1),
                'out': out
            }

        return out

class NatureCNN(nn.Module):

    def __init__(self, input_channels, args):
        super().__init__()
        self.feature_size = args.feature_size
        self.hidden_size = self.feature_size
        self.downsample = not args.no_downsample
        self.input_channels = input_channels
        self.end_with_relu = args.end_with_relu
        self.args = args
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.flatten = Flatten()

        if self.downsample:
            self.final_conv_size = 32 * 7 * 7
            self.final_conv_shape = (32, 7, 7)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                #nn.ReLU()
            )
        else:
            self.final_conv_size = 64 * 9 * 6
            self.final_conv_shape = (64, 9, 6)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 128, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(128, 64, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                #nn.ReLU()
            )
        self.train()

    @property
    def local_layer_depth(self):
        return self.main[4].out_channels

    def forward(self, inputs, fmaps=False):
        f5 = self.main[:6](inputs)
        f7 = self.main[6:8](f5)
        out = self.main[8:](f7)
        if self.end_with_relu:
            assert self.args.method != "vae", "can't end with relu and use vae!"
            out = F.relu(out)
        if fmaps:
            return {
                'f5': f5.permute(0, 2, 3, 1),
                'f7': f7.permute(0, 2, 3, 1),
                'out': out
            }
        return out



class PPOEncoder(nn.Module):
    def __init__(self, env_name, checkpoint_index):
        super().__init__()
        checkpoint_step = checkpointed_steps_full_sorted[checkpoint_index]
        filepath = download_run(env_name, checkpoint_step)
        while not os.path.exists(filepath):
            time.sleep(5)

        self.masks = torch.zeros(1, 1)
        self.ppo_model, ob_rms = torch.load(filepath, map_location=lambda storage, loc: storage)

    def forward(self, x):
        _, _, _, _, feature_vectors, _ = self.ppo_model.act(x,
                                                            None,
                                                            self.masks,
                                                            deterministic=False)
        return feature_vectors


class LinearRepEncoder(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = nn.Linear(input_size, output_size)
    self.model.to(self.device)
    self.input_size = input_size
    self.feature_size = output_size
    self.hidden_size = self.feature_size

  def forward(self, inputs):
      x = self.model(inputs)
      return x

class MLPRepEncoder(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    #self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.input_size = input_size
    self.feature_size = output_size
    #self.hidden_size = self.feature_size

    self.layer1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.layer2 = nn.Linear(hidden_size, output_size)
    self.sigmoid = nn.Sigmoid()

  def forward(self, inputs):
      x = self.layer1(inputs)
      x = self.relu(x)
      x = self.layer2(x)
      return self.sigmoid(x)

class LinearGRURepEncoder(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, gru_layers=2):
    super().__init__()
    #self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.input_size = input_size
    self.feature_size = output_size
    #self.hidden_size = self.feature_size

    self.linear = nn.Linear(input_size, hidden_size)
    self.gru = nn.GRU(input_size=hidden_size, hidden_size=output_size, num_layers=gru_layers, batch_first=True)

  def forward(self, inputs):
      x = self.linear(inputs)
      x = self.gru(x)
      return x

class MLPRepEncoder2(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    #self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.input_size = input_size
    self.feature_size = output_size
    #self.hidden_size = self.feature_size

    self.layer1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.layer2 = nn.Linear(hidden_size, output_size)
    self.sigmoid = nn.Sigmoid()

  def forward(self, inputs):
      x = self.layer1(inputs)
      x = self.relu(x)
      x = self.layer2(x)
      return x


class ClipEncoder(nn.Module):
  def __init__(self, input_channels, feature_size):
    super().__init__()
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.clip_model, _ = clip.load("ViT-B/32", device=self.device, jit=False)
    self.preprocess = Compose([
        Resize((224, 224), interpolation=Image.BICUBIC),
        Normalize(
          (0.48145466, 0.4578275, 0.40821073),
          (0.26862954, 0.26130258, 0.27577711)
        )
    ])
    self.feature_size = feature_size
    self.input_channels = input_channels

  def forward(self, inputs):
      x = self.get_clip_features(inputs)
      x = x.view(x.size(0), -1)
      return x
  
  def get_clip_features(self, image):
    with torch.no_grad():
      image_features = self.clip_model.encode_image(self.preprocess(image)).float()
    return image_features

class MLPCPCEncoder(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    #self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.input_size = input_size
    self.feature_size = output_size
    self.hidden_size = self.feature_size

    self.layer1 = nn.Linear(input_size, hidden_size)
    self.relu1 = nn.ReLU()
    self.layer2 = nn.Linear(hidden_size, output_size)
    self.relu2 = nn.ReLU()

  def forward(self, inputs):
      x = self.layer1(inputs)
      x = self.relu1(x)
      x = self.layer2(x)
      return self.relu2(x)

class LinearCPCEncoder(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.input_size = input_size
    self.feature_size = output_size
    self.hidden_size = self.feature_size

    self.model = nn.Linear(input_size, output_size)

  def forward(self, inputs):
      return self.model(inputs)

class LinearSTDIMEncoder(nn.Module):
  def __init__(self, input_size, output_size, n_patches, full_img_emb_size=512, emb_size=512, log=False):
    super().__init__()
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = nn.Linear(input_size, output_size)
    self.model.to(self.device)
    self.input_size = input_size
    self.hidden_size = full_img_emb_size
    self.patch_emb_size = n_patches * emb_size
    self.full_img_emb_size = full_img_emb_size
    self.feature_size = output_size
    self.log = log

  def forward(self, inputs, fmaps=False):
      x = self.model(inputs)
      if fmaps:
        fmaps_out = {
            'patch': x[:, self.full_img_emb_size:],
            'full': x[:, :self.full_img_emb_size]
        }

        if self.log:
            print(fmaps_out)

        return fmaps_out
          
      return x

class MLPSTDIMEncoder(nn.Module):
  def __init__(self, input_size, output_size, n_patches, full_img_emb_size=512, emb_size=512, log=False):
    super().__init__()
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.input_size = input_size
    self.hidden_size = full_img_emb_size
    self.patch_emb_size = n_patches * emb_size
    self.full_img_emb_size = full_img_emb_size
    self.feature_size = output_size
    self.log = log

    self.layer1 = nn.Linear(input_size, self.full_img_emb_size + self.patch_emb_size)
    self.relu1 = nn.ReLU()
    self.layer2 = nn.Linear(self.full_img_emb_size + self.patch_emb_size, output_size)
    self.relu2 = nn.ReLU()

  def forward(self, inputs, fmaps=False):
      x = self.layer1(inputs)
      x = self.relu1(x)
      x = self.layer2(x)
      x = self.relu2(x)
      
      if fmaps:
        fmaps_out = {
            'patch': x[:, self.full_img_emb_size:],
            'full': x[:, :self.full_img_emb_size]
        }

        if self.log:
            print(fmaps_out)

        return fmaps_out
          
      return x