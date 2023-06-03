# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tools import init
from graph_encoder import GraphAttentionEncoder
from pointnet import ResnetPointnet
from dataProcess import *

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
  def __init__(self, args, action_space, shapeArray):
    super(DQN, self).__init__()

    self.atoms = args.atoms # c51
    self.action_space = action_space
    if shapeArray is not None:
        self.shapeArray = shapeArray
    self.packed_holder = args.packed_holder
    self.boundingBoxVec = args.boundingBoxVec
    self.heightMap = args.heightMap
    self.rotNum = args.DownRotNum * args.ZRotNum if args.enable_rotation else 1
    self.heightMapSize = int(action_space / self.rotNum)
    self.MapLength = int(args.bin_dimension[0] / args.resolutionH)
    self.ActLength = int(args.bin_dimension[0] / args.resolutionA)
    self.physics = False if args.envName != 'Physics-v0' else True
    self.lineAction = args.actionType == 'LineAction'
    self.rotAction = args.actionType == 'RotAction'
    self.heuAction = args.actionType == 'HeuAction'
    self.elementWise = args.elementWise

    self.trianglePre = args.shapePreType == 'Triangle'
    self.indexPresentation = args.shapePreType == 'Index'
    # self.preEncoder = args.shapePreType == 'PreTrain'
    self.SurfacePointsPre = args.shapePreType == 'SurfacePoints'

    self.args = args
    zDim = 256
    self.zDim = zDim
    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('leaky_relu'))
    self.output_size = zDim * 2
    self.basicFeatureSize = int(zDim / 2)
    self.dataStorage = torch.zeros(64, self.packed_holder, 256, int(self.zDim / 2)).to(self.args.device)

    if self.heightMap and not self.physics:
        self.heightEncoder = nn.Sequential(
            init_(nn.Conv2d(1, 16, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            init_(nn.Conv2d(16, 32, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            init_(nn.Conv2d(32, 16, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            init_(nn.Conv2d(16, 1, 3, stride=1, padding=1)),
            nn.LeakyReLU())


        self.shapeEncoder = nn.Sequential()
        self.shapeEncoder.add_module('linear1', init_(nn.Linear(3, 128)))
        self.shapeEncoder.add_module('relu2', nn.LeakyReLU())
        self.shapeEncoder.add_module('linear3', init_(nn.Linear(128, int(zDim/2))))
        self.shapeEncoder.add_module('relu3', nn.LeakyReLU())

        self.simplify = nn.Sequential()
        self.simplify.add_module('linear1', init_(nn.Linear(zDim, int(zDim/2))))
        self.simplify.add_module('relu1', nn.LeakyReLU())

    else:
        if not self.boundingBoxVec:
            # # Encode a mesh into mesh feature.

            if self.args.shapePreType == 'SurfacePointsEncode':
                self.shapeEncoder = ResnetPointnet(c_dim=128)
                state_dict = torch.load(args.encoderPath)
                self.shapeEncoder.load_state_dict(state_dict)
                for param in self.shapeEncoder.parameters():
                    param.requires_grad = False
            else:
                print('run here')
                self.shapeEncoder = nn.Sequential()
                self.shapeEncoder.add_module('linear1', init_(nn.Linear(3, 128)))
                self.shapeEncoder.add_module('relu2', nn.LeakyReLU())
                self.shapeEncoder.add_module('linear3', init_(nn.Linear(128, int(zDim/2))))
                self.shapeEncoder.add_module('relu3', nn.LeakyReLU())

            transLen = 7 if self.physics else 16
            # Encode the mesh translation.
            if self.elementWise:
                self.transEncoder = nn.Sequential()
                self.transEncoder.add_module('linear1', init_(nn.Linear(transLen, 128)))
                self.transEncoder.add_module('relu2', nn.LeakyReLU())
                self.transEncoder.add_module('linear3', init_(nn.Linear(128, int(zDim / 2))))
                self.transEncoder.add_module('relu3', nn.LeakyReLU())

            if self.heightMap:

                if args.resolutionH == 0.01:
                    self.heightEncoder = nn.Sequential()
                    if np.isclose(args.bin_dimension[0], 0.4):
                        self.heightEncoder.add_module('conv1', init_(nn.Conv2d(1, 16, 4, stride=2, padding=1))) # 40 -> 20
                        self.heightEncoder.add_module('relu1', nn.LeakyReLU())
                        self.heightEncoder.add_module('conv2', init_(nn.Conv2d(16, 32, 6, stride=2))) # 20 -> 8
                    else:
                        self.heightEncoder.add_module('conv1', init_(nn.Conv2d(1, 16, 4, stride=2, padding=1))) # 32 -> 16
                        self.heightEncoder.add_module('relu1', nn.LeakyReLU())
                        self.heightEncoder.add_module('conv2', init_(nn.Conv2d(16, 32, 4, stride=2, padding=1))) # 16 -> 8
                    self.heightEncoder.add_module('relu2', nn.LeakyReLU())
                    if self.elementWise:
                        self.heightEncoder.add_module('conv3', init_(nn.Conv2d(32, 2, 3, stride=1, padding=1))) # 16 -> 8
                    else:
                        self.heightEncoder.add_module('conv3', init_(nn.Conv2d(32, 4, 3, stride=1, padding=1))) # 16 -> 8
                    self.heightEncoder.add_module('relu3', nn.LeakyReLU())

                elif args.resolutionH == 0.005:
                    self.heightEncoder = nn.Sequential(
                        init_(nn.Conv2d(1, 16, 4, stride=2, padding=2)),  # 64 -> 32
                        nn.LeakyReLU(),
                        init_(nn.Conv2d(16, 32, 4, stride=2, padding=1)), # 32 -> 16
                        nn.LeakyReLU(),
                        init_(nn.Conv2d(32, 1, 3, stride=1, padding=1)),  # 16 -> 16
                        nn.LeakyReLU())
                elif args.resolutionH == 0.002:
                    self.heightEncoder = nn.Sequential(
                        init_(nn.Conv2d(1, 16, 9, stride=5, padding=2)),  # 160 -> 32
                        nn.LeakyReLU(),
                        init_(nn.Conv2d(16, 32, 4, stride=2, padding=1)),  # 32 -> 16
                        nn.LeakyReLU(),
                        init_(nn.Conv2d(32, 1, 3, stride=1, padding=1)),  # 16 -> 16
                        nn.LeakyReLU())

                if self.elementWise:
                    self.shorter = nn.Sequential()
                    self.shorter.add_module('linear1', init_(nn.Linear(zDim, int(zDim / 2))))
                    self.shorter.add_module('relu1', nn.LeakyReLU())

        else:
            self.shapeEncoder = nn.Sequential()
            self.shapeEncoder.add_module('linear1', init_(nn.Linear(3, 128)))
            # self.shapeEncoder.add_module('linear1', init_(nn.Linear(6, 128)))
            self.shapeEncoder.add_module('relu1', nn.LeakyReLU())
            self.shapeEncoder.add_module('linear2', init_(nn.Linear(128, int(zDim/2) )))
            # self.shapeEncoder.add_module('linear2', init_(nn.Linear(128, zDim )))
            self.shapeEncoder.add_module('relu2', nn.LeakyReLU())
            #
            self.onemoreEncoder = nn.Sequential()
            self.onemoreEncoder.add_module('linear1', init_(nn.Linear(zDim, zDim)))
            self.onemoreEncoder.add_module('relu1', nn.LeakyReLU())

            self.nextshapeEncoder = nn.Sequential()
            self.nextshapeEncoder.add_module('linear1', init_(nn.Linear(6, 128)))
            self.nextshapeEncoder.add_module('relu1', nn.LeakyReLU())
            self.nextshapeEncoder.add_module('linear2', init_(nn.Linear(128, int(zDim / 2))))
            self.nextshapeEncoder.add_module('relu2', nn.LeakyReLU())

            self.transEncoder = nn.Sequential()
            self.transEncoder.add_module('linear1', init_(nn.Linear(16, 128)))
            self.transEncoder.add_module('relu2', nn.LeakyReLU())
            self.transEncoder.add_module('linear3', init_(nn.Linear(128, int(zDim / 2))))
            self.transEncoder.add_module('relu3', nn.LeakyReLU())

        # # Graph attention model
        if self.elementWise:
            n_heads = 1
            n_layers = 1
            graph_size = args.packed_holder
            # Use graph attention network to encode the packed items.
            self.gatherShape = GraphAttentionEncoder(
              n_heads=n_heads,
              embed_dim=zDim,
              n_layers=n_layers,
              graph_size=graph_size,
              feed_forward_hidden=256)

        # Encode the object mask
        if self.lineAction or self.rotAction or self.heuAction:
            self.maskEncoder = nn.Sequential()
            self.maskEncoder.add_module('linear1', init_(nn.Linear(self.action_space, zDim)))
            self.maskEncoder.add_module('relu1', nn.LeakyReLU())
            self.maskEncoder.add_module('linear2', init_(nn.Linear(zDim, int(zDim / 2))))
            self.maskEncoder.add_module('relu2', nn.LeakyReLU())
        else:
            if args.resolutionA == 0.04:
                self.maskEncoder = nn.Sequential(
                    init_(nn.Conv2d(self.rotNum,  32,  3, stride=1, padding=1)), # 8 -> 8
                    nn.LeakyReLU(),
                    init_(nn.Conv2d(32, 32, 3, stride=1, padding=1)),  # 8 -> 8
                    nn.LeakyReLU(),
                    init_(nn.Conv2d(32, 2,  3, stride=1, padding=1)),
                    nn.LeakyReLU())
            elif args.resolutionA == 0.02:
                self.maskEncoder = nn.Sequential(
                    init_(nn.Conv2d(self.rotNum,  32,  3, stride=1, padding=1)), # 16 -> 16
                    nn.LeakyReLU(),
                    init_(nn.Conv2d(32, 32, 4, stride=2, padding=1)),  # 16 -> 8
                    nn.LeakyReLU(),
                    init_(nn.Conv2d(32, 2,  3, stride=1, padding=1)),
                    nn.LeakyReLU())
            elif  args.resolutionA == 0.01:
                self.maskEncoder = nn.Sequential(
                    init_(nn.Conv2d(self.rotNum,  32,  4, stride=2, padding=1)), # 32 -> 16
                    nn.LeakyReLU(),
                    init_(nn.Conv2d(32, 32, 4, stride=2, padding=1)),  # 16 -> 8
                    nn.LeakyReLU(),
                    init_(nn.Conv2d(32, 2,  3, stride=1, padding=1)),
                    nn.LeakyReLU())

    self.fc_h_v = NoisyLinear(self.output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def decode_mesh(self, observation):
      batchSize = observation.shape[0]
      observation = observation.reshape((batchSize, -1))
      packed_items = observation[:, 0:self.packed_holder * 20].reshape((batchSize, -1, 20))
      next_item = observation[:, self.packed_holder * 20:(self.packed_holder + 1) * 20].reshape((batchSize, 20))
      masks = observation[:, (self.packed_holder + 1) * 20:]
      return packed_items, next_item, masks

  def decode_element(self, observation):
      batchSize = observation.shape[0]
      observation = observation.reshape((batchSize, -1))
      packed_items = observation[:, 0:self.packed_holder * 9].reshape((batchSize, -1, 9))
      packed_items = packed_items[:,:,0:6]
      next_item = observation[:, self.packed_holder * 9:(self.packed_holder + 1) * 9].reshape((batchSize, 9))
      next_item = next_item[:, 0:6]
      masks = observation[:, (self.packed_holder + 1) * 9:]
      return packed_items, next_item, masks

  def decode_element_in_mesh_style(self, observation):

      batchSize   = observation.shape[0]
      observation = observation.reshape((batchSize, -1))
      packed_items = observation[:, 0:self.packed_holder * 9].reshape((batchSize, -1, 9))

      packed_items_dimension = packed_items[:,:,0:3]
      translation = packed_items[:,:,3:6]
      packed_items_translation = torch.zeros((batchSize, packed_items.shape[1], 16)).to(self.args.device)
      packed_items_translation[:,:,0] = 1
      packed_items_translation[:,:,5] = 1
      packed_items_translation[:,:,10] = 1
      packed_items_translation[:,:,15] = 1
      packed_items_translation[:,:,3] = translation[:,:,0]
      packed_items_translation[:,:,7] = translation[:,:,1]
      packed_items_translation[:,:,11] = translation[:,:,2]

      next_item = observation[:, self.packed_holder * 9:(self.packed_holder + 1) * 9].reshape((batchSize, 9))
      next_item = next_item[:, 3:6]
      masks = observation[:, (self.packed_holder + 1) * 9:]

      return packed_items_dimension, packed_items_translation, next_item, masks

  def decode_heightMap(self, observation):
      batchSize = observation.shape[0]

      observation = observation.reshape((batchSize, -1))
      masks = observation[:, 0:self.action_space].reshape((batchSize, self.action_space))
      next_item = observation[:, self.action_space + self.heightMapSize : ].reshape((batchSize, 20))
      heightMap = observation[:, self.action_space : self.action_space + self.heightMapSize]

      return heightMap, next_item, masks

  def embed_physic_mesh_with_heightmap(self, x):
      batchSize = x.shape[0]
      pointSize = self.shapeArray.shape[2]

      packed_items, next_item, masks, heightMap = self.decode_physics_mesh(x)
      objVecLen, transLen = 9, 7
      heightMap = heightMap.reshape((batchSize, 1, self.MapLength, self.MapLength))

      if self.lineAction or self.rotAction or self.heuAction:
        masks = masks.reshape((batchSize, -1))
      else:
        masks = masks.reshape((batchSize, self.rotNum, self.ActLength, self.ActLength))

      packed_items = packed_items.reshape(batchSize, -1, objVecLen)
      packed_items_ID = packed_items[:, :, 0].long()
      # packed_items_rotIdx = packed_items[:, :, -2].long()
      packed_items_trans = packed_items[:, :, 1: 1 + transLen].reshape(batchSize, self.packed_holder, -1)
      packed_items_mask = 1 - packed_items[:, :, -1]
      packed_items_ID[packed_items_mask == 1] = -1
      packed_items_length = torch.sum(1 - packed_items_mask, dim=1)
      packed_items_length[packed_items_length == 0] = 1

      next_item_ID = next_item[:, 0].long()
      # inputShape = self.shapeArray[packed_items_ID, packed_items_rotIdx]
      inputShape = self.shapeArray[packed_items_ID, 0].to(self.args.device)
      inputShape = inputShape.reshape(batchSize, self.packed_holder, pointSize, -1)
      nextShape = self.shapeArray[next_item_ID, 0].to(self.args.device)

      # bug exsits
      array1 = self.shapeEncoder(inputShape)
      # array1 = self.pointTransformer(array1.view(-1, pointSize, self.basicFeatureSize))[0]
      array1 = array1.view(batchSize, -1, pointSize, self.basicFeatureSize)
      array1 = torch.mean(array1, dim=2)
      array2 = self.transEncoder(packed_items_trans)
      packed_items_feature = torch.cat([array1, array2], dim=2)

      packed_items_mask = packed_items_mask.view(batchSize, packed_items_feature.shape[1], 1).expand(
          packed_items_feature.shape).bool()
      packed_items_feature[packed_items_mask] = 0
      packed_items_feature = self.gatherShape(packed_items_feature)[0]
      packed_items_feature[packed_items_mask] = 0
      packed_items_feature = packed_items_feature.sum(1) / packed_items_length.reshape((-1, 1))
      packed_items_feature = self.shorter(packed_items_feature)

      map_feature = self.heightEncoder(heightMap).reshape((batchSize, -1))
      shape_feature = self.shapeEncoder(nextShape)
      shape_feature = torch.max(shape_feature, dim=1)[0]
      mask_feature = self.maskEncoder(masks).reshape((batchSize, -1))

      x = torch.cat([packed_items_feature, map_feature, shape_feature, mask_feature], dim=1)

      return x

  def decode_physic_only_with_heightmap(self, observation):
      batchSize = observation.shape[0]
      observation = observation.reshape((batchSize, -1))
      next_item = observation[:, 0 : 9].reshape((batchSize, 9))
      masks = observation[:, 9: 9 + self.action_space]
      heightMap = observation[:, 9 + self.action_space : ] if self.heightMap else None
      return next_item, masks,  heightMap

  def embed_physic_only_with_heightmap(self, x):
      batchSize = x.shape[0]

      next_item, masks, heightMap = self.decode_physic_only_with_heightmap(x)
      heightMap = heightMap.reshape((batchSize, 1, self.MapLength, self.MapLength))
      if self.lineAction or self.rotAction or self.heuAction:
        masks = masks.reshape((batchSize, -1))
      else:
        masks = masks.reshape((batchSize, self.rotNum, self.ActLength, self.ActLength))

      next_item_ID = next_item[:, 0].long()
      if self.args.shapePreType == 'SurfacePointsRandom' or self.args.shapePreType == 'SurfacePointsEncode':
          nextShape = self.shapeArray[next_item_ID]
          indices = np.random.randint(self.shapeArray.shape[1], size=self.args.samplePointsNum)  # 这里是不是最好是不重复的元素啊
          nextShape = nextShape[:, indices].to(self.args.device)
      else:
          nextShape = self.shapeArray[next_item_ID, 0].to(self.args.device)

      map_feature = self.heightEncoder(heightMap).reshape((batchSize, -1))

      assert not self.preEncoder
      if  self.args.shapePreType == 'SurfacePointsEncode':
          self.shapeEncoder.eval()
          shape_feature = self.shapeEncoder(nextShape)
          shape_feature = F.normalize(shape_feature, dim=1)
      else:
          shape_feature = self.shapeEncoder(nextShape)
          shape_feature = torch.max(shape_feature, dim=1)[0]

      mask_feature = self.maskEncoder(masks).reshape((batchSize, -1))

      x = torch.cat([map_feature, shape_feature, mask_feature], dim=1)

      return x

  def embed_physic_only_with_heightmap_index(self, x):
      batchSize = x.shape[0]

      next_item, masks, heightMap = self.decode_physic_only_with_heightmap(x)
      heightMap = heightMap.reshape((batchSize, 1, self.MapLength, self.MapLength))
      if self.lineAction or self.rotAction or self.heuAction:
        masks = masks.reshape((batchSize, -1))
      else:
        masks = masks.reshape((batchSize, self.rotNum, self.ActLength, self.ActLength))

      next_item_ID = next_item[:, 0].float()
      # nextShape = self.shapeArray[next_item_ID, 0]
      next_item_ID = next_item_ID.reshape((-1, 1)).repeat((1,128))
      map_feature = self.heightEncoder(heightMap).reshape((batchSize, -1))
      shape_feature = self.shapeEncoder(next_item_ID)
      mask_feature = self.maskEncoder(masks).reshape((batchSize, -1))

      x = torch.cat([map_feature, shape_feature, mask_feature], dim=1)

      return x

  def embed_physic_only_with_heightmap_pretrained(self, x):
      batchSize = x.shape[0]

      next_item, masks, heightMap = self.decode_physic_only_with_heightmap(x)
      heightMap = heightMap.reshape((batchSize, 1, self.MapLength, self.MapLength))
      if self.lineAction or self.rotAction or self.heuAction:
        masks = masks.reshape((batchSize, -1))
      else:
        masks = masks.reshape((batchSize, self.rotNum, self.ActLength, self.ActLength))

      next_item_ID = next_item[:, 0].long()
      nextShape = self.shapeArray[next_item_ID, 0].to(self.args.device)

      map_feature = self.heightEncoder(heightMap).reshape((batchSize, -1))

      # with torch.no_grad:
      shape_feature = self.shapeEncoder(nextShape)

      # shape_feature = self.shapeEncoderHead(shape_feature)
      mask_feature = self.maskEncoder(masks).reshape((batchSize, -1))
      x = torch.cat([map_feature, shape_feature, mask_feature], dim=1)

      return x

  def embed_as_mesh(self, x):
      batchSize = x.shape[0]
      pointSize = self.shapeArray.shape[2]

      if self.physics:
          packed_items, next_item, masks, _ = self.decode_physics_mesh(x)
      else:
          packed_items, next_item, masks = self.decode_mesh(x)

      if self.physics:
          objVecLen, transLen = 9, 7
      else:
          objVecLen, transLen = 20, 16

      packed_items = packed_items.reshape(batchSize, -1, objVecLen)
      packed_items_ID = packed_items[:, :, 0].long()
      # packed_items_rotIdx = packed_items[:, :, -2].long()
      packed_items_trans = packed_items[:, :, 1: 1 + transLen].reshape(batchSize, self.packed_holder, -1)
      packed_items_mask = 1 - packed_items[:, :, -1]
      packed_items_ID[packed_items_mask == 1] = -1
      packed_items_length = torch.sum(1 - packed_items_mask, dim=1)
      packed_items_length[packed_items_length==0] = 1

      next_item_ID = next_item[:,0].long()
      # inputShape = self.shapeArray[packed_items_ID, packed_items_rotIdx]
      inputShape = self.shapeArray[packed_items_ID, 0]
      inputShape = inputShape.reshape(batchSize, self.packed_holder,  pointSize, -1)
      nextShape  = self.shapeArray[next_item_ID, 0].to(self.args.device)

      # bug exsits
      array1 = self.shapeEncoder(inputShape)
      # array1 = self.pointTransformer(array1.view(-1, pointSize, self.basicFeatureSize))[0]
      array1 = array1.view(batchSize, -1, pointSize, self.basicFeatureSize)
      array1 = torch.mean(array1, dim=2)
      array2 = self.transEncoder(packed_items_trans)
      packed_items_feature = torch.cat([array1, array2], dim = 2)

      packed_items_mask = packed_items_mask.view(batchSize, packed_items_feature.shape[1], 1).expand(packed_items_feature.shape).bool()
      packed_items_feature[packed_items_mask] = 0
      packed_items_feature = self.gatherShape(packed_items_feature)[0]
      packed_items_feature[packed_items_mask] = 0
      packed_items_feature = packed_items_feature.sum(1) / packed_items_length.reshape((-1, 1))

      array3 = self.shapeEncoder(nextShape)
      array3 = torch.max(array3, dim=1)[0]
      array4 = self.maskEncoder(masks)

      x = torch.cat([packed_items_feature, array3, array4], dim=1)

      return x

  def embed_as_mesh_efficiently(self, x):
      batchSize = x.shape[0]
      pointSize = self.shapeArray.shape[2]
      packed_items, next_item, masks = self.decode_mesh(x)

      packed_items = packed_items.reshape(batchSize, -1, 20)
      packed_items_ID = packed_items[:, :, 0].long()

      packed_items_trans = packed_items[:, :, 1:17].reshape(batchSize, self.packed_holder, -1)
      packed_items_mask = packed_items[:, :, -1]
      packed_index = torch.where(packed_items_mask!=0)

      packed_items_length = torch.sum(packed_items_mask, dim=1)

      next_item_ID = next_item[:,0].long()
      inputShape = self.shapeArray[packed_items_ID, 0]
      inputShape = inputShape.reshape(batchSize, self.packed_holder,  pointSize, -1)
      inputShape = inputShape[packed_index].to(self.args.device)

      array1 = self.shapeEncoder(inputShape)
      shapeFeatures = array1.split(packed_items_length.type(torch.int).cpu().numpy().tolist())

      dataStorage = self.dataStorage.clone()
      for batchI in range(batchSize):
          if packed_items_length[batchI] != 0:
              dataStorage[batchI][0:int(packed_items_length[batchI])] = shapeFeatures[batchI]
      dataStorage = dataStorage.mean(dim = 2)[0:batchSize]

      nextShape  = self.shapeArray[next_item_ID, 0].to(self.args.device)
      array2 = self.transEncoder(packed_items_trans)
      packed_items_feature = torch.cat([dataStorage, array2], dim = 2)
      mask_indicator = (1 - packed_items_mask).view(batchSize, packed_items_feature.shape[1], 1).expand(packed_items_feature.shape).bool()
      packed_items_feature[mask_indicator] = 0

      packed_items_feature = self.gatherShape(packed_items_feature)[0]

      packed_items_feature[mask_indicator] = 0
      packed_items_length[packed_items_length == 0] = 1
      packed_items_feature = packed_items_feature.sum(1) / packed_items_length.reshape((-1, 1))

      array3 = self.shapeEncoder(nextShape)
      array3 = torch.max(array3, dim=1)[0]
      array4 = self.maskEncoder(masks)

      x = torch.cat([packed_items_feature, array3, array4], dim=1)
      return x


  def embed_as_element(self, x):

      batchSize = x.shape[0]
      packed_items, next_item, masks = self.decode_element(x)

      packed_items = packed_items.reshape(batchSize, -1, 6)
      packed_items_mask = 1 - packed_items[:, :, -1]
      packed_items_length = torch.sum(1 - packed_items_mask, dim=1)
      packed_items_length[packed_items_length==0] = 1

      # bug exsits
      packed_items_feature = self.shapeEncoder(packed_items)
      packed_items_feature = self.gatherShape(packed_items_feature)[0]
      packed_items_mask = packed_items_mask.view(batchSize, packed_items_feature.shape[1], 1).expand(packed_items_feature.shape).bool()

      packed_items_feature[packed_items_mask] = 0
      packed_items_feature = packed_items_feature.sum(1) / packed_items_length.reshape((-1, 1))

      nextItemFeature = self.nextshapeEncoder(next_item)
      maskFeature = self.maskEncoder(masks)

      x = torch.cat([packed_items_feature, nextItemFeature, maskFeature], dim=1)

      return x

  def embed_element_as_mesh_style(self, x):

      batchSize = x.shape[0]
      packed_items_dimension, packed_items_translation, next_item, masks = self.decode_element_in_mesh_style(x)

      packed_items = packed_items_dimension.reshape(batchSize, -1, 3)
      packed_items_mask = 1 - packed_items[:, :, -1]
      packed_items_length = torch.sum(1 - packed_items_mask, dim=1)
      packed_items_length[packed_items_length==0] = 1

      # bug exsits
      array1 = self.shapeEncoder(packed_items)
      array2 = self.transEncoder(packed_items_translation)
      packed_items_feature = torch.cat([array1, array2], dim=2)
      packed_items_feature = self.onemoreEncoder(packed_items_feature)
      packed_items_feature = self.gatherShape(packed_items_feature)[0]
      packed_items_mask = packed_items_mask.view(batchSize, packed_items_feature.shape[1], 1).expand(packed_items_feature.shape).bool()

      packed_items_feature[packed_items_mask] = 0
      packed_items_feature = packed_items_feature.sum(1) / packed_items_length.reshape((-1, 1))

      nextItemFeature = self.shapeEncoder(next_item)
      maskFeature = self.maskEncoder(masks)

      x = torch.cat([packed_items_feature, nextItemFeature, maskFeature], dim=1)

      return x

  def decode_physics_mesh(self, observation):
      batchSize = observation.shape[0]
      observation = observation.reshape((batchSize, -1))
      packed_items = observation[:, 0:self.packed_holder * 9].reshape((batchSize, -1, 9))
      next_item = observation[:, self.packed_holder * 9:(self.packed_holder + 1) * 9].reshape((batchSize, 9))
      masks = observation[:, (self.packed_holder + 1) * 9: (self.packed_holder + 1) * 9 + self.action_space]
      heightMap = observation[:, (self.packed_holder + 1) * 9 + self.action_space : ] if self.heightMap else None
      return packed_items, next_item, masks,  heightMap

  def embed_mesh_and_heightmap(self, x):
      batchSize = x.shape[0]
      heightMap, next_item, masks = self.decode_heightMap(x)

      heightMap = heightMap.reshape(batchSize, 1, self.MapLength, self.MapLength)
      # masks     = masks.reshape(batchSize, -1, self.MapLength, self.MapLength)
      next_item_ID = next_item[:, 0].long()
      nextShape = self.shapeArray[next_item_ID, 0].to(self.args.device)
      packed_items_feature = self.heightEncoder(heightMap).reshape(batchSize, -1)
      array3 = self.shapeEncoder(nextShape)
      array3 = torch.max(array3, dim=1)[0]
      # array4 = self.maskEncoder(masks).reshape(batchSize, -1)
      # array4 = self.simplify(array4)
      # x = torch.cat([packed_items_feature, array3, array4], dim=1)
      x = torch.cat([packed_items_feature, array3, array3], dim=1)
      return x


  def forward(self, x, log=False, getCL = False):
      loss_cl = None
      if not self.heightMap:
          if self.boundingBoxVec:
              x = self.embed_as_element(x)
          else:
              x = self.embed_as_mesh(x)
      else:
          if not self.physics:
            x = self.embed_mesh_and_heightmap(x)
          else:
              if self.elementWise:
                x = self.embed_physic_mesh_with_heightmap(x)
              else:
                x = self.embed_physic_only_with_heightmap(x)

      x = x.view(-1, self.output_size)

      v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
      a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
      v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
      q = v + a - a.mean(1, keepdim=True)  # Combine streams
      if log:  # Use log softmax for numerical stability
        q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
      else:
        q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
      # return q
      if getCL:
          return q, loss_cl
      else:
          return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()

class DQNP(nn.Module):
  def __init__(self, args, action_space, shapeArray):
    super(DQNP, self).__init__()
    assert args.selectedAction
    self.args = args
    self.atoms = args.atoms # c51
    self.action_space = action_space
    assert shapeArray is not None
    self.orginArray = shapeArray
    self.arrayPointNum  = 10000
    self.shapeArray = shapeArray
    if self.args.shapePreType == 'SurfacePointsRandom' or self.args.shapePreType == 'SurfacePointsEncode':
        if self.args.level == 'order':
            self.shapeArray = torch.zeros((shapeArray.shape[0], self.arrayPointNum, shapeArray.shape[2]))
    self.updateShapeArray()

    self.packed_holder = args.packed_holder
    self.boundingBoxVec = args.boundingBoxVec
    self.heightMap = args.heightMap
    self.rotNum = args.DownRotNum * args.ZRotNum if args.enable_rotation else 1
    self.heightMapSize = int(action_space / self.rotNum)
    self.MapLength = int(args.bin_dimension[0] / args.resolutionH)
    self.ActLength = int(args.bin_dimension[0] / args.resolutionA)
    self.physics = False if args.envName != 'Physics-v0' else True
    self.elementWise = args.elementWise

    self.trianglePre = args.shapePreType == 'Triangle'
    self.indexPresentation = args.shapePreType == 'Index'
    self.preEncoder = args.shapePreType == 'PreTrain'
    self.SurfacePointsPre = args.shapePreType == 'SurfacePoints'


    zDim = 256
    self.zDim = zDim
    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('leaky_relu'))
    self.output_size = zDim * 2

    # Network components
    if args.resolutionH == 0.01:
        self.heightEncoder = nn.Sequential()
        if np.isclose(args.bin_dimension[0], 0.4):
            self.heightEncoder.add_module('conv1', init_(nn.Conv2d(1, 16, 4, stride=2, padding=1)))  # 40 -> 20
            self.heightEncoder.add_module('relu1', nn.LeakyReLU())
            self.heightEncoder.add_module('conv2', init_(nn.Conv2d(16, 32, 6, stride=2)))  # 20 -> 8
        else:
            self.heightEncoder.add_module('conv1', init_(nn.Conv2d(1, 16, 4, stride=2, padding=1)))  # 32 -> 16
            self.heightEncoder.add_module('relu1', nn.LeakyReLU())
            self.heightEncoder.add_module('conv2', init_(nn.Conv2d(16, 32, 4, stride=2, padding=1)))  # 16 -> 8
        self.heightEncoder.add_module('relu2', nn.LeakyReLU())
        self.heightEncoder.add_module('conv3', init_(nn.Conv2d(32, 4, 3, stride=1, padding=1)))  # 16 -> 8
        self.heightEncoder.add_module('relu3', nn.LeakyReLU())
    elif args.resolutionH == 0.005:
        self.heightEncoder = nn.Sequential(
            init_(nn.Conv2d(1, 16, 4, stride=2, padding=2)),  # 64 -> 32
            nn.LeakyReLU(),
            init_(nn.Conv2d(16, 32, 4, stride=2, padding=1)),  # 32 -> 16
            nn.LeakyReLU(),
            init_(nn.Conv2d(32, 1, 3, stride=1, padding=1)),  # 16 -> 16
            nn.LeakyReLU())
    elif args.resolutionH == 0.002:
        self.heightEncoder = nn.Sequential(
            init_(nn.Conv2d(1, 16, 9, stride=5, padding=2)),  # 160 -> 32
            nn.LeakyReLU(),
            init_(nn.Conv2d(16, 32, 4, stride=2, padding=1)),  # 32 -> 16
            nn.LeakyReLU(),
            init_(nn.Conv2d(32, 1, 3, stride=1, padding=1)),  # 16 -> 16
            nn.LeakyReLU())

    if self.preEncoder:
        self.shapeEncoder = nn.Sequential()
        self.shapeEncoder.add_module('linear1', init_(nn.Linear(512, 128)))
        self.shapeEncoder.add_module('relu1', nn.LeakyReLU())
    elif self.indexPresentation:
        self.shapeEncoder = nn.Sequential()
        self.shapeEncoder.add_module('linear1', init_(nn.Linear(128, int(zDim / 2))))
        self.shapeEncoder.add_module('relu1', nn.LeakyReLU())
        assert False
    elif self.args.shapePreType == 'SurfacePointsEncode':
        self.shapeEncoder = ResnetPointnet(c_dim=128)
        if args.encoderPath is not None:
            print('Load encoder from', args.encoderPath)
            state_dict = torch.load(args.encoderPath, map_location='cpu')
            self.shapeEncoder.load_state_dict(state_dict)
        for param in self.shapeEncoder.parameters():
            param.requires_grad = False
        self.shapeEncoder.eval()
    else:
        self.shapeEncoder = nn.Sequential()
        if self.trianglePre:
            self.shapeEncoder.add_module('linear1', init_(nn.Linear(12, 128)))
        else:
            self.shapeEncoder.add_module('linear1', init_(nn.Linear(3, 128)))
        self.shapeEncoder.add_module('relu1', nn.LeakyReLU())
        self.shapeEncoder.add_module('linear2', init_(nn.Linear(128, 128)))
        self.shapeEncoder.add_module('relu2', nn.LeakyReLU())

    self.init_ems_embed = nn.Sequential(
        init_(nn.Linear(4, 32)),
        nn.LeakyReLU(),
        init_(nn.Linear(32, 128)))

    self.embedding_dim = 128
    self.oneMore = nn.Sequential(
        init_(nn.Linear(512, 256)),
        # init_(nn.Linear(640, 256)),
        nn.LeakyReLU(),
        init_(nn.Linear(256, self.embedding_dim)))

    self.noAction = nn.Sequential(
        init_(nn.Linear(384, 256)),
        nn.LeakyReLU(),
        init_(nn.Linear(256, self.embedding_dim)))

    if args.previewNum > 1:
    # if True:
        self.embedder = GraphAttentionEncoder(
            n_heads=1,
            embed_dim=self.embedding_dim,
            n_layers=1,
            graph_size=args.previewNum,
        )

    self.fc_h_v = NoisyLinear(self.embedding_dim, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.embedding_dim, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    # This one should be changed.
    # self.fc_z_a = NoisyLinearP(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)

  def updateShapeArray(self):
      if self.args.shapePreType == 'SurfacePointsRandom' or self.args.shapePreType == 'SurfacePointsEncode':
          if self.args.level == 'order':
              indices = np.random.randint(self.shapeArray.shape[1], size=self.arrayPointNum)
              for idx in range(len(self.orginArray)):
                  self.shapeArray[idx] = self.orginArray[idx][indices].clone().detach()
      self.forwardCounter = 0


  def decode_physic_only_with_heightmap(self, observation):
      batchSize = observation.shape[0]
      observation = observation.reshape((batchSize, -1))
      next_item = observation[:, 0 : 9].reshape((batchSize, 9))
      masks = observation[:, 9: 9 + self.action_space]
      heightMap = observation[:, 9 + self.action_space : ] if self.heightMap else None
      return next_item, masks,  heightMap

  def embed_physic_only_with_heightmap(self, x):
      batchSize = x.shape[0]
      next_item, actionMask, heightMap, candidates = observation_decode_irregular(x, self.args)
      graph_size = candidates.size(1)

      valid_mask = actionMask
      invalid_ones = 1 - valid_mask  # mask 为1的地方是被删掉的地方

      candidates_size = candidates.size(1)
      heightMap = heightMap.reshape((batchSize, 1, self.MapLength, self.MapLength))
      map_feature = self.heightEncoder(heightMap).reshape((batchSize, -1))

      next_item_ID = next_item[:, 0].long()
      if self.args.shapePreType == 'SurfacePointsRandom' or self.args.shapePreType == 'SurfacePointsEncode':
          nextShape = self.shapeArray[next_item_ID.cpu()]
          indices = np.random.randint(self.shapeArray.shape[1], size=self.args.samplePointsNum)  # 这里是不是最好是不重复的元素啊
          # indices = self.args.globalIndices
          nextShape = nextShape[:, indices].to(self.args.device)
      else:
          nextShape = self.shapeArray[next_item_ID, 0].to(self.args.device)

      assert not self.preEncoder
      if  self.args.shapePreType == 'SurfacePointsEncode':
          self.shapeEncoder.eval()
          shape_feature = self.shapeEncoder(nextShape)
          shape_feature = F.normalize(shape_feature, dim=1)
      else:
          shape_feature = self.shapeEncoder(nextShape)
          shape_feature = torch.max(shape_feature, dim=1)[0]

      ems_inputs = candidates.contiguous().view(batchSize, candidates_size, -1)
      ems_embedded_inputs = self.init_ems_embed(ems_inputs)
      init_embedding = torch.cat((shape_feature.repeat(1, candidates_size).reshape(batchSize, candidates_size, -1),
                                  map_feature.repeat((1, candidates_size)).reshape(batchSize, candidates_size, -1),
                                  ems_embedded_inputs), dim=2).view(batchSize * candidates_size, -1)
      init_embedding = self.oneMore(init_embedding).view(batchSize, candidates_size, self.embedding_dim)

      embeddings = init_embedding
      embedding_shape = embeddings.shape

      transEmbedding = embeddings.view((batchSize, graph_size, -1))
      invalid_ones = invalid_ones.view(embedding_shape[0], embedding_shape[1], 1).expand(embedding_shape).bool()
      transEmbedding[invalid_ones] = 0
      graph_embed = transEmbedding.view(embedding_shape).mean(1)  # 其实这里的取均值一定程度上相当于相加了

      return embeddings, graph_embed

  def embed_physic_only_with_heightmap_advantage(self, x):
      batchSize = x.shape[0]
      next_item, actionMask, heightMap, candidates = observation_decode_irregular(x, self.args)
      graph_size = candidates.size(1)

      valid_mask = actionMask
      invalid_ones = 1 - valid_mask  # mask 为1的地方是被删掉的地方

      candidates_size = candidates.size(1)
      heightMap = heightMap.reshape((batchSize, 1, self.MapLength, self.MapLength))
      map_feature = self.heightEncoder(heightMap).reshape((batchSize, -1))

      next_item_ID = next_item[:, 0].long()
      if self.args.shapePreType == 'SurfacePointsRandom' or self.args.shapePreType == 'SurfacePointsEncode':
          nextShape = self.shapeArray[next_item_ID]
          indices = np.random.randint(self.shapeArray.shape[1], size=self.args.samplePointsNum)  # 这里是不是最好是不重复的元素啊
          # indices = self.args.globalIndices
          nextShape = nextShape[:, indices].to(self.args.device)
      else:
          nextShape = self.shapeArray[next_item_ID, 0].to(self.args.device)

      assert not self.preEncoder
      if  self.args.shapePreType == 'SurfacePointsEncode':
          self.shapeEncoder.eval()
          shape_feature = self.shapeEncoder(nextShape)
          shape_feature = F.normalize(shape_feature, dim=1)
      else:
          shape_feature = self.shapeEncoder(nextShape)
          shape_feature = torch.max(shape_feature, dim=1)[0]

      ems_inputs = candidates.contiguous().view(batchSize, candidates_size, -1)
      ems_embedded_inputs = self.init_ems_embed(ems_inputs)
      init_embedding = torch.cat((shape_feature.repeat(1, candidates_size).reshape(batchSize, candidates_size, -1),
                                  map_feature.repeat((1, candidates_size)).reshape(batchSize, candidates_size, -1),
                                  ems_embedded_inputs), dim=2).view(batchSize * candidates_size, -1)
      init_embedding = self.oneMore(init_embedding).view(batchSize, candidates_size, self.embedding_dim)

      # embeddings = init_embedding

      embeddings = torch.cat((shape_feature, map_feature), dim=1).view(batchSize, -1)
      graph_embed = self.noAction(embeddings)

      # init_embedding_shape = init_embedding.shape
      # transEmbedding = init_embedding.view((batchSize, graph_size, -1))
      # invalid_ones = invalid_ones.view(init_embedding_shape[0], init_embedding_shape[1], 1).expand(init_embedding_shape).bool()
      # transEmbedding[invalid_ones] = 0
      # graph_embed = transEmbedding.view(init_embedding_shape).mean(1)  # 其实这里的取均值一定程度上相当于相加了

      return init_embedding, graph_embed

  def embed_physic_k_shape(self, x):
      batchSize = x.shape[0]

      next_k_shapes_ID, heightMap = observation_decode_irregular_k_shape(x, self.args)
      graph_size = next_k_shapes_ID.size(1)

      candidates_size = graph_size
      heightMap = heightMap.reshape((batchSize, 1, self.MapLength, self.MapLength))
      map_feature = self.heightEncoder(heightMap).reshape((batchSize, -1))

      shapeIdx = next_k_shapes_ID.detach().long().reshape(-1)

      if self.args.shapePreType == 'SurfacePointsRandom' or self.args.shapePreType == 'SurfacePointsEncode':
          next_k_shapes = self.shapeArray[shapeIdx]
          indices = np.random.randint(self.shapeArray.shape[1], size=self.args.samplePointsNum)  # 这里是不是最好是不重复的元素啊
          next_k_shapes = next_k_shapes[:, indices].to(self.args.device)
      else:
          rotIdx = torch.zeros_like(shapeIdx).long()
          next_k_shapes = self.shapeArray[shapeIdx, rotIdx].float().to(self.args.device)

      shape_feature = self.shapeEncoder(next_k_shapes)
      if not self.preEncoder:
        shape_feature = torch.max(shape_feature, dim=1)[0]

      init_embedding = torch.cat((shape_feature.reshape(batchSize, candidates_size, -1),
                                  map_feature.repeat((1, candidates_size)).reshape(batchSize, candidates_size, -1)), dim=2).view(batchSize * candidates_size, -1)
      init_embedding = self.noAction(init_embedding).view(batchSize, candidates_size, self.embedding_dim)
      embeddings = init_embedding
      embedding_shape = embeddings.shape

      transEmbedding = embeddings.view((batchSize, graph_size, -1))
      graph_embed = transEmbedding.view(embedding_shape).mean(1)  # 其实这里的取均值一定程度上相当于相加了

      return embeddings, graph_embed

  def embed_physic_k_shape_with_gat(self, x):
      batchSize = x.shape[0]

      next_k_shapes_ID, heightMap = observation_decode_irregular_k_shape(x, self.args)
      graph_size = next_k_shapes_ID.size(1)

      candidates_size = graph_size
      heightMap = heightMap.reshape((batchSize, 1, self.MapLength, self.MapLength))
      map_feature = self.heightEncoder(heightMap).reshape((batchSize, -1))

      shapeIdx = next_k_shapes_ID.detach().long().reshape(-1)
      if self.args.shapePreType == 'SurfacePointsRandom' or self.args.shapePreType == 'SurfacePointsEncode':
          next_k_shapes = self.shapeArray[shapeIdx] # 这一步其实很费时间
          indices = np.random.randint(self.shapeArray.shape[1], size=self.args.samplePointsNum)  # 这里是不是最好是不重复的元素啊
          next_k_shapes = next_k_shapes[:, indices].to(self.args.device)
      else:
          rotIdx = torch.zeros_like(shapeIdx).long()
          next_k_shapes = self.shapeArray[shapeIdx, rotIdx].float().to(self.args.device)
      shape_feature = self.shapeEncoder(next_k_shapes)
      if not self.preEncoder:
        shape_feature = torch.max(shape_feature, dim=1)[0]

      init_embedding = torch.cat((shape_feature.reshape(batchSize, candidates_size, -1),
                                  map_feature.repeat((1, candidates_size)).reshape(batchSize, candidates_size, -1)), dim=2).view(batchSize * candidates_size, -1)
      init_embedding = self.noAction(init_embedding).view(batchSize, candidates_size, self.embedding_dim)
      invalid_ones = torch.zeros((batchSize, candidates_size))
      embeddings, _ = self.embedder(init_embedding, mask=invalid_ones, limited=True)
      embedding_shape = embeddings.shape

      transEmbedding = embeddings.view((batchSize, graph_size, -1))
      graph_embed = transEmbedding.view(embedding_shape).mean(1)
      return embeddings, graph_embed

  def embed_physic_with_heightmap_and_k_shape(self, x):
      batchSize = x.shape[0]

      next_item, actionMask, heightMap, candidates, next_k_shapes_ID = observation_decode_irregular_with_k_shape(x, self.args)
      graph_size = candidates.size(1)

      valid_mask = actionMask
      invalid_ones = 1 - valid_mask  # mask 为1的地方是被删掉的地方

      candidates_size = candidates.size(1)
      heightMap = heightMap.reshape((batchSize, 1, self.MapLength, self.MapLength))
      map_feature = self.heightEncoder(heightMap).reshape((batchSize, -1))

      next_item_ID = next_item[:, 0].long()
      nextShape = self.shapeArray[next_item_ID, 0].to(self.args.device)
      selected_shape_feature = self.shapeEncoder(nextShape)
      if not self.preEncoder:
        selected_shape_feature = torch.max(selected_shape_feature, dim=1)[0]

      shapeNum = next_k_shapes_ID.shape[-1]
      allShapeIdx = next_k_shapes_ID.detach().long().reshape(-1)
      allRotIdx = torch.zeros_like(allShapeIdx).long()
      next_k_shapes = self.shapeArray[allShapeIdx, allRotIdx].float().to(self.args.device)
      all_shape_feature = self.shapeEncoder(next_k_shapes)
      if not self.preEncoder:
        all_shape_feature = torch.max(all_shape_feature, dim=1)[0]
      all_shape_feature = all_shape_feature.reshape(batchSize, shapeNum, -1)
      all_shape_feature = torch.max(all_shape_feature, dim=1)[0]

      ems_inputs = candidates.contiguous().view(batchSize, candidates_size, -1)
      ems_embedded_inputs = self.init_ems_embed(ems_inputs)
      init_embedding = torch.cat((selected_shape_feature.repeat(1, candidates_size).reshape(batchSize, candidates_size, -1),
                                  all_shape_feature.repeat(1, candidates_size).reshape(batchSize, candidates_size,-1),
                                  map_feature.repeat((1, candidates_size)).reshape(batchSize, candidates_size, -1),
                                  ems_embedded_inputs), dim=2).view(batchSize * candidates_size, -1)
      init_embedding = self.oneMore(init_embedding).view(batchSize, candidates_size, self.embedding_dim)
      # embeddings, _ = self.embedder(init_embedding, mask=invalid_ones, limited=True)  # 前向没啥问题了，然后开始算概率
      embeddings = init_embedding
      embedding_shape = embeddings.shape

      transEmbedding = embeddings.view((batchSize, graph_size, -1))
      invalid_ones = invalid_ones.view(embedding_shape[0], embedding_shape[1], 1).expand(embedding_shape).bool()
      transEmbedding[invalid_ones] = 0
      graph_embed = transEmbedding.view(embedding_shape).mean(1)  # 其实这里的取均值一定程度上相当于相加了


      return embeddings, graph_embed

  def embed_physic_only_with_heightmap_fixed_indices(self, x):
      batchSize = x.shape[0]

      next_item, actionMask, heightMap, candidates, indices = observation_decode_irregular_indices(x, self.args)
      indices = indices.long()
      graph_size = candidates.size(1)

      valid_mask = actionMask
      invalid_ones = 1 - valid_mask  # mask 为1的地方是被删掉的地方

      candidates_size = candidates.size(1)
      heightMap = heightMap.reshape((batchSize, 1, self.MapLength, self.MapLength))
      map_feature = self.heightEncoder(heightMap).reshape((batchSize, -1))

      next_item_ID = next_item[:, 0].long()
      if self.args.shapePreType == 'SurfacePointsRandom' or self.args.shapePreType == 'SurfacePointsEncode':
          nextShape = self.shapeArray[next_item_ID]
          batchIdx = torch.arange(0, batchSize).reshape(-1, 1).repeat(repeats=(1,self.args.samplePointsNum)).reshape(-1)
          indices = indices.reshape(-1)
          nextShape = nextShape[batchIdx, indices]
          nextShape = nextShape.reshape((batchSize, -1, 3)).to(self.args.device)
      else:
          nextShape = self.shapeArray[next_item_ID, 0].to(self.args.device)

      assert not self.preEncoder
      if  self.args.shapePreType == 'SurfacePointsEncode':
          self.shapeEncoder.eval()
          shape_feature = self.shapeEncoder(nextShape)
          shape_feature = F.normalize(shape_feature, dim=1)
      else:
          shape_feature = self.shapeEncoder(nextShape)
          shape_feature = torch.max(shape_feature, dim=1)[0]

      ems_inputs = candidates.contiguous().view(batchSize, candidates_size, -1)
      ems_embedded_inputs = self.init_ems_embed(ems_inputs)
      init_embedding = torch.cat((shape_feature.repeat(1, candidates_size).reshape(batchSize, candidates_size, -1),
                                  map_feature.repeat((1, candidates_size)).reshape(batchSize, candidates_size, -1),
                                  ems_embedded_inputs), dim=2).view(batchSize * candidates_size, -1)
      init_embedding = self.oneMore(init_embedding).view(batchSize, candidates_size, self.embedding_dim)
      embeddings = init_embedding
      embedding_shape = embeddings.shape

      transEmbedding = embeddings.view((batchSize, graph_size, -1))
      invalid_ones = invalid_ones.view(embedding_shape[0], embedding_shape[1], 1).expand(embedding_shape).bool()
      transEmbedding[invalid_ones] = 0
      graph_embed = transEmbedding.view(embedding_shape).mean(1)

      return embeddings, graph_embed


  def forward(self, x, log=False,  getCL = False):

      loss_cl = None
      assert self.args.previewNum > 1
      x, xGlobal = self.embed_physic_k_shape_with_gat(x)
      v = self.fc_z_v(F.relu(self.fc_h_v(xGlobal)))  # Value stream
      a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
      v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
      q = v + a - a.mean(1, keepdim=True)  # Combine streams
      if log:  # Use log softmax for numerical stability
        q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
      else:
        q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
      self.forwardCounter += 1
      if self.forwardCounter == 1000:
          self.updateShapeArray()

      if getCL:
          return q, loss_cl
      else:
          return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()
