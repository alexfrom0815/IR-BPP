# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tools import init
from graph_encoder import GraphAttentionEncoder
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

    self.shapeEncoder = nn.Sequential()
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
        self.embedder = GraphAttentionEncoder(
            n_heads=1,
            embed_dim=self.embedding_dim,
            n_layers=1,
            graph_size=args.previewNum,
        )

    self.fc_h_v = NoisyLinear(self.embedding_dim, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.embedding_dim, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
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
          indices = np.random.randint(self.shapeArray.shape[1], size=self.args.samplePointsNum)
          # indices = self.args.globalIndices
          nextShape = nextShape[:, indices].to(self.args.device)
      else:
          nextShape = self.shapeArray[next_item_ID, 0].to(self.args.device)

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


  def forward(self, x, log=False,  getCL = False):

      loss_cl = None
      if self.args.previewNum > 1:
        x, xGlobal = self.embed_physic_k_shape_with_gat(x)
      else:
        x, xGlobal = self.embed_physic_only_with_heightmap(x)

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
