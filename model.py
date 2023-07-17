# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tools import init


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return {'data':input['data'] + self.module(input), 'mask': input['mask'], 'graph_size':input['graph_size']}

class SkipConnection_Linear(nn.Module):
    def __init__(self, module):
        super(SkipConnection_Linear, self).__init__()
        self.module = module

    def forward(self, input):
        return {'data':input['data'] + self.module(input['data']), 'mask': input['mask'], 'graph_size': input['graph_size']}

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Linear(input_dim, key_dim, bias=False)
        self.W_key = nn.Linear(input_dim, key_dim, bias=False)
        self.W_val = nn.Linear(input_dim, val_dim, bias=False)

        if embed_dim is not None:
            # self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
            self.W_out = nn.Linear(key_dim, embed_dim)

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, data, h=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        q = data['data']
        mask = data['mask']
        graph_size = data['graph_size']
        if h is None:
            h = q

        # batch_size = int(q.size()[0] / graph_size)
        batch_size = q.size()[0]
        graph_size = graph_size
        input_dim = h.size()[-1]
        n_query = graph_size
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        Q = self.W_query(qflat).view(shp_q)
        K = self.W_key(hflat).view(shp)
        V = self.W_val(hflat).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.unsqueeze(1).repeat((1, graph_size, 1)).bool()
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            if data['evaluate']:
                compatibility[mask] = -math.inf
            else:
                compatibility[mask] = -30
        attn = torch.softmax(compatibility, dim=-1) #

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)
        out = self.W_out(heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim))
        out = out.view(batch_size, n_query, self.embed_dim)
        return out

class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=128):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                )
            ),
            SkipConnection_Linear(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
        )

class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            feed_forward_hidden=128,
            graph_size=None,
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        self.graph_size = graph_size
        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None, limited=False, evaluate = False):
        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        data = {'data':h, 'mask': mask, 'graph_size': self.graph_size, 'evaluate': evaluate}
        h = self.layers(data)['data']
        return (h, h.view(h.size()[0], self.graph_size, -1).mean(dim=1),)


def observation_decode_irregular(observation, args):
    batchSize = observation.shape[0]
    observation = observation.reshape((batchSize, -1))
    actions = observation[:, 0 : args.selectedAction * 5].reshape(batchSize, -1, 5)
    next_item = observation[:, args.selectedAction * 5 : args.selectedAction * 5 + 1].reshape((batchSize, -1))
    actionMasks = actions[:,:, -1]
    actions = actions[:,:, 0:-1]
    heightMap = observation[:, args.selectedAction * 5 + 9:]
    return next_item, actionMasks, heightMap, actions

def observation_decode_irregular_k_shape(observation, args):
    batchSize = observation.shape[0]
    observation = observation.reshape((batchSize, -1))
    shapes = observation[:, 0 : args.bufferSize].reshape(batchSize, args.bufferSize)
    heightMap = observation[:, args.bufferSize:]
    return shapes, heightMap

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


class DQNBPP(nn.Module):
  def __init__(self, args, action_space, shapeArray):
    super(DQNBPP, self).__init__()
    assert args.selectedAction
    self.args = args
    self.atoms = args.atoms # c51
    self.action_space = action_space
    assert shapeArray is not None
    self.orginArray = shapeArray
    self.arrayPointNum  = 10000
    self.shapeArray = shapeArray
    if self.args.level == 'order':
        self.shapeArray = torch.zeros((shapeArray.shape[0], self.arrayPointNum, shapeArray.shape[2]))
    self.updateShapeArray()

    self.heightMap = args.heightMap
    self.rotNum = args.ZRotNum
    self.MapLength = int(args.bin_dimension[0] / args.resolutionH)

    zDim = 256
    self.zDim = zDim
    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('leaky_relu'))
    self.output_size = zDim * 2

    # Network components
    # you need to customize your cnn kernel here.
    assert args.resolutionH == 0.01, 'you need to customize your cnn kernel here'
    self.heightEncoder = nn.Sequential()
    self.heightEncoder.add_module('conv1', init_(nn.Conv2d(1, 16, 4, stride=2, padding=1)))  # 32 -> 16
    self.heightEncoder.add_module('relu1', nn.LeakyReLU())
    self.heightEncoder.add_module('conv2', init_(nn.Conv2d(16, 32, 4, stride=2, padding=1)))  # 16 -> 8
    self.heightEncoder.add_module('relu2', nn.LeakyReLU())
    self.heightEncoder.add_module('conv3', init_(nn.Conv2d(32, 4, 3, stride=1, padding=1)))  # 16 -> 8
    self.heightEncoder.add_module('relu3', nn.LeakyReLU())

    self.shapeEncoder = nn.Sequential()
    self.shapeEncoder.add_module('linear1', init_(nn.Linear(3, 128)))
    self.shapeEncoder.add_module('relu1', nn.LeakyReLU())
    self.shapeEncoder.add_module('linear2', init_(nn.Linear(128, 128)))
    self.shapeEncoder.add_module('relu2', nn.LeakyReLU())

    self.init_candidate_embed = nn.Sequential(
        init_(nn.Linear(4, 32)),
        nn.LeakyReLU(),
        init_(nn.Linear(32, 128)))
    self.embedding_dim = 128
    self.project_layer = nn.Sequential(
        init_(nn.Linear(512, 256)),
        nn.LeakyReLU(),
        init_(nn.Linear(256, self.embedding_dim)))



    if args.bufferSize > 1:
        self.embedder = GraphAttentionEncoder(
            n_heads=1,
            embed_dim=self.embedding_dim,
            n_layers=1,
            graph_size=args.bufferSize,
        )
        self.gat_project_layer = nn.Sequential(
            init_(nn.Linear(384, 256)),
            nn.LeakyReLU(),
            init_(nn.Linear(256, self.embedding_dim)))
        
    self.fc_h_v = NoisyLinear(self.embedding_dim, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.embedding_dim, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)

  def updateShapeArray(self):
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

  def embed_heightmap_and_sampled_point_cloud(self, x):
      batchSize = x.shape[0]
      next_item, actionMask, heightMap, candidates = observation_decode_irregular(x, self.args)
      graph_size = candidates.size(1)

      valid_mask = actionMask
      invalid_ones = 1 - valid_mask

      candidates_size = candidates.size(1)
      heightMap = heightMap.reshape((batchSize, 1, self.MapLength, self.MapLength))
      map_feature = self.heightEncoder(heightMap).reshape((batchSize, -1))

      next_item_ID = next_item[:, 0].long()

      nextShape = self.shapeArray[next_item_ID.cpu()]
      indices = np.random.randint(self.shapeArray.shape[1], size=self.args.samplePointsNum)
      nextShape = nextShape[:, indices].to(self.args.device)

      shape_feature = self.shapeEncoder(nextShape)
      shape_feature = torch.max(shape_feature, dim=1)[0]

      candidate_inputs = candidates.contiguous().view(batchSize, candidates_size, -1)
      candidate_embedded_inputs = self.init_candidate_embed(candidate_inputs)
      init_embedding = torch.cat((shape_feature.repeat(1, candidates_size).reshape(batchSize, candidates_size, -1),
                                  map_feature.repeat((1, candidates_size)).reshape(batchSize, candidates_size, -1),
                                  candidate_embedded_inputs), dim=2).view(batchSize * candidates_size, -1)
      init_embedding = self.project_layer(init_embedding).view(batchSize, candidates_size, self.embedding_dim)

      embeddings = init_embedding
      embedding_shape = embeddings.shape

      transEmbedding = embeddings.view((batchSize, graph_size, -1))
      invalid_ones = invalid_ones.view(embedding_shape[0], embedding_shape[1], 1).expand(embedding_shape).bool()
      transEmbedding[invalid_ones] = 0
      graph_embed = transEmbedding.view(embedding_shape).mean(1)

      return embeddings, graph_embed


  def embed_k_buffer_shape_with_gat(self, x):
      batchSize = x.shape[0]

      next_k_shapes_ID, heightMap = observation_decode_irregular_k_shape(x, self.args)
      graph_size = next_k_shapes_ID.size(1)

      candidates_size = graph_size
      heightMap = heightMap.reshape((batchSize, 1, self.MapLength, self.MapLength))
      map_feature = self.heightEncoder(heightMap).reshape((batchSize, -1))

      shapeIdx = next_k_shapes_ID.detach().cpu().long().reshape(-1)
      next_k_shapes = self.shapeArray[shapeIdx]
      indices = np.random.randint(self.shapeArray.shape[1], size=self.args.samplePointsNum)
      next_k_shapes = next_k_shapes[:, indices].to(self.args.device)

      shape_feature = self.shapeEncoder(next_k_shapes)

      shape_feature = torch.max(shape_feature, dim=1)[0]

      init_embedding = torch.cat((shape_feature.reshape(batchSize, candidates_size, -1),
                                  map_feature.repeat((1, candidates_size)).reshape(batchSize, candidates_size, -1)), dim=2).view(batchSize * candidates_size, -1)
      init_embedding = self.gat_project_layer(init_embedding).view(batchSize, candidates_size, self.embedding_dim)
      invalid_ones = torch.zeros((batchSize, candidates_size))
      embeddings, _ = self.embedder(init_embedding, mask=invalid_ones, limited=True)
      embedding_shape = embeddings.shape

      transEmbedding = embeddings.view((batchSize, graph_size, -1))
      graph_embed = transEmbedding.view(embedding_shape).mean(1)
      return embeddings, graph_embed


  def forward(self, x, log=False):

      if self.args.bufferSize > 1:
        x, xGlobal = self.embed_k_buffer_shape_with_gat(x)
      else:
        x, xGlobal = self.embed_heightmap_and_sampled_point_cloud(x)

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
      return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()
