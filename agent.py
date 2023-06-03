# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from model import DQNBPP
import math

# Finished

class Agent():
  def __init__(self, args):
    self.action_space = args.action_space
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step # For GAE implementation.
    self.discount = args.discount
    self.norm_clip = args.norm_clip

    shapeArray = torch.tensor(args.shapeArray).type(torch.float).share_memory_()
    network = DQNBPP
    self.online_net = network(args, self.action_space, shapeArray).to(device=args.device).share_memory()

    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
        self.online_net.load_state_dict(state_dict)
        print("Loading pretrained model: " + args.model)
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)

    self.online_net.train()
    self.target_net = network(args, self.action_space, shapeArray).to(device=args.device).share_memory()
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False
    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state, mask):
    with torch.no_grad():
      q_map = self.online_net(state)
      sum_q_map = q_map * self.support
      sum_q_map = sum_q_map.sum(2)
      if mask is not None:
        sum_q_map[(1 - mask).bool()] = -math.inf
    return sum_q_map.argmax(1)

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, mask, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    if np.random.random() < epsilon:
      return np.random.randint(0, self.action_space)
    else:
      return self.act(state, mask)

  # core part
  def learn(self, memory):
    segment_size = int(self.batch_size/len(memory))
    idxs, states, actions, returns, next_states, nonterminals, weights = [],[],[],[],[],[],[]

    for mem in memory:
      idx, state, action, ret, next_state, nonterminal, weight = mem.sample(segment_size)
      idxs.append(idx), states.append(state), actions.append(action), returns.append(ret)
      next_states.append(next_state), nonterminals.append(nonterminal), weights.append(weight)

    states = torch.cat(states, 0)
    actions = torch.cat(actions, 0)
    returns = torch.cat(returns, 0)
    next_states = torch.cat(next_states, 0)
    nonterminals = torch.cat(nonterminals, 0).reshape(self.batch_size, 1)
    weights = torch.cat(weights, 0)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps= self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]

      self.target_net.reset_noise()  # Sample new target net noise

      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values

      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)

      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(m * log_ps_a, 1)
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    for i in range(len(memory)):
      memory[i].update_priorities(idxs[i], loss[i * segment_size : (i+1) * segment_size].detach().cpu())  # Update priorities of sampled transitions
    return loss

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
