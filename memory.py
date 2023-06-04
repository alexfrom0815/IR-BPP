# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
import torch
import torch.multiprocessing as mp


Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))

class Value():
  def __init__(self,  value):
    self.value = value

# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
  def __init__(self, size, obs_len, args):
    self.distributed = args.distributed

    if self.distributed:
      self.index = mp.Value('l', 0)
      self.full = mp.Value('b', False)
    else:
      self.index = Value(0)
      self.full  = Value(False)

    self.size  = size
    self.max = 1  # Initial max value to return (1 = 1^ω)

    self.sum_tree = torch.zeros((2 * size - 1, ), dtype=torch.float32)
    self.timesteps = torch.zeros(size, 1)
    self.states = torch.zeros(size, obs_len)
    self.actions = torch.zeros(size, 1)
    self.rewards = torch.zeros(size, 1)
    self.nonterminals = torch.zeros(size, 1)

    if args.distributed:
      self.sum_tree = self.sum_tree.share_memory_()
      self.timesteps = self.timesteps.share_memory_()
      self.states = self.states.share_memory_()
      self.actions = self.actions.share_memory_()
      self.rewards = self.rewards.share_memory_()
      self.nonterminals = self.nonterminals.share_memory_()

  # Propagates value up tree given a tree index
  def _propagate(self, index, value):
    parent = (index - 1) // 2
    left, right = 2 * parent + 1, 2 * parent + 2
    self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
    if parent != 0:
      self._propagate(parent, value)

  # Updates value given a tree index
  def update(self, index, value):
    self.sum_tree[index] = value  # Set new value
    self._propagate(index, value)  # Propagate value
    self.max = max(value, self.max)

  def append(self, data, value):
    # self.data[self.index] = data  # Store data in underlying data structure
    index = self.index.value
    self.timesteps[index] = data[0]
    self.states[index] = data[1]
    self.actions[index] = data[2]
    self.rewards[index] = data[3]
    self.nonterminals[index] = data[4]
    self.update(index + self.size - 1, value)  # Update tree
    self.index.value = (index + 1) % self.size  # Update index
    self.full.value = self.full.value or self.index.value == 0  # Save when capacity reached
    self.max = max(value, self.max)

  # Searches for the location of a value in sum tree
  def _retrieve(self, index, value):
    left, right = 2 * index + 1, 2 * index + 2
    if left >= len(self.sum_tree):
      return index
    elif value <= self.sum_tree[left]:
      return self._retrieve(left, value)
    else:
      return self._retrieve(right, value - self.sum_tree[left])

  # Searches for a value in sum tree and returns value, data index and tree index
  def find(self, value):
    index = self._retrieve(0, value)  # Search for index of item from root
    data_index = index - self.size + 1
    return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index


  def getBatch(self, data_indexs):
    data_indexs = data_indexs % self.size
    return self.timesteps[data_indexs], self.states[data_indexs], self.actions[data_indexs], self.rewards[data_indexs], self.nonterminals[data_indexs]

  def total(self):
    return self.sum_tree[0]

class ReplayMemory():
  def __init__(self, args, capacity, obs_len):
    self.device = args.device
    self.capacity = capacity
    # self.history = args.history_length
    self.discount = args.discount
    self.n = args.multi_step
    self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
    self.priority_exponent = args.priority_exponent
    self.t = 0  # Internal episode timestep counter
    self.transitions = SegmentTree(capacity, obs_len, args)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
    self.blank_trans = [torch.tensor((0,)), torch.zeros(obs_len, dtype=torch.float), torch.tensor((0,)), torch.tensor((0,)), torch.tensor((False,))]
    self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32, device=self.device)  # Discount-scaling vector for n-step returns

  # Adds state and action at time t, reward and terminal at time t + 1
  def append(self, state, action, reward, terminal):
    state = state.to(dtype=torch.float32, device=torch.device('cpu'))  # Only store last frame and discretise to save memory
    self.transitions.append(Transition(self.t, state, action, reward, not terminal), self.transitions.max)  # Store new transition with maximum priority
    self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

  def _get_transition_new(self, idx):
    timesteps, states, actions, rewards, nonterminals = [], [], [], [], []

    for t in range(0, 1 + self.n):  # e.g. 4 5 6
      if  t == 0 or nonterminals[-1] :
        timestep, state, action, reward, nonterminal = self.transitions.getBatch(idx + t)
      else:
        timestep, state, action, reward, nonterminal = self.blank_trans
      timesteps.append(timestep)
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      nonterminals.append(nonterminal)

    return torch.cat(timesteps, dim=0), torch.stack(states, dim=0),torch.cat(actions, dim=0),\
           torch.cat(rewards, dim=0),torch.cat(nonterminals, dim=0)

  def _get_transitions_batch(self, idxs):
    transition_idxs = np.arange(0, self.n + 1) + np.expand_dims(idxs, axis=1)
    index_shape = transition_idxs.shape
    timesteps, states, actions, rewards, nonterminals = self.transitions.getBatch(transition_idxs)
    transitions_firsts = timesteps == 0
    transitions_firsts = transitions_firsts.reshape(index_shape)
    blank_mask = torch.zeros_like(transitions_firsts, dtype = torch.bool)

    for t in range(1, 1 + self.n):  # e.g. 4 5 6
      blank_mask[:, t] = torch.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t]) # True if current or past frame has timestep 0
    blank_mask = blank_mask.reshape(-1)

    timesteps[blank_mask] = 0
    timesteps = timesteps.reshape(*index_shape)

    states[blank_mask][:] = 0
    states = states.reshape((*index_shape, -1))

    actions[blank_mask] = 0
    actions = actions.reshape(index_shape)

    rewards[blank_mask] = 0
    rewards = rewards.reshape(index_shape)

    nonterminals[blank_mask] = False
    nonterminals = nonterminals.reshape(index_shape)

    return timesteps, states, actions, rewards, nonterminals


  # Returns a valid sample from a segment
  def _get_sample_from_segment(self, segment, i):
    valid = False
    while not valid:
      sample = np.random.uniform(i * segment, (i + 1) * segment)  # Uniformly sample an element from within a segment
      prob, idx, tree_idx = self.transitions.find(sample)  # Retrieve sample from tree with un-normalised probability
      # Resample if transition straddled current index or probablity 0
      if (self.transitions.index.value - idx) % self.capacity > self.n and (idx - self.transitions.index.value) % self.capacity >= 1 and prob != 0:
        valid = True  # Note that conditions are valid but extra conservative around buffer index 0

    # Retrieve all required transition data (from t - h to t + n)
    Btimesteps, Bstates, Bactions, Brewards, Bnonterminals  = self._get_transition_new(idx)

    state = Bstates[0].to(dtype=torch.float32, device=self.device)
    next_state = Bstates[self.n].to(dtype=torch.float32, device=self.device)

    action = Bactions[0].to(dtype=torch.int64, device=self.device)

    # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
    reward = Brewards[0:-1].to(dtype=torch.float32, device=self.device)
    R = torch.matmul(reward, self.n_step_scaling)

    # Mask for non-terminal nth next states
    nonterminal = Bnonterminals[self.n] .to(dtype=torch.float32, device=self.device)

    return prob, idx, tree_idx, state, action, R, next_state, nonterminal


  # Some data augumentation tricks can be used.
  def sample(self, batch_size):
    p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
    segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
    batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]  # Get batch of valid samples
    probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)

    states, next_states, = torch.stack(states), torch.stack(next_states)
    actions, returns, nonterminals = torch.stack(actions), torch.stack(returns), torch.stack(nonterminals)
    probs = np.array(probs, dtype=np.float32) / p_total  # Calculate normalised probabilities
    capacity = self.capacity if self.transitions.full.value else self.transitions.index.value
    weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
    weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch
    return list(tree_idxs), states, actions, returns, next_states, nonterminals, weights


  def update_priorities(self, idxs, priorities):
    priorities = np.power(priorities, self.priority_exponent)
    [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

  # Set up internal state for iterator
  def __iter__(self):
    self.current_idx = 0
    return self

