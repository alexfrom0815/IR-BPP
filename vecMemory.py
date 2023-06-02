# -*- coding: utf-8 -*-
import numpy as np
import torch

# 本身就有bug，实现效果不是很好
# blank_trans = (0, np.zeros((84, 84), dtype=np.uint8), 0, 0.0, False)

# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
  def __init__(self, size, obs_len):
    self.index = 0
    self.size = size
    self.full = False  # Used to track actual capacity
    self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
    self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
    self.max = 1  # Initial max value to return (1 = 1^ω)

    self.timesteps = torch.zeros(size, 1).share_memory_()
    self.states = torch.zeros(size, obs_len).share_memory_()
    self.actions = torch.zeros(size, 1).share_memory_()
    self.rewards = torch.zeros(size, 1).share_memory_()
    self.nonterminals = torch.zeros(size, 1).share_memory_()

  # Updates nodes values from current tree
  def _update_nodes(self, indices):
    children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
    self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

  # Propagates changes up tree given tree indices
  def _propagate(self, indices):
    parents = (indices - 1) // 2
    unique_parents = np.unique(parents)
    self._update_nodes(unique_parents)
    if parents[0] != 0:
      self._propagate(parents)

  # Propagates single value up tree given a tree index for efficiency
  def _propagate_index(self, index):
    parent = (index - 1) // 2
    left, right = 2 * parent + 1, 2 * parent + 2
    self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
    if parent != 0:
      self._propagate_index(parent)

  # Updates values given tree indices
  def update(self, indices, values):
    self.sum_tree[indices] = values  # Set new values
    self._propagate(indices)  # Propagate values
    current_max_value = np.max(values)
    self.max = max(current_max_value, self.max)

  # Updates single value given a tree index for efficiency
  def _update_index(self, index, value):
    self.sum_tree[index] = value  # Set new value
    self._propagate_index(index)  # Propagate value
    self.max = max(value, self.max)

  def append(self, data, value):
    self.timesteps[self.index] = data[0]
    self.states[self.index] = data[1]
    self.actions[self.index] = data[2]
    self.rewards[self.index] = data[3]
    self.nonterminals[self.index] = data[4]

    self._update_index(self.index + self.tree_start, value)  # Update tree
    self.index = (self.index + 1) % self.size  # Update index
    self.full = self.full or self.index == 0  # Save when capacity reached
    self.max = max(value, self.max)

  # Searches for the location of values in sum tree
  def _retrieve(self, indices, values):
    children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
    # If indices correspond to leaf nodes, return them
    if children_indices[0, 0] >= self.sum_tree.shape[0]:
      return indices
    # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
    elif children_indices[0, 0] >= self.tree_start:
      children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
    left_children_values = self.sum_tree[children_indices[0]]
    successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
    successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
    successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
    return self._retrieve(successor_indices, successor_values)

  # Searches for values in sum tree and returns values, data indices and tree indices
  def find(self, values):
    indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
    data_index = indices - self.tree_start
    return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

  # Returns data given a data index
  # def get(self, data_index):
  #   return self.data[data_index % self.size]

  def getBatch(self, data_indexs):
    data_indexs = data_indexs.reshape(-1) % self.size
    return self.timesteps[data_indexs], \
           self.states[data_indexs], \
           self.actions[data_indexs], \
           self.rewards[data_indexs], \
           self.nonterminals[data_indexs]

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
    self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32, device=self.device)  # Discount-scaling vector for n-step returns
    # self.blank_trans = (0, np.zeros(obs_len, dtype=np.float32), 0, 0.0, False)
    self.transitions = SegmentTree(capacity,  obs_len)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

  # Adds state and action at time t, reward and terminal at time t + 1
  def append(self, state, action, reward, terminal):
    state = state[-1].to(dtype=torch.float32, device=torch.device('cpu'))  # Only store last frame and discretise to save memory
    self.transitions.append((self.t, state, action, reward, not terminal), self.transitions.max)  # Store new transition with maximum priority
    self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

  # (0, np.zeros(obs_len, dtype=np.float32), 0, 0.0, False)
  # Returns the transitions with blank states where appropriate
  def _get_transitions(self, idxs):
    transition_idxs = np.arange(0, self.n + 1) + np.expand_dims(idxs, axis=1)
    transitions = self.transitions.get(transition_idxs)
    transitions_firsts = transitions['timestep'] == 0
    blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)

    for t in range(1, 1 + self.n):  # e.g. 4 5 6
      blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t]) # True if current or past frame has timestep 0

    transitions[blank_mask] = self.blank_trans
    return transitions

  # Returns the transitions with blank states where appropriate
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

  # Returns a valid sample from each segment 你怎么判断中间有没有中止呢
  def _get_samples_from_segments(self, batch_size, p_total):

    segment_length = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
    segment_starts = np.arange(batch_size) * segment_length
    valid = False
    while not valid:
      samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts  # Uniformly sample from within all segments
      probs, idxs, tree_idxs = self.transitions.find(samples)  # Retrieve samples from tree with un-normalised probability
      if np.all((self.transitions.index - idxs) % self.capacity > self.n) and np.all((idxs - self.transitions.index) % self.capacity >= 1) and np.all(probs != 0):
        valid = True  # Note that conditions are valid but extra conservative around buffer index 0

    # probs = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    # idxs = np.array([ 4 , 8 ,14 ,23 ,25 ,35 ,38 ,45 ,55 ,60 ,65 ,68 ,75 ,83 ,89 ,96])
    # tree_idxs = np.array([4099 ,4103 ,4109 ,4118 ,4120 ,4130 ,4133 ,4140 ,4150 ,4155 ,4160 ,4163 ,4170 ,4178, 4184 ,4191])

    # Retrieve all required transition data (from t - h to t + n)
    Btimesteps, Bstates, Bactions, Brewards, Bnonterminals = self._get_transitions_batch(idxs)
    # Create un-discretised states and nth next states
    states = Bstates[:, :1].to(dtype=torch.float32, device=self.device)
    next_states = Bstates[:, self.n:self.n + 1].to(dtype=torch.float32, device=self.device)

    # Discrete actions to be used as index
    actions = Bactions[:, 0].to(dtype=torch.int64, device=self.device)

    # Calculate truncated n-step discounted returns R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
    rewards = Brewards[:, 0:-1].to(dtype=torch.float32, device=self.device)
    R = torch.matmul(rewards, self.n_step_scaling)

    # Mask for non-terminal nth next states
    nonterminals = torch.unsqueeze(Bnonterminals[:, self.n], dim=1).to(dtype=torch.float32, device=self.device)
    return probs, idxs, tree_idxs, states, actions, R, next_states, nonterminals

  def sample(self, batch_size):
    p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
    probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = self._get_samples_from_segments(batch_size, p_total)  # Get batch of valid samples
    probs = probs / p_total  # Calculate normalised probabilities
    capacity = self.capacity if self.transitions.full else self.transitions.index
    weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
    weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch
    return tree_idxs, states, actions, returns, next_states, nonterminals, weights

  def update_priorities(self, idxs, priorities):
    priorities = np.power(priorities, self.priority_exponent)
    self.transitions.update(idxs, priorities)

  # Set up internal state for iterator
  def __iter__(self):
    self.current_idx = 0
    return self

  # def __next__(self):
  #   if self.current_idx == self.capacity:
  #     raise StopIteration
  #
  #   assert False
  #
  #   r = np.arange(self.current_idx, self.current_idx + 1)
  #   timesteps = self.transitions.timesteps[r]
  #   states = self.transitions.states[r]
  #   actions = self.transitions.actions[r]
  #   rewards = self.transitions.rewards[r]
  #   nonterminals = self.transitions.nonterminals[r]
  #
  #   transitions_firsts = timesteps == 0
  #   blank_mask = torch.zeros_like(transitions_firsts, dtype=torch.bool)
  #
  #   timesteps[blank_mask] = 0
  #   states[blank_mask][:] = 0
  #   actions[blank_mask] = 0
  #   rewards[blank_mask] = 0
  #   nonterminals[blank_mask] = False
  #
  #   state = states.to(dtype=torch.float32, device=self.device)
  #   self.current_idx += 1
  #   return state

  # next = __next__  # Alias __next__ for Python 2 compatibility
