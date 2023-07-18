# -*- coding: utf-8 -*-
import copy

import os
import numpy as np
import torch
from agent import Agent
from memory import ReplayMemory
from tensorboardX import SummaryWriter
import time
from tools import backup, registration_envs,  load_shape_dict, shotInfoPre, shapeProcessing, test_hierachical, test
from trainer import trainer, trainer_hierarchical
from arguments import get_args
import gym
from envs import make_vec_envs

def main(args):

    # The name of this experiment, file backups and experiment tensorboard logs will be saved in related folder
    if args.custom is None:
        args.custom = input('Please input the experiment name\n')
    timeStr = args.custom + '-' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

    # Set the device
    if torch.cuda.is_available() and not args.disable_cuda:
      args.device = torch.device('cuda:{}'.format(args.device))
      torch.cuda.manual_seed(args.seed)
      torch.backends.cudnn.enabled = args.enable_cudnn
    else:
        args.device = torch.device('cpu')

    if args.device.type.lower() != 'cpu':
        torch.cuda.set_device(args.device)

    torch.set_num_threads(1)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    if args.distributed:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)

    # Backup all py files and create tensorboard logs
    backup(timeStr, args)
    log_writer_path = './logs/runs/{}'.format('IR-' + timeStr)
    if not os.path.exists(log_writer_path):
      os.makedirs(log_writer_path)
    writer = SummaryWriter(log_writer_path)

    tempenv = gym.make(args.envName, args=args)
    args.action_space = tempenv.action_space.n


    if not args.hierachical:
        # Not hierachical, only a location policy to solve the online packing problem
        args.level = 'location'
        args.model = args.locmodel
        dqn = Agent(args)
        memNum = args.num_processes
        memory_capacity = int(args.memory_capacity / memNum)
        mem = [ReplayMemory(args, memory_capacity, tempenv.obs_len) for _ in range(memNum)]
        trainTool = trainer(writer, timeStr, dqn, mem)
    else:
        # Besides the location policy, an order policy is also needed to solve the bufferd packing problem
        orderArgs = copy.deepcopy(args)
        orderArgs.level = 'order'
        orderArgs.action_space = args.bufferSize
        orderArgs.model = args.ordmodel
        orderDQN = Agent(orderArgs)

        locArgs = copy.deepcopy(args)
        locArgs.level = 'location'
        locArgs.bufferSize = 1
        locArgs.action_space = args.selectedAction
        locArgs.model = args.locmodel
        locDQN = Agent(locArgs)

        memNum = args.num_processes
        memory_capacity = int(args.memory_capacity / memNum)
        heightmapSize = np.prod(np.ceil(args.bin_dimension[0:2] / args.resolutionH).astype(np.int32))
        order_obs_len = heightmapSize + args.bufferSize

        loc_obs_len = heightmapSize + args.selectedAction * 5 + args.objVecLen

        orderMem = [ReplayMemory(args, memory_capacity, order_obs_len) for _ in range(memNum)]
        locMem = [ReplayMemory(args, memory_capacity, loc_obs_len) for _ in range(memNum)]

        # Perform all training.
        trainTool = trainer_hierarchical(writer, timeStr, [orderDQN, locDQN], [orderMem, locMem])

    if args.evaluate:
        # Perform testing
        if args.hierachical:
            test_hierachical(args, [orderDQN, locDQN], True, timeStr)  # Test
        else:
            test(args, dqn, True,  timeStr)  # Test
    else:
        # Perform all training.
        envs, spaces, obs_len = make_vec_envs(args, './logs/runinfo', True)
        trainTool.train_q_value(envs, args)

if __name__ == '__main__':
    registration_envs()
    args = get_args()
    main(args)

