# -*- coding: utf-8 -*-
import copy

import os
import numpy as np
import torch
from agent import Agent
from memory import ReplayMemory
from tensorboardX import SummaryWriter
import time
import config
from tools import backup, registration_envs,  load_shape_dict, shotInfoPre, shapeProcessing
from trainer import trainer, trainer_hierarchical
from arguments import get_args
from envs import make_vec_envs

def main(args):

    # The name of this experiment, related file backups and experiment tensorboard logs will
    if args.custom is None:
        assert args.data_name is None
        custom = input('Please input the experiment name\n')
    else:
        custom = args.custom

    timeStr = custom + '-' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

    if torch.cuda.is_available() and not args.disable_cuda:
      if args.data_name is not None:
          args.device = torch.device('cuda:{}'.format(args.device))
      else:
          args.device = torch.device('cuda:{}'.format(config.device))

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
    args.distributed = config.distributed if args.data_name is None else args.distributed
    if args.distributed:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)

    # Backup all py files and create tensorboard logs
    backup(timeStr, args)
    log_writer_path = './logs/runs/{}'.format('IR-' + timeStr)
    if not os.path.exists(log_writer_path):
      os.makedirs(log_writer_path)
    writer = SummaryWriter(log_writer_path)
    args.envName = config.envName

    if args.data_name is None:
        args.objPath = config.objPath
        args.pointCloud = config.pointCloud
        args.dicPath = config.dicPath
        args.dataSample = config.dataSample

        args.resolutionA = config.resolutionA
        args.resolutionH = config.resolutionH

    else:
        args.objPath = './data/final_data/{}/vhacd_with_pose'.format(args.data_name)
        args.pointCloud = './data/final_data/{}/pointCloud_with_pose'.format(args.data_name)
        if 'IR_mix' in args.data_name:
            args.dicPath = './data/final_data/{}/dicPathHalf.pt'.format(args.data_name)
        else:
            args.dicPath = './data/final_data/{}/dicPath.pt'.format(args.data_name)
        if 'concave' in args.data_name:
            args.dataSample = 'category'
        else:
            args.dataSample = 'instance'

    args.categories = len(torch.load(args.dicPath))

    if args.data_name is None:
        args.bin_dimension = config.bin_dimension
        args.ZRotNum = config.ZRotNum
    else:
        if 'tetris' in args.dicPath:
            args.bin_dimension = [0.32, 0.32, 0.30]
        else:
            args.bin_dimension = [0.32, 0.32, 0.30]
        if 'IR' in args.dicPath:
            args.ZRotNum = 8  # Max: 4/8
        elif 'Box' in args.dicPath:
            args.ZRotNum = 2  # Max: 4/8
        else:
            args.ZRotNum = 4  # Max: 4/8

    args.bin_dimension = np.round(args.bin_dimension, decimals=6)
    args.boundingBoxVec = config.boundingBoxVec
    args.objVecLen = config.objVecLen
    args.load_memory_path = config.load_memory_path
    args.save_memory_path = config.save_memory_path
    args.scale = config.scale
    args.meshScale = config.meshScale
    args.heightResolution = config.heightResolution

    args.selectedAction = config.selectedAction if args.data_name is None else args.selectedAction
    if args.data_name is None:
        args.model = config.model

    args.samplePointsNum = config.samplePointsNum if args.data_name is None else args.samplePointsNum
    assert 'vhacd' in  args.objPath


    if config.originShape:
        args.originDict = load_shape_dict(args, origin = True, scale=args.meshScale)
    args.shapeDict, args.infoDict = load_shape_dict(args, True,  scale=args.meshScale)
    args.physics = True

    args.num_processes = config.num_processes if args.data_name is None else args.num_processes
    args.seed = config.seed
    args.heightMap = config.heightMap
    args.useHeightMap = config.useHeightMap
    args.visual = config.visual
    args.globalView = config.globalView if args.data_name is None else args.globalView
    args.poseDist  = config.poseDist
    args.shotInfo = shotInfoPre(args, args.meshScale)
    args.elementWise = config.elementWise
    args.encoderPath = config.encoderPath
    args.simulation = config.simulation
    args.test = config.test
    args.test_name = config.test_name
    args.hierachical = config.hierachical if args.data_name is None else args.hierachical
    args.previewNum = config.previewNum if args.data_name is None else args.previewNum
    args.shapeArray = shapeProcessing(args.shapeDict, args)

    args.maxBatch = config.maxBatch


    envs, spaces, obs_len = make_vec_envs(args, './logs/runinfo', True)


    args.action_space = spaces[1].n


    args.level = 'location'
    if not args.hierachical:
    # Create the main Agent for IR pack
        dqn = Agent(args)
        memNum = args.num_processes
        memory_capacity = int(args.memory_capacity / memNum)
        mem = [ReplayMemory(args, memory_capacity, obs_len) for _ in range(memNum)]
        trainTool = trainer(writer, timeStr, dqn, mem)

    else:
        args.orderTrain = True
        args.locTrain = True
        orderModelPath = None
        locModelPath = None

        # todo finish the init order policy part
        orderArgs = copy.deepcopy(args)
        orderArgs.action_space = args.previewNum
        orderArgs.model = orderModelPath
        orderArgs.level = 'order'
        orderDQN = Agent(orderArgs)

        locArgs = copy.deepcopy(args)
        locArgs.previewNum = 1
        locArgs.action_space = args.selectedAction
        locArgs.model = locModelPath
        locDQN = Agent(locArgs)

        memNum = args.num_processes
        memory_capacity = int(args.memory_capacity / memNum)
        heightmapSize = np.prod(np.ceil(args.bin_dimension[0:2] / args.resolutionH).astype(np.int32))
        order_obs_len = heightmapSize + args.previewNum
        loc_obs_len = heightmapSize + args.selectedAction * 5 + args.objVecLen
        orderMem = [ReplayMemory(args, memory_capacity, order_obs_len) for _ in
                    range(memNum)] if args.orderTrain else None
        locMem = [ReplayMemory(args, memory_capacity, loc_obs_len) for _ in range(memNum)] if args.locTrain else None

        # Perform all training.
        trainTool = trainer_hierarchical(writer, timeStr, [orderDQN, locDQN], [orderMem, locMem])

    # Perform all training.
    trainTool.train_q_value(envs, args)

if __name__ == '__main__':
    registration_envs()
    args = get_args()
    main(args)

