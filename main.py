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
from tools import backup, registration_envs, load_memory, load_shape_dict, shotInfoPre, shapeProcessing
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
    # custom = 'debug'
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
        print('args.dicPath', args.dicPath)

    # args.objPath = config.objPath if args.data_name is None else args.objPath
    args.enable_rotation = config.enable_rotation
    args.categories = len(torch.load(args.dicPath))

    if args.data_name is None:
        args.bin_dimension = config.bin_dimension
        args.ZRotNum = config.ZRotNum
    else:
        if 'BlockL' in args.dicPath:
            args.bin_dimension = [0.32, 0.32, 0.035]
        elif 'tetris' in args.dicPath:
            args.bin_dimension = [0.32, 0.32, 0.30]
        else:
            args.bin_dimension = [0.32, 0.32, 0.30]
        if 'IR' in args.dicPath:
            args.ZRotNum = 8  # Max: 4/8
        elif 'Box' in args.dicPath:
            args.ZRotNum = 2  # Max: 4/8
        else:
            args.ZRotNum = 4  # Max: 4/8
        if args.doubleRot:
            args.ZRotNum *= 2

    args.bin_dimension = np.round(args.bin_dimension, decimals=6)
    args.packed_holder = config.packed_holder
    args.boxPack = config.boxPack
    args.boundingBoxVec = config.boundingBoxVec
    args.DownRotNum = config.DownRotNum
    args.boxset = config.boxset
    args.triangleNum = config.triangleNum
    args.objVecLen = config.objVecLen
    # args.dicPath = config.dicPath
    args.dataAugmentation = config.dataAugmentation
    args.heuristicExplore = config.heuristicExplore
    args.load_memory_path = config.load_memory_path
    args.save_memory_path = config.save_memory_path
    args.scale = config.scale
    args.meshScale = config.meshScale
    args.heightResolution = config.heightResolution

    args.actionType = config.actionType if args.data_name is None else args.actionType
    args.selectedAction = config.selectedAction if args.data_name is None else args.selectedAction
    if args.selectedAction: assert  args.actionType == 'Uniform'
    if args.data_name is None:
        args.model = config.model
    args.convexAction = config.convexAction if args.data_name is None else args.convexAction
    args.samplePointsNum = config.samplePointsNum if args.data_name is None else args.samplePointsNum
    # args.dataSample = config.dataSample if args.data_name is None else args.dataSample
    assert 'vhacd' in  args.objPath

    if args.envName != "Physics-v0":
        args.shapeDict = load_shape_dict(args)
        args.physics = False
    else:
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
    args.stability = config.stability
    args.poseDist  = config.poseDist
    args.shotInfo = shotInfoPre(args, args.meshScale)
    args.rewardType = config.rewardType
    args.elementWise = config.elementWise
    args.encoderPath = config.encoderPath
    # args.pointCloud = config.pointCloud if args.data_name is None else args.pointCloud
    args.simulation = config.simulation
    args.shapePreType = config.shapePreType
    args.test = config.test
    args.test_name = config.test_name
    args.hierachical = config.hierachical if args.data_name is None else args.hierachical
    args.previewNum = config.previewNum if args.data_name is None else args.previewNum
    args.shapeArray = shapeProcessing(args.shapeDict, args)

    args.maxBatch = config.maxBatch
    args.randomConvex = config.randomConvex
    args.LFSS = config.LFSS

    envs, spaces, obs_len = make_vec_envs(args, './logs/runinfo', True)
    if args.shapePreType == 'GlobalIndices':
        args.globalIndices = np.random.randint(100000, size=args.samplePointsNum)
    args.action_space = spaces[1].n


    args.level = 'location'
    if not args.hierachical:
    # Create the main Agent for IR pack
        if config.originShape:
            dqn = Agent(args)
        else:
            dqn = Agent(args)

        # for buffer fixed indices
        # if args.shapePreType == 'SurfacePointsRandom' or args.shapePreType == 'SurfacePointsEncode':
        #     obs_len += args.samplePointsNum

        if args.load_memory_path is not None: # load exsiting memeory dir
            mem = []
            for i in range(args.num_processes):
                memory_path = os.path.join(args.memory, 'memory{}'.format(i))
                mem.append(load_memory(memory_path, args.disable_bzip_memory))
        else:
            memNum = args.num_processes * 4 if args.dataAugmentation else args.num_processes
            memory_capacity = int(args.memory_capacity / memNum)
            mem = [ReplayMemory(args, memory_capacity, obs_len) for _ in range(memNum)]
        trainTool = trainer(writer, timeStr, dqn, mem)

    else:
        args.orderTrain = True
        args.locTrain = True
        # orderModelPath = './checkpoints/preOrderModel.pt'
        orderModelPath = None
        # orderModelPath = './logs/experiment/IR_concaveArea3_mass_hier_10-2022.08.02-22-23-07/orderCheckpoint29.pt'
        # locModelPath = './checkpoints/preLocModel.pt'
        locModelPath = None
        # locModelPath = './logs/experiment/IR_concaveArea3_mass_hier_10-2022.08.02-22-23-07/locCheckpoint29.pt'

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

        memNum = args.num_processes * 4 if args.dataAugmentation else args.num_processes
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

