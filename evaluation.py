# -*- coding: utf-8 -*-
import torch
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import gym
from arguments import get_args
from agent import Agent
from tools import registration_envs, load_shape_dict, shotInfoPre,\
    test, test_hierachical, make_eval_env, shapeProcessing, backup
import numpy as np
import testconfig as config
import time
import copy



if __name__ == '__main__':
    custom = input('Please input the test name\n')
    timeStr = custom + '-' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

    # custom = 'debug'
    registration_envs()
    args = get_args()
    args.model = config.model

    if torch.cuda.is_available() and not args.disable_cuda and isinstance(config.device, int):
      args.device = torch.device('cuda:{}'.format(config.device))
      torch.cuda.manual_seed(np.random.randint(1, 10000))
      torch.backends.cudnn.enabled = args.enable_cudnn
    else:
      args.device = torch.device('cpu')

    if args.device.type.lower() != 'cpu':
        torch.cuda.set_device(config.device)

    args.envName = config.envName
    args.objPath = config.objPath
    args.resolutionA = config.resolutionA
    args.enable_rotation = config.enable_rotation
    args.categories = config.categories
    args.bin_dimension = config.bin_dimension
    args.packed_holder = config.packed_holder
    args.boxPack = config.boxPack
    args.boundingBoxVec = config.boundingBoxVec
    args.DownRotNum = config.DownRotNum
    args.ZRotNum = config.ZRotNum
    args.boxset = config.boxset
    args.triangleNum = config.triangleNum
    args.objVecLen = config.objVecLen
    args.distributed = config.distributed
    args.dicPath = config.dicPath
    args.scale = config.scale
    args.meshScale = config.meshScale

    args.selectedAction = config.selectedAction
    args.convexAction = config.convexAction
    args.previewNum = config.previewNum
    args.shapePreType = config.shapePreType
    args.samplePointsNum = config.samplePointsNum
    args.dataSample = config.dataSample
    args.hierachical = config.hierachical
    args.heightResolution = config.heightResolution

    args.test_name = config.test_name
    args.evaluate  = True
    backup(timeStr, args)

    if args.envName != "Physics-v0":
        args.shapeDict = load_shape_dict(args)
        args.physics = False
    else:
        if config.originShape:
            args.originDict = load_shape_dict(args, origin = True, scale=args.meshScale)
        args.shapeDict, args.infoDict = load_shape_dict(args, True, scale=args.meshScale)
        args.physics = True

    args.seed = config.seed
    args.heightMap = config.heightMap
    args.useHeightMap = config.useHeightMap
    args.visual = config.visual
    args.resolutionH = config.resolutionH
    args.globalView = config.globalView
    args.stability = config.stability
    args.poseDist  = config.poseDist
    args.shotInfo = shotInfoPre(args, args.meshScale)
    args.evaluation_episodes = config.evaluation_episodes
    args.rewardType = config.rewardType
    args.actionType = config.actionType
    args.elementWise = config.elementWise
    args.encoderPath = config.encoderPath
    args.pointCloud = config.pointCloud
    args.simulation = config.simulation
    args.samplePointsNum = config.samplePointsNum
    args.shapeArray = shapeProcessing(args.shapeDict, args)
    args.maxBatch = 2
    args.select_item_with_one_dqn = config.select_item_with_one_dqn
    args.timeStr = timeStr
    env = make_eval_env(args)

    videoName = './video/{}.mp4'.format(timeStr) if config.video else None

    args.action_space = env.act_len
    shapeDict = env.shapeDict
    args.packed_holder = config.packed_holder
    args.boundingBoxVec = config.boundingBoxVec
    args.level = 'location'

    if not args.hierachical:
        if config.originShape:
            dqn = Agent(args)
        else:
            dqn = Agent(args)
        env.close()
        del env
        avg_reward= test(args, dqn,  True, videoName, timeStr)  # Test
        # avg_reward= test_with_given_traj(args, dqn,  True, videoName, timeStr)  # Test
    else:
        # orderModelPath = './checkpoints/orderCheckpoint16.pt'
        # locModelPath = './checkpoints/locCheckpoint16.pt'
        orderModelPath = config.orderModelPath
        locModelPath = config.locModelPath

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

        env.close()
        del env
        avg_reward = test_hierachical(args, [orderDQN, locDQN], True, videoName, timeStr)  # Test

