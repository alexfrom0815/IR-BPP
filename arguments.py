import argparse
import torch
import numpy as np
from tools import load_shape_dict, shotInfoPre, shapeProcessing

def get_args():
    # Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
    parser = argparse.ArgumentParser(description='Rainbow for IR BPP')

    parser.add_argument('--hidden-size', type=int, default=128, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--noisy-std', type=float, default=0.5, metavar='σ',
                        help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--atoms', type=int, default=31, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--V-min', type=float, default=-1, metavar='V', help='Minimum of value distribution support')
    # parser.add_argument('--V-min', type=float, default=0, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=8, metavar='V', help='Maximum of value distribution support')
    # parser.add_argument('--V-max', type=float, default=1, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--target-update', type=int, default=int(1e3), metavar='τ',
                        help='Number of steps after which to update target network')
    parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    parser.add_argument('--reward-clip', type=int, default=0, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, metavar='SIZE', help='Batch size')
    parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
    parser.add_argument('--memory-capacity', type=int, default=int(1e5), metavar='CAPACITY',
                        help='Experience replay memory capacity')
    parser.add_argument('--replay-frequency', type=int, default=4, metavar='k',
                        help='Frequency of sampling from memory')
    parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                        help='Prioritised experience replay exponent (originally denoted α)')
    parser.add_argument('--priority-weight', type=float, default=1.0, metavar='β',
                        help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--id', type=str, default='default', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS',
                        help='Number of training steps (4x number of frames)')
    parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH',
                        help='Max episode length in game frames (0 to disable)')
    parser.add_argument('--history-length', type=int, default=1, metavar='T',
                        help='Number of consecutive states processed')
    parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'dataset-efficient'],
                        metavar='ARCH', help='Network architecture')
    parser.add_argument('--load-model', action='store_true', help='Load the trained model')
    parser.add_argument('--learn-start', type=int, default=int(5e2), metavar='STEPS',
                        help='Number of steps before starting training')

    parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS',
                        help='Number of training steps between evaluations')
    parser.add_argument('--evaluation-episodes', type=int, default=100, metavar='N',
                        help='Number of evaluation episodes to average over')
    # TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
    parser.add_argument('--evaluation-size', type=int, default=500, metavar='N',
                        help='Number of transitions to use for validating Q')

    parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
    parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')

    parser.add_argument('--checkpoint-interval', default=10000,
                        help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
    parser.add_argument('--save-interval', default=1000, help='How often to save the model.')
    parser.add_argument('--model-save-path',type=str, default='./logs/experiment', help='The path to save the trained model')

    parser.add_argument('--disable-bzip-memory', action='store_true',
                        help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
    parser.add_argument('--print-log-interval',     type=int,   default=10, help='How often to print training logs')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')

    # some ir packing settings
    parser.add_argument('--envName', type=str, default='Physics-v0')
    parser.add_argument('--dataset', type=str, default='blockout') # blockout general kitchen abc
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--custom', type=str, default=None)
    parser.add_argument('--hierachical', action='store_true')
    parser.add_argument('--bufferSize', type=int, default=1) # 1 3 5 10
    parser.add_argument('--num_processes', type=int, default=2) # 16 1
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--samplePointsNum', type=int, default=1024)
    parser.add_argument('--selectedAction', type=int, default=500) # defalt 500
    parser.add_argument('--maxBatch', type=int, default=2) # how many batches for simulation
    parser.add_argument('--visual', action='store_true', help='Render the scene')
    parser.add_argument('--resolutionA', type=float, default = 0.02)
    parser.add_argument('--resolutionH', type=float, default = 0.01)
    parser.add_argument('--resolutionZ', type=float, default = 0.01)

    parser.add_argument('--locmodel', type=str, default=None)
    parser.add_argument('--ordmodel', type=str, default=None)

    parser.add_argument('--only_simulate_current', action='store_true', help='Only simulate the current item')

    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation_episodes', type=int, default=2000)


    args = parser.parse_args()
    print('first hierachical',args.hierachical)
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
      print(' ' * 26 + k + ': ' + str(v))


    args.objPath = './dataset/{}/shape_vhacd'.format(args.dataset)
    args.pointCloud = './dataset/{}/pointCloud'.format(args.dataset)
    args.dicPath = './dataset/{}/id2shape.pt'.format(args.dataset)

    if  args.dataset == 'kitchen':
        args.dataSample = 'category'
    else:
        args.dataSample = 'instance'

    args.categories = len(torch.load(args.dicPath))
    args.bin_dimension = np.round([0.32, 0.32, 0.30], decimals=6)

    args.ZRotNum = 8  # Max: 4/8
    if  args.dataset == 'blockout':
        args.ZRotNum = 4  # Max: 4/8
    elif 'box' in args.dataset:
        args.ZRotNum = 2  # Max: 4/8
    else:
        assert False

    args.objVecLen = 9
    args.model = None
    args.load_memory_path = None
    args.save_memory_path = None
    args.scale =  [100, 100, 100] # fix it! don't change it!
    args.meshScale = 1
    args.heightResolution = args.resolutionZ
    args.shapeDict, args.infoDict = load_shape_dict(args, True, scale=args.meshScale)
    args.physics = True
    args.heightMap = True
    args.useHeightMap = True
    args.globalView = True if args.evaluate else False
    args.shotInfo = shotInfoPre(args, args.meshScale)
    args.simulation = True
    args.distributed = True
    args.test_name = './dataset/{}/test_sequence.pt'.format(args.dataset)
    args.shapeArray = shapeProcessing(args.shapeDict, args)

    if args.evaluate:
        args.num_processes = 1

    # temp setting
    # args.evaluate = True

    return args
