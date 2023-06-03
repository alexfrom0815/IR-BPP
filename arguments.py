import argparse

def get_args():
    # Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
    parser = argparse.ArgumentParser(description='Rainbow')

    # A smaller width is better.
    parser.add_argument('--hidden-size', type=int, default=128, metavar='SIZE', help='Network hidden size')

    # 0.1 is suitable. √
    parser.add_argument('--noisy-std', type=float, default=0.5, metavar='σ',
    # parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ',
    # parser.add_argument('--noisy-std', type=float, default=1, metavar='σ',
                        help='Initial standard deviation of noisy linear layers')
    # 31 performs well
    parser.add_argument('--atoms', type=int, default=31, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--V-min', type=float, default=-1, metavar='V', help='Minimum of value distribution support')
    # parser.add_argument('--V-min', type=float, default=0, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=8, metavar='V', help='Maximum of value distribution support')
    # parser.add_argument('--V-max', type=float, default=1, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--target-update', type=int, default=int(1e3), metavar='τ',
                        help='Number of steps after which to update target network')
    # For GAE implementation.
    parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    # 1 perform better than 10.
    parser.add_argument('--reward-clip', type=int, default=0, metavar='VALUE', help='Reward clipping (0 to disable)')
    # The default value perfroms better than 1e-3 √
    parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, metavar='SIZE', help='Batch size')
    # 64 is ok for now
    # parser.add_argument('--batch-size', type=int, default=128, metavar='SIZE', help='Batch size')

    # 10 performs better than 1.
    parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
    # parser.add_argument('--norm-clip', type=float, default=1, metavar='NORM', help='Max L2 norm for gradient clipping')

    # The larger the better
    parser.add_argument('--memory-capacity', type=int, default=int(1e5), metavar='CAPACITY',
                        help='Experience replay memory capacity')
    # 4 performs better than 2.
    parser.add_argument('--replay-frequency', type=int, default=4, metavar='k',
                        help='Frequency of sampling from memory')
    parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                        help='Prioritised experience replay exponent (originally denoted α)')
    # The original value is 0.4, now is changed to 1.0.
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
                        help='Number of consecutive states processed') # 这里是几叠帧的意思，不是N步更新的意思
    parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'],
                        metavar='ARCH', help='Network architecture')
    parser.add_argument('--load-model', action='store_true', help='Load the trained model')
    # parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--learn-start', type=int, default=int(5e2), metavar='STEPS',
                        help='Number of steps before starting training')

    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS',
    # parser.add_argument('--evaluation-interval', type=int, default=600, metavar='STEPS',
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

    # added for zherong pan
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--custom', type=str, default=None) # very important and cannot be saved, dafalt value must be none
    parser.add_argument('--data_name', type=str, default=None) # very important and cannot be saved, dafalt value must be none
    parser.add_argument('--hierachical', action='store_true')
    parser.add_argument('--previewNum', type=int, default=1) # 1 3 5 10
    parser.add_argument('--actionType', type=str, default='Uniform')  # Uniform, RotAction, LineAction, HeuAction
    parser.add_argument('--num_processes', type=int, default=16) # 16 1
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--samplePointsNum', type=int, default=1024)
    parser.add_argument('--convexAction', type=str, default=None)
    parser.add_argument('--selectedAction', type=int, default=False)
    parser.add_argument('--globalView', type=bool, default = False)
    parser.add_argument('--resolutionA', type=float, default = 0.02)
    parser.add_argument('--resolutionH', type=float, default = 0.01)
    parser.add_argument('--doubleRot', action='store_true')
    parser.add_argument('--model', type=str, default=None)

    args = parser.parse_args()
    print('first hierachical',args.hierachical)
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
      print(' ' * 26 + k + ': ' + str(v))

    return args
