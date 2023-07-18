import os
import numpy as np
import torch
from tools import get_mask_from_state
from tqdm import trange
from collections import deque
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
np.set_printoptions(threshold=np.inf)
import time

# Distributed training for online packing policy
def learningPara(T, priority_weight_increase, model_save_path, dqn, mem, timeStr, args, counter, lock, sub_time_str):
    log_writer_path = './logs/runs/{}'.format('IR-' + timeStr + '-loss')
    if not os.path.exists(log_writer_path):
      os.makedirs(log_writer_path)
    writer = SummaryWriter(log_writer_path)
    targetCounter = T
    checkCounter = T
    logCounter = T
    timeStep = T
    if args.device.type.lower() != 'cpu':
        torch.cuda.set_device(args.device)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = args.enable_cudnn
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print('Distributed Training Start')
    torch.set_num_threads(1)
    while True:
        if not lock.value:
            for i in range(len(mem)):
                mem[i].priority_weight = min(mem[i].priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

            dqn.reset_noise()
            loss = dqn.learn(mem)  # Train with n-step distributional double-Q learning

            # Update target network
            if timeStep - targetCounter >= args.target_update:
                targetCounter = timeStep
                dqn.update_target_net()

            if timeStep % args.checkpoint_interval == 0:
                    sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

            # Checkpoint the network #
            if (args.checkpoint_interval != 0) and (timeStep - checkCounter >= args.save_interval):
                checkCounter = timeStep
                dqn.save(model_save_path, 'checkpoint{}.pt'.format(sub_time_str))

            if timeStep - logCounter >= args.print_log_interval:
                logCounter = timeStep
                writer.add_scalar("Training/Value loss", loss.mean().item(), logCounter)

            timeStep += 1
        else:
            time.sleep(0.5)

# Distributed training for buffered packing policy
def learningParaHierachical(T, priority_weight_increase, model_save_path, orderDQN, locDQN,
                 orderMem, locMem, timeStr, args, counter, lock, sub_time_str):
    log_writer_path = './logs/runs/{}'.format('IR-' + timeStr + '-loss')
    if not os.path.exists(log_writer_path):
      os.makedirs(log_writer_path)
    writer = SummaryWriter(log_writer_path)
    targetCounter = T
    checkCounter = T
    logCounter = T
    timeStep = T
    print('Distributed Training Start')

    torch.cuda.set_device(args.device)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = args.enable_cudnn
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    bufferNum = len(orderMem) if orderMem is not None else len(locMem)

    while True:
        if not lock.value:
            for i in range(bufferNum):
                orderMem[i].priority_weight = min(orderMem[i].priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1
                locMem[i].priority_weight = min(locMem[i].priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

            orderDQN.reset_noise()
            orderLoss = orderDQN.learn(orderMem)  # Train with n-step distributional double-Q learning

            locDQN.reset_noise()
            locLoss = locDQN.learn(locMem)  # Train with n-step distributional double-Q learning

            # Update target network
            if timeStep - targetCounter >= args.target_update:
                targetCounter = timeStep
                orderDQN.update_target_net()
                locDQN.update_target_net()

            if timeStep % args.checkpoint_interval == 0:
                    sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

            # Checkpoint the network #
            if (args.checkpoint_interval != 0) and (timeStep - checkCounter >= args.save_interval):
                checkCounter = timeStep
                # if checkCounter % args.checkpoint_interval == 0:
                #     sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
                orderDQN.save(model_save_path, 'orderCheckpoint{}.pt'.format(sub_time_str))
                locDQN.save(model_save_path, 'locCheckpoint{}.pt'.format(sub_time_str))

            if timeStep - logCounter >= args.print_log_interval:
                logCounter = timeStep
                writer.add_scalar("Training/Value loss", locLoss.mean().item(), logCounter)
                writer.add_scalar("Training/Order value loss", orderLoss.mean().item(), logCounter)

            timeStep += 1
        else:
            time.sleep(0.5)

class trainer(object):
    def __init__(self, writer, timeStr, dqn, mem):
        self.writer = writer
        self.timeStr = timeStr
        self.dqn = dqn
        self.mem = mem

    def train_q_value(self, envs, args):
        priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if args.save_memory_path is not None:
            memory_save_path = os.path.join(model_save_path, args.save_memory_path)
            if not os.path.exists(memory_save_path):
                os.makedirs(memory_save_path)


        episode_rewards = deque(maxlen=10)
        episode_ratio = deque(maxlen=10)
        episode_counter = deque(maxlen=10)
        state = envs.reset()
        reward_clip = torch.ones((args.num_processes, 1)) * args.reward_clip
        R, loss = 0, 0
        if args.distributed:
            counter= mp.Value('i', 0)
            lock = mp.Value('b', False)
        # Training loop
        self.dqn.train()
        for T in trange(1, args.T_max + 1):

            if T % args.replay_frequency == 0 and not args.distributed:
                self.dqn.reset_noise()  # Draw a new set of noisy weights

            mask = get_mask_from_state(state, args, args.bufferSize)
            action = self.dqn.act(state, mask)  # Choose an action greedily (with noisy weights)


            next_state, reward, done, infos = envs.step(action.cpu().numpy())  # Step

            validSample = []
            for _ in range(len(infos)):
                validSample.append(infos[_]['Valid'])
                if done[_] and infos[_]['Valid']:
                    if 'reward' in infos[_].keys():
                        episode_rewards.append(infos[_]['reward'])
                    else:
                        episode_rewards.append(infos[_]['episode']['r'])
                    if 'ratio' in infos[_].keys():
                        episode_ratio.append(infos[_]['ratio'])
                    if 'counter' in infos[_].keys():
                        episode_counter.append(infos[_]['counter'])


            if args.reward_clip > 0:
                reward = torch.maximum(torch.minimum(reward, reward_clip), -reward_clip)  # Clip rewards

            for i in range(len(state)):
                if validSample[i]:
                    self.mem[i].append(state[i], action[i], reward[i], done[i])  # Append transition to memory

            if args.distributed:
                counter.value = T
                if T == args.learn_start:
                    learningProcess = mp.Process(target=learningPara, args=(T, priority_weight_increase, model_save_path, self.dqn, self.mem, self.timeStr, args, counter, lock, sub_time_str))
                    learningProcess.start()
            else:
                # Train and test
                if T >= args.learn_start:
                    for i in range(len(self.mem)):
                        self.mem[i].priority_weight = min(self.mem[i].priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

                    if T % args.replay_frequency == 0:
                        loss = self.dqn.learn(self.mem)  # Train with n-step distributional double-Q learning
                    # Update target network
                    if T % args.target_update == 0:
                        self.dqn.update_target_net()

                    # Checkpoint the network #
                    if (args.checkpoint_interval != 0) and (T % args.save_interval == 0):
                        if T % args.checkpoint_interval == 0:
                            sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
                        self.dqn.save(model_save_path, 'checkpoint{}.pt'.format(sub_time_str))

                    if T % args.print_log_interval == 0:
                        self.writer.add_scalar("Training/Value loss",  loss.mean().item(), T)

            state = next_state
            if len(episode_rewards)!= 0:
                self.writer.add_scalar('Metric/Reward mean', np.mean(episode_rewards), T)
                self.writer.add_scalar('Metric/Reward max', np.max(episode_rewards), T)
                self.writer.add_scalar('Metric/Reward min', np.min(episode_rewards), T)
            if len(episode_ratio) != 0:
                self.writer.add_scalar('Metric/Ratio', np.mean(episode_ratio), T)
            if len(episode_counter) != 0:
                self.writer.add_scalar('Metric/Length', np.mean(episode_counter), T)

class trainer_hierarchical(object):
    def __init__(self, writer, timeStr, DQNs, MEMs):
        self.writer = writer
        self.timeStr = timeStr
        self.orderDQN, self.locDQN = DQNs
        self.orderMem, self.locMem = MEMs

    def train_q_value(self, envs, args):
        priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
        actionNum = args.action_space
        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if args.save_memory_path is not None:
            memory_save_path = os.path.join(model_save_path, args.save_memory_path)
            if not os.path.exists(memory_save_path):
                os.makedirs(memory_save_path)

        episode_rewards = deque(maxlen=10)
        episode_ratio = deque(maxlen=10)
        episode_counter = deque(maxlen=10)
        orderState = envs.reset()
        reward_clip = torch.ones((args.num_processes, 1)) * args.reward_clip
        R, orderLoss, locLoss = 0, 0, 0
        if args.distributed:
            counter= mp.Value('i', 0)
            lock = mp.Value('b', False)

        self.orderDQN.eval()
        self.locDQN.eval()

        # Training loop
        self.orderDQN.train()
        self.locDQN.train()
        for T in trange(1, args.T_max + 1):

            if T % args.replay_frequency == 0:
                self.orderDQN.reset_noise()  # Draw a new set of noisy weights
                self.locDQN.reset_noise()  # Draw a new set of noisy weights
            orderAction = self.orderDQN.act(orderState, None)
            locState = envs.get_action_candidates(orderAction.cpu().numpy())
            locState = torch.from_numpy(np.array(locState)).float().to(args.device)
            if not args.selectedAction:
                if args.heightMap and not args.physics:
                    locMask = locState[:, 0:args.action_space].reshape(-1, actionNum)
                else:
                    if args.elementWise:
                        locMask = locState[:, (args.packed_holder + 1) * args.objVecLen : (args.packed_holder + 1) * args.objVecLen + actionNum].reshape(-1, actionNum)
                    else:
                        locMask = locState[:, args.objVecLen : args.objVecLen + actionNum].reshape(-1, actionNum)
            else:
                locMask = locState[:, 0 : args.selectedAction * 5].reshape(args.num_processes, args.selectedAction, 5)[:,:,-1]

            locAction = self.locDQN.act(locState, locMask)  # Choose an action greedily (with noisy weights)
            next_order_state, reward, done, infos = envs.step(locAction.cpu().numpy())  # Step

            validSample = []
            for _ in range(len(infos)):
                validSample.append(infos[_]['Valid'])
                if done[_] and infos[_]['Valid']:
                    if 'reward' in infos[_].keys():
                        episode_rewards.append(infos[_]['reward'])
                    else:
                        episode_rewards.append(infos[_]['episode']['r'])
                    if 'ratio' in infos[_].keys():
                        episode_ratio.append(infos[_]['ratio'])
                    if 'counter' in infos[_].keys():
                        episode_counter.append(infos[_]['counter'])

            if args.reward_clip > 0:
                reward = torch.maximum(torch.minimum(reward, reward_clip), -reward_clip)  # Clip rewards

            for i in range(len(orderState)):
                if validSample[i]:
                    self.orderMem[i].append(orderState[i], orderAction[i], reward[i], done[i])  # Append transition to memory
                    self.locMem[i].append(locState[i], locAction[i], reward[i], done[i])  # Append transition to memory

            # todo: sample outside and update priorities uniformly, or maintain their memory seperately
            if args.distributed:
                counter.value = T
                if T == args.learn_start:
                    learningProcess = mp.Process(target=learningParaHierachical, args=(T, priority_weight_increase, model_save_path, self.orderDQN, self.locDQN,
                                                                            self.orderMem, self.locMem, self.timeStr, args, counter, lock, sub_time_str))
                    learningProcess.start()
            else:
                if T >= args.learn_start:
                    for i in range(args.num_processes):
                        self.orderMem[i].priority_weight = min(self.orderMem[i].priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1
                        self.locMem[i].priority_weight = min(self.locMem[i].priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

                    if T % args.replay_frequency == 0:
                        orderLoss = self.orderDQN.learn(self.orderMem)  # Train with n-step distributional double-Q learning
                        locLoss = self.locDQN.learn(self.locMem)  # Train with n-step distributional double-Q learning

                    # Update target network
                    if T % args.target_update == 0:
                        self.orderDQN.update_target_net()
                        self.locDQN.update_target_net()

                    # Checkpoint the network #
                    if (args.checkpoint_interval != 0) and (T % args.save_interval == 0):
                        if T % args.checkpoint_interval == 0:
                            sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
                        self.orderDQN.save(model_save_path, 'orderCheckpoint_{}.pt'.format(sub_time_str))
                        self.locDQN.save(model_save_path, 'locCheckpoint_{}.pt'.format(sub_time_str))

                    if T % args.print_log_interval == 0:
                        self.writer.add_scalar("Training/Value loss",  locLoss.mean().item(), T)
                        self.writer.add_scalar("Training/Order value loss",  orderLoss.mean().item(), T)

            orderState = next_order_state

            if len(episode_rewards)!= 0:
                self.writer.add_scalar('Metric/Reward mean', np.mean(episode_rewards), T)
                self.writer.add_scalar('Metric/Reward max', np.max(episode_rewards), T)
                self.writer.add_scalar('Metric/Reward min', np.min(episode_rewards), T)
            if len(episode_ratio) != 0:
                self.writer.add_scalar('Metric/Ratio', np.mean(episode_ratio), T)
            if len(episode_counter) != 0:
                self.writer.add_scalar('Metric/Length', np.mean(episode_counter), T)

