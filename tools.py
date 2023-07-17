#--coding:utf-8--
import copy

import numpy as np
import trimesh
import os
from shutil import copyfile, copytree
from gym.envs.registration import register
import bz2
import pickle
import torch
from matplotlib import pyplot as plt
import transforms3d
import pybullet as p
import gym


def load_mesh_plain(path,  ZRotNum, init = 'Centroid', scale = 1):
    mesh = trimesh.load(path)
    # print('len', len(mesh.vertices))
    if scale != 1:
        mesh.apply_scale(scale)
    mesh.apply_translation(- mesh.centroid)
    meshList = []
    DownFaceList, ZRotList = getRotationMatrix(1, ZRotNum)

    for d in DownFaceList:
        for z in ZRotList:
            tmpObj = mesh.copy()
            Transform = np.dot(z, d)
            tmpObj.apply_transform(Transform)

            if init == 'BoundingBox':  # Place the front-left-bottom point of object bounding box at origin.
                mesh.apply_translation(- mesh.bounds[0])
            else:
                assert False
            meshList.append(tmpObj)

    return meshList

def extendMat(mat3, translation = None):
    mat4 = np.eye(4)
    mat4[0:3,0:3] = mat3
    if translation is not None:
        mat4[0:3,3] = translation
    return mat4

def getRotationMatrix(DownRotNum, ZRotNum):
    DownRotNum = 1
    Tx00  = extendMat(transforms3d.euler.euler2mat(0, 0, 0, 'sxyz'))
    Tx180 = extendMat(transforms3d.euler.euler2mat(np.pi, 0, 0, 'sxyz'))
    Tx90  = extendMat(transforms3d.euler.euler2mat(np.pi * 0.5, 0, 0, 'sxyz'))
    Tx_90 = extendMat(transforms3d.euler.euler2mat(np.pi * - 0.5, 0, 0, 'sxyz'))
    Ty90  = extendMat(transforms3d.euler.euler2mat(0, np.pi * 0.5, 0, 'sxyz'))
    Ty_90 = extendMat(transforms3d.euler.euler2mat(0, np.pi * - 0.5, 0, 'sxyz'))
    DownFaceList = [Tx00, Tx180, Tx90, Tx_90, Ty90, Ty_90]

    Tz00  = extendMat(transforms3d.euler.euler2mat(0, 0, 0, 'sxyz'))
    Tz90  = extendMat(transforms3d.euler.euler2mat(0, 0, np.pi * 0.5, 'sxyz'))
    Tz180 = extendMat(transforms3d.euler.euler2mat(0, 0, np.pi * 1, 'sxyz'))
    Tz270 = extendMat(transforms3d.euler.euler2mat(0, 0, np.pi * 1.5, 'sxyz'))
    Tz45  = extendMat(transforms3d.euler.euler2mat(0, 0,  np.pi * 0.25, 'sxyz'))
    Tz135 = extendMat(transforms3d.euler.euler2mat(0, 0, np.pi * 0.75, 'sxyz'))
    Tz225 = extendMat(transforms3d.euler.euler2mat(0, 0, np.pi * 1.25, 'sxyz'))
    Tz315 = extendMat(transforms3d.euler.euler2mat(0, 0, np.pi * 1.75, 'sxyz'))

    Tz22_5  = extendMat(transforms3d.euler.euler2mat(0, 0,   np.pi * 0.125, 'sxyz'))
    Tz67_5  = extendMat(transforms3d.euler.euler2mat(0, 0,   np.pi * 0.375, 'sxyz'))
    Tz112_5  = extendMat(transforms3d.euler.euler2mat(0, 0,  np.pi * 0.625, 'sxyz'))
    Tz157_5  = extendMat(transforms3d.euler.euler2mat(0, 0,  np.pi * 0.875, 'sxyz'))
    Tz202_5  = extendMat(transforms3d.euler.euler2mat(0, 0,  np.pi * 1.125, 'sxyz'))
    Tz247_5  = extendMat(transforms3d.euler.euler2mat(0, 0,  np.pi * 1.375, 'sxyz'))
    Tz292_5  = extendMat(transforms3d.euler.euler2mat(0, 0,  np.pi * 1.625, 'sxyz'))
    Tz337_5  = extendMat(transforms3d.euler.euler2mat(0, 0,  np.pi * 1.875, 'sxyz'))

    ZRotList = [Tz00, Tz90, Tz180, Tz270, Tz45, Tz135, Tz225, Tz315,
                Tz22_5, Tz67_5, Tz112_5, Tz157_5, Tz202_5, Tz247_5, Tz292_5,Tz337_5]

    return DownFaceList[0:int(DownRotNum)], ZRotList[0:int(ZRotNum)]

def gen_ray_origin_direction(xRange, yRange, resolution_h, boxPack = False, shift = 0.001):

    bottom = np.arange(0, xRange * yRange)
    bottom = bottom.reshape((xRange, yRange))

    origin = np.zeros((xRange, yRange, 3))
    origin[:, :, 0] = bottom // yRange * resolution_h + shift
    origin[:, :, 1] = bottom %  yRange * resolution_h + shift
    origin[:, :, 2] = -10e2
    origin = origin.reshape((xRange, yRange, 3))

    direction = np.zeros_like(origin)
    direction[:,:,2] = 1

    return origin, direction


def shot_item(mesh, ray_origins_ini, ray_directions_ini, xRange = 20, yRange = 20, start = [0,0,0]): # xRange, yRange the grid range.
    mesh = mesh.copy()
    mesh.apply_translation(- mesh.bounding_box.vertices[0])

    heightMapB = np.zeros(xRange * yRange)
    heightMapH = np.zeros(xRange * yRange)
    maskB = np.zeros(xRange * yRange)
    maskH = np.zeros(xRange * yRange)

    ray_origins = ray_origins_ini[start[0] : start[0] + xRange, start[1] : start[1] + yRange].copy().reshape((-1,3))
    ray_directions = ray_directions_ini[start[0] : start[0] + xRange, start[1] : start[1] + yRange].copy().reshape(-1,3)
    index_triB, index_rayB, locationsB = mesh.ray.intersects_id( ray_origins=ray_origins, ray_directions=ray_directions,
                                                                 return_locations=True,   multiple_hits=False)

    if len(index_rayB) != 0:
        heightMapB[index_rayB] = locationsB[:, 2]
        maskB[index_rayB] = 1
    else:
        heightMapB[:] = 0
        maskB[:] = 1
    heightMapB = heightMapB.reshape((xRange, yRange))
    maskB = maskB.reshape((xRange, yRange))

    ray_origins[:, 2] *= -1
    ray_directions[:, 2] *= -1
    # print(np.concatenate((ray_origins, ray_origins + ray_directions), axis=1))
    index_triH, index_rayH, locationsH = mesh.ray.intersects_id( ray_origins=ray_origins, ray_directions=ray_directions,
                                                                 return_locations=True,   multiple_hits=False)
    if len(index_rayH) != 0:
        heightMapH[index_rayH] = locationsH[:, 2]
        maskH[index_rayH] = 1
    else:
        heightMapH[:] = mesh.extents[2]
        maskH[:] = 1
    heightMapH = heightMapH.reshape((xRange, yRange))
    maskH = maskH.reshape((xRange, yRange))

    return heightMapH, heightMapB, maskH, maskB

def shot_after_item_placement(mesh, ray_origins_ini, ray_directions_ini, xRange = 20, yRange = 20, start = [0,0,0]): # xRange, yRange the grid range.
    mesh = mesh.copy()

    heightMapH = np.zeros(xRange * yRange)
    maskH = np.zeros(xRange * yRange)

    ray_origins = ray_origins_ini[start[0] : start[0] + xRange, start[1] : start[1] + yRange].copy().reshape((-1,3))
    ray_directions = ray_directions_ini[start[0] : start[0] + xRange, start[1] : start[1] + yRange].copy().reshape(-1,3)

    ray_origins[:, 2] *= -1
    ray_directions[:, 2] *= -1

    index_triH, index_rayH, locationsH = mesh.ray.intersects_id( ray_origins=ray_origins, ray_directions=ray_directions,
                                                                 return_locations=True,   multiple_hits=False)
    if len(index_rayH) != 0:
        heightMapH[index_rayH] = locationsH[:, 2]
        maskH[index_rayH] = 1
    else:
        heightMapH[:] = mesh.extents[2]
        maskH[:] = 1

    heightMapH = heightMapH.reshape((xRange, yRange))
    maskH = maskH.reshape((xRange, yRange))

    return heightMapH, maskH

def backup(timeStr, args):
    if args.evaluate:
        targetDir = os.path.join('./logs/evaluation', timeStr)
    else:
        targetDir = os.path.join('./logs/experiment', timeStr)

    if not os.path.exists('./logs/runinfo'):
        os.makedirs('./logs/runinfo')

    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')

    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    copyfile('agent.py',  os.path.join(targetDir, 'agent.py'))
    copyfile('arguments.py',  os.path.join(targetDir, 'arguments.py'))
    copyfile('envs.py',    os.path.join(targetDir, 'envs.py'))
    copyfile('main.py',   os.path.join(targetDir, 'main.py'))
    copyfile('model.py',   os.path.join(targetDir, 'model.py'))
    copyfile('tools.py', os.path.join(targetDir, 'tools.py'))
    copyfile('trainer.py', os.path.join(targetDir, 'trainer.py'))
    copyfile('memory.py', os.path.join(targetDir, 'memory.py'))

    gymPath = './environment'
    envName = args.envName.split('-v')
    envName = envName[0].lower() + envName[1]
    envPath = os.path.join(gymPath, envName)
    copytree(envPath, os.path.join(targetDir, envName))

def init(module, weight_init, bias_init, gain=1):
      weight_init(module.weight.data, gain=gain)
      bias_init(module.bias.data)
      return module

def registration_envs():
    register(
        id='Physics-v0',                                  # Format should be xxx-v0, xxx-v1
        entry_point='environment.physics0:PackingGame',   # Expalined in envs/__init__.py
    )

# Visualize each heightMap with colormap.
def draw_heatmap(heightMap, vmin = 0, vmax = 0.3):
    # print(heightMap)
    plt.imshow(heightMap,  cmap=plt.cm.hot, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()

# Transfer a mesh into triangle representation.
def shapeProcessing(shapeDict, args):
    if shapeDict is None:
        return None
    else:
        shapeDict = torch.load(args.dicPath)
        pointCloudPath = args.pointCloud

        pointsNum = 100000
        shapeArray = np.zeros((len(shapeDict), pointsNum, 3))
        for shapeIdx in shapeDict.keys():
            # dataset = shapeDict[shapeIdx].replace('.obj', '.npz')
            data = shapeDict[shapeIdx][0:-4] + '.npz'
            data = np.load(os.path.join(pointCloudPath, data))['points']
            shapeArray[shapeIdx] = data[0:pointsNum]
        return shapeArray

def load_shape_dict(args, returnInfo = False, origin = False, scale = 1):
    backDict = {}
    infoDict = {}
    dicPath = args.dicPath
    objPath = args.objPath if not origin else args.objPath.replace('_vhacd', '')
    print('Load objects from:', objPath)
    shapeDict = torch.load(dicPath)
    for k in shapeDict.keys():
        if k >= args.categories: break

        loadPath = os.path.join(objPath, shapeDict[k])
        backDict[k] = load_mesh_plain(loadPath,  args.ZRotNum, 'BoundingBox', scale)
        infoDict[k] = []

        for idx in range(len(backDict[k])):
            infoDict[k].append({'volume': backDict[k][idx].volume, 'extents': backDict[k][idx].extents})
    if returnInfo:
        return backDict, infoDict
    else:
        return backDict

def shotInfoPre(args, meshScale = 1):
    shapeDict = args.shapeDict
    rangeX_C = int(np.ceil(args.bin_dimension[0] / args.resolutionH))
    rangeY_C = int(np.ceil(args.bin_dimension[1] / args.resolutionH))
    ray_origins, ray_directions = gen_ray_origin_direction(rangeX_C, rangeY_C, args.resolutionH, False)
    shotInfo = {}
    data_name = args.objPath.split('/')[-2]
    dicPath = args.dicPath.replace('.pt', '')
    dicPath = dicPath.split('/')[-1]
    if meshScale != 1:
        dataStorePath = os.path.join('dataset/shotInfo', '{}_{}_{}_{}'.format(data_name, dicPath, args.resolutionH, meshScale))
    else:
        dataStorePath = os.path.join('dataset/shotInfo', '{}_{}_{}'.format(data_name, dicPath, args.resolutionH))
    if not os.path.exists(dataStorePath):
        os.makedirs(dataStorePath)
    for k in shapeDict.keys():
        if k >= args.categories:
            break
        next_item = shapeDict[k]
        shotInfo[k] = []
        for rotIdx in range(len(next_item)):
            boundingSize = np.round(next_item[rotIdx].extents, decimals=6)
            rangeX_O, rangeY_O = np.ceil(boundingSize[0:2] / args.resolutionH).astype(np.int32)
            subdataPath = os.path.join(dataStorePath, '{}_{}.pt'.format(k, rotIdx))
            if os.path.exists(subdataPath):
                heightMapT, heightMapB, maskH, maskB = torch.load(subdataPath)
            else:
                heightMapT, heightMapB, maskH, maskB = shot_item(next_item[rotIdx], ray_origins,
                                                             ray_directions, rangeX_O, rangeY_O)
                torch.save([heightMapT, heightMapB, maskH, maskB], subdataPath)
            shotInfo[k].append((heightMapT, heightMapB, maskH, maskB))
    return shotInfo



def get_mask_from_state(state, args, bufferSize):
    actionNum = args.action_space

    if bufferSize > 1:
        mask = None
    else:
        if not args.selectedAction:
            if args.heightMap and not args.physics:
                mask = state[:, 0:args.action_space].reshape(-1, actionNum)
            else:
                if args.elementWise:
                    mask = state[:, (args.packed_holder + 1) * args.objVecLen: (args.packed_holder + 1) * args.objVecLen + actionNum].reshape(-1, actionNum)
                else:
                    mask = state[:, args.objVecLen: args.objVecLen + actionNum].reshape(-1, actionNum)
        else:
            mask = state[:, 0: args.selectedAction * 5]
            mask = mask.reshape(-1, args.selectedAction, 5)[:, :, -1]
    return mask

# Test DQN
def test(args, dqn, printInfo = False, timeStr = None, times = ''):
    env = make_eval_env(args)
    T_rewards, T_lengths, T_ratio, T_ratio_local = [], [], [], []
    all_episodes = []
    print('Evaluation Start')
    # Test performance over several episodes
    done = True
    dqn.online_net.eval()
    assert not dqn.online_net.training

    for _ in range(args.evaluation_episodes_test):
        while True:
            if done:
                state, reward_sum, done, episode_length = env.reset(), 0, False, 0
            state = torch.FloatTensor(state).reshape((1, -1)).to(args.device)
            mask = get_mask_from_state(state, args, args.bufferSize)
            action = dqn.act_e_greedy(state, mask, -1)
            state, reward, done, _ = env.step(action.item())  # Step


            reward_sum += reward
            episode_length += 1

            if done:
                ratio = env.get_ratio()
                T_ratio.append(ratio)
                T_rewards.append(reward_sum)
                T_lengths.append(episode_length)
                if printInfo:
                    print('avg_reward:', np.mean(T_rewards))
                    print('avg_length:', np.mean(T_lengths))
                    print('var_reward:', np.var(T_rewards))
                    print('var_length:', np.var(T_lengths))
                    print('Mean Ratio:', np.mean(T_ratio))
                    print('Var Ratio:', np.var(T_ratio))
                    print('Episode {} Ratio {}'.format(env.item_creator.traj_index, reward_sum))
                all_episodes.append(copy.deepcopy( env.packed))
                np.save(os.path.join('./logs/evaluation', timeStr, 'trajs{}.npy'.format(times)), all_episodes)
                break
    env.close()

    avg_reward= np.mean(T_rewards)
    avg_length= np.mean(T_lengths)
    print('avg_reward:', avg_reward)
    print('avg_length:', avg_length)
    print('var_reward:', np.var(T_rewards))
    print('var_length:', np.var(T_lengths))
    print('Mean Ratio:', np.mean(T_ratio))
    print('Var Ratio:', np.var(T_ratio))
    if not os.path.exists(os.path.join('./logs/evaluation', timeStr)):
        os.makedirs(os.path.join('./logs/evaluation', timeStr))
    np.save(os.path.join('./logs/evaluation', timeStr, 'trajs{}.npy'.format(times)), all_episodes)
    dqn.online_net.train()
    assert dqn.online_net.training
    # Return average reward and Q-value
    return avg_reward, avg_length

# Test DQN
def test_hierachical(args, dqns, printInfo = False, timeStr = None, times = ''):
    env = make_eval_env(args)
    T_rewards, T_lengths, T_ratio, T_ratio_local = [], [], [], []
    all_episodes = []
    print('Evaluation Start')
    done = True
    for dqn in dqns:
        dqn.online_net.eval()
        assert not dqn.online_net.training
    orderDQN, locDQN = dqns

    placementCounter = 0

    for _ in range(args.evaluation_episodes_test):
        while True:
            if done:
                orderState, reward_sum, done, episode_length = env.reset(), 0, False, 0
            orderState = torch.FloatTensor(orderState).reshape((1, -1)).to(args.device)

            orderAction = orderDQN.act(orderState, None)

            locState = env.get_action_candidates(orderAction.cpu().numpy().astype(np.int)[0] if len(orderAction.shape) > 0 else orderAction.item())
            locState = torch.from_numpy(np.array(locState)).float().to(args.device).reshape((1, -1))
            mask = get_mask_from_state(locState, args, 1)
            locAction = locDQN.act_e_greedy(locState, mask, -1)

            orderState, reward, done, _ = env.step(locAction.item())  # Step

            placementCounter += 1

            reward_sum += reward
            episode_length += 1

            if done:
                ratio = env.get_ratio()
                T_ratio.append(ratio)
                T_rewards.append(reward_sum)
                T_lengths.append(episode_length)
                all_episodes.append(copy.deepcopy( env.packed))
                if printInfo:
                    print('avg_reward:', np.mean(T_rewards))
                    print('avg_length:', np.mean(T_lengths))
                    print('var_reward:', np.var(T_rewards))
                    print('var_length:', np.var(T_lengths))

                    print('Mean Ratio:', np.mean(T_ratio))
                    print('Var Ratio:', np.var(T_ratio))
                    print('Episode {} Ratio {}'.format(env.item_creator.traj_index, reward_sum))

                np.save(os.path.join('./logs/evaluation', timeStr, 'trajs{}.npy'.format(times)), all_episodes)
                break
    env.close()

    avg_reward= np.mean(T_rewards)
    avg_length= np.mean(T_lengths)
    print('avg_reward:', avg_reward)
    print('avg_length:', avg_length)
    print('var_reward:', np.var(T_rewards))
    print('var_length:', np.var(T_lengths))
    print('Mean Ratio:', np.mean(T_ratio))
    print('Mean Ratio Local:', np.mean(T_ratio_local))
    print('Var Ratio:', np.var(T_ratio))
    print('Var Ratio Local:', np.var(T_ratio_local))
    if not os.path.exists(os.path.join('./logs/evaluation', timeStr)):
        os.makedirs(os.path.join('./logs/evaluation', timeStr))
    np.save(os.path.join('./logs/evaluation', timeStr, 'trajs{}.npy'.format(times)), all_episodes)
    for dqn in dqns:
        dqn.online_net.train()
        assert dqn.online_net.training
    # Return average reward and Q-value
    return avg_reward, avg_length

def make_eval_env(args):
    env = gym.make(args.envName,
                   args = args)
    return env
