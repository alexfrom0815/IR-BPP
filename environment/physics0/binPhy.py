import time
import gym
import numpy as np
from torch import load
from tools import getRotationMatrix
import transforms3d
from .Interface import Interface
from .IRcreator import RandomItemCreator, LoadItemCreator, RandomInstanceCreator, RandomCateCreator
from .space import Space
import threading
from .cvTools import getConvexHullActions
import random

class PackingGame(gym.Env):
    def __init__(self,
                 resolutionAct,
                 resolutionH = 0.01,
                 bin_dimension = [0.32, 0.32, 0.3],
                 objPath = './data/datas/256',
                 shapeDict = None,
                 infoDict = None,
                 dicPath = None,
                 test = False,
                 dataname = None,
                 packed_holder = 50,
                 DownRotNum=None,
                 ZRotNum=None,
                 heightMap = False,
                 useHeightMap=False,
                 visual = False,
                 timeStr = None,
                 globalView = False, # if we cancel the global view, the heightmap need to be re-scanned.
                 stability = False,
                 poseDist = False,
                 shotInfo=None,
                 rewardType='ratio',  # number, aabb, ratio
                 actionType='Uniform', # Uniform, RotAction, LineAction, HeuAction
                 elementWise = False,
                 simulation=True,
                 scale = [100,100,100],
                 selectedAction = False,
                 convexAction = None,
                 previewNum = 1,
                 dataSample='instance',
                 maxBatch = 2,
                 LFSS = False,
                 randomConvex = False,
                 meshScale = 1,
                 heightResolution = 0.01,
                 **kwargs):


        self.resolutionAct = resolutionAct
        self.bin_dimension = np.round(bin_dimension, decimals=6)
        self.scale = np.array(scale)
        self.objPath = objPath
        self.meshScale = meshScale

        self.interface = Interface(bin=self.bin_dimension, foldername = objPath,
                                   visual=visual, scale = self.scale, simulationScale=self.meshScale)
        self.shapeDict = shapeDict
        # print('shapeDictLen',len(self.shapeDict))
        self.infoDict  = infoDict
        self.dicPath = load(dicPath)
        self.rangeX_A, self.rangeY_A = np.ceil(self.bin_dimension[0:2] / resolutionAct).astype(np.int32)

        self.DownRotNum = DownRotNum
        self.ZRotNum    = ZRotNum
        self.packed_holder = packed_holder
        self.useHeightMap = useHeightMap
        self.heightMapPre = heightMap
        self.globalView = globalView
        self.stability = stability
        self.rewardType = rewardType
        self.actionType = actionType
        self.heuristicPool = ['DBLF', 'HM', 'MINZ', 'FIRSTFIT']
        # self.heuristicPool = ['DBLF', 'HM', 'MINZ']
        self.selectedAction = selectedAction
        self.convexAction = convexAction
        if self.convexAction is not None:
            assert self.selectedAction
        self.chooseItem = previewNum > 1
        self.previewNum = previewNum

        self.lineAction = self.actionType == 'LineAction'
        self.rotAction = self.actionType == 'RotAction'
        self.heuAction = self.actionType == 'HeuAction'
        self.elementWise = elementWise
        self.simulation = simulation
        if self.elementWise:
            self.item_vec = np.zeros((packed_holder, 9))
        else:
            self.item_vec = np.zeros((1000, 9))
        if self.heightMapPre: assert useHeightMap

        if self.useHeightMap:
            self.space = Space(self.bin_dimension, resolutionAct, resolutionH, False,  self.DownRotNum, self.ZRotNum, shotInfo, self.scale)
        if test and dataname is not None:
            self.item_creator = LoadItemCreator(data_name=dataname)
        else:
            if dataSample == 'category':
                self.item_creator = RandomCateCreator(np.arange(0, len(self.shapeDict.keys())), self.dicPath)
            elif dataSample == 'instance':
                self.item_creator = RandomInstanceCreator(np.arange(0, len(self.shapeDict.keys())), self.dicPath)
            else:
                assert dataSample == 'pose'
                self.item_creator = RandomItemCreator(np.arange(0, len(self.shapeDict.keys())))

        self.next_item_vec = np.zeros((9))

        self.item_idx = 0

        self.transformation = []
        DownFaceList, ZRotList = getRotationMatrix(DownRotNum, ZRotNum)
        for d in DownFaceList:
            for z in ZRotList:
                quat = transforms3d.quaternions.mat2quat(np.dot(z, d)[0:3, 0:3])
                self.transformation.append([quat[1],quat[2],quat[3],quat[0]]) # Saved in xyzw
        self.transformation = np.array(self.transformation)

        self.rotNum = self.ZRotNum * self.DownRotNum
        if self.selectedAction:
            self.act_len = self.selectedAction
        elif self.rotAction:
            self.act_len = self.rotNum
        elif self.lineAction:
            self.act_len = self.rangeX_A * self.rotNum
        elif self.heuAction:
            # self.act_len = len(self.heuristicPool) * self.rotNum * 4
            self.act_len = len(self.heuristicPool)
        else:
            self.act_len = self.rangeX_A * self.rangeY_A * self.rotNum

        if self.chooseItem:
            self.act_len = self.previewNum

        if not self.chooseItem:
            self.obs_len = len(self.next_item_vec.reshape(-1))

            if self.elementWise:
                self.obs_len += len(self.item_vec.reshape(-1))
            if self.selectedAction:
                self.obs_len += self.selectedAction * 5
            else:
                self.obs_len += self.act_len
        else:
            self.obs_len = self.previewNum

        if self.heightMapPre:
            self.obs_len += self.space.heightmapC.size

        self.observation_space = gym.spaces.Box(low=0.0, high=self.bin_dimension[2],
                                                shape=(self.obs_len,))
        self.action_space = gym.spaces.Discrete(self.act_len)

        # self.trajsInfo = []
        self.timeStr = timeStr
        self.tolerance = 0 # defalt 0.002
        self.poseDist = poseDist
        if self.stability:
            assert self.simulation
            self.finished = [True]
            self.score = [0]
            self.nowTask = False
            self.nullObs = np.zeros((self.obs_len))
        self.episodeCounter = 0
        self.updatePeriod = 500
        assert not elementWise
        self.trajs = []
        self.orderAction = 0
        self.hierachical = False

        self.calFreq = False

        if self.calFreq:
            self.dicRecorder = {}
            for v in self.dicPath.values():
                self.dicRecorder[v] = 0
        self.test = test

        # self.oneshapeTime = 0
        # self.oneshapeFreq = 0
        # self.figure8 = []
        self.timeStr = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
        self.maxBatch = maxBatch
        self.LFSS = LFSS
        self.randomConvex = randomConvex
        self.heightResolution = heightResolution
        self.candidates_num = []
    def seed(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]

    def close(self):
        self.interface.close()

    def reset(self, index = None):
        if self.useHeightMap:
            self.space.reset()

        self.episodeCounter = (self.episodeCounter + 1) % self.updatePeriod
        if self.episodeCounter == 0:
            self.interface.close()
            del self.interface
            self.interface = Interface(bin=self.bin_dimension, foldername=self.objPath,
                                       visual=False, scale=self.scale, simulationScale=self.meshScale)
        else:
            self.interface.reset()

        self.item_creator.reset(index)
        self.packed = []
        self.packedId = []
        self.next_item_vec[:] = 0
        self.item_idx = 0
        self.areaSum = []
        self.dists = []
        self.item_vec[:] = 0
        self.save = False
        return self.cur_observation()

    def get_ratio(self):
        totalVolume = 0
        for idx in range(self.item_idx):
            totalVolume += self.infoDict[int(self.item_vec[idx][0])][0]['volume']
        return totalVolume / np.prod(self.bin_dimension)

    def get_occupancy(self):
        heightMapShape = self.space.heightmapC.shape
        return np.sum(self.space.heightmapC) / (heightMapShape[0] * heightMapShape[1] * self.bin_dimension[2])

    def get_item_ratio(self, next_item_ID):
        return self.infoDict[next_item_ID][0]['volume'] / np.prod(self.bin_dimension)

    def gen_next_item_ID(self):
        return self.item_creator.preview(1)[0]


    def get_action_candidates(self, orderAction):
        if not self.LFSS:
            self.hierachical = True
            self.next_item_ID = self.next_k_item_ID[orderAction]
            self.space.get_possible_position(self.next_item_ID, self.shapeDict[self.next_item_ID], self.selectedAction)
            self.chooseItem = False
            locObservation  = self.cur_observation(genItem = False)
            self.chooseItem = True
            self.orderAction = orderAction
            return locObservation
        else:
            # 完全无视order action
            self.hierachical = True
            vList = []
            for largeID in self.next_k_item_ID:
                vList.append(self.infoDict[largeID][0]['volume'])
            orderAction = np.argmax(vList)
            self.next_item_ID = self.next_k_item_ID[orderAction]
            self.space.get_possible_position(self.next_item_ID, self.shapeDict[self.next_item_ID], self.selectedAction)
            self.chooseItem = False
            locObservation  = self.cur_observation(genItem = False)
            self.chooseItem = True
            self.orderAction = orderAction
            return locObservation

    def get_all_possible_observation(self):
        # 完全无视order action
        self.hierachical = True
        self.chooseItem = False

        all_obs = []
        for itemID in self.next_k_item_ID:
            self.next_item_ID = itemID
            self.space.get_possible_position(self.next_item_ID, self.shapeDict[self.next_item_ID], self.selectedAction)
            locObservation  = self.cur_observation(genItem = False)
            all_obs.append(locObservation)
        return np.concatenate(all_obs, axis=0)


    def cur_observation(self, genItem = True, draw = False):
        if self.item_idx != 0 and self.elementWise:
            positions, orientations = self.interface.getAllPositionAndOrientation(inner=False)
            self.item_vec[0:self.item_idx, 1:4] = np.array([positions[0:self.item_idx]])
            self.item_vec[0:self.item_idx, 4:8] = np.array([orientations[0:self.item_idx]])
        self.endSimulation = time.time()
        self.action_start = time.time()
        if not self.chooseItem:
            if genItem:
                self.next_item_ID = self.gen_next_item_ID()
            self.next_item_vec[0] = self.next_item_ID

            if self.useHeightMap:
                naiveMask = self.space.get_possible_position(self.next_item_ID, self.shapeDict[self.next_item_ID], self.selectedAction)
                if self.lineAction:
                    naiveMask = np.sum(naiveMask.reshape((-1, self.rangeY_A)), axis=1)
                    naiveMask = np.where(naiveMask > 0, 1, 0)
                elif self.rotAction:
                    naiveMask = np.sum(naiveMask.reshape((-1, self.rangeX_A * self.rangeY_A)), axis=1)
                    naiveMask = np.where(naiveMask > 0, 1, 0)
            else:
                naiveMask = self.get_possible_position(self.next_item_ID)

            if self.heuAction: naiveMask = np.ones(self.act_len)
            result = self.next_item_vec.reshape(-1)

            if not self.selectedAction:
                result = np.concatenate((self.next_item_vec.reshape(-1),
                                     naiveMask.reshape(-1)))

            if self.heightMapPre:
                result = np.concatenate((result, self.space.heightmapC.reshape(-1)))
            if self.elementWise:
                result = np.concatenate((self.item_vec.reshape(-1), result))
            if self.selectedAction:
                self.candidates = None
                if self.convexAction is not None:
                    self.candidates, save = getConvexHullActions(self.space.posZValid, self.space.naiveMask,
                                                                 self.convexAction,
                                                                 self.heightResolution,
                                                           draw=[draw, len(self.packed)])
                    self.candidates_num.append(len(self.candidates) if self.candidates is not None else 0)
                    if save:
                        self.save = True
                    if self.candidates is not None:
                        if len(self.candidates) > self.selectedAction:
                            # sort with height
                            if not self.randomConvex:
                                selectedIndex = np.argsort(self.candidates[:,3])[0: self.selectedAction]
                            # sort randomly
                            else:
                                selectedIndex = np.arange(len(self.candidates))
                                np.random.shuffle(selectedIndex)
                                selectedIndex = selectedIndex[0:self.selectedAction]
                            self.candidates = self.candidates[selectedIndex]
                        elif len(self.candidates) < self.selectedAction:
                            dif = self.selectedAction - len(self.candidates)
                            self.candidates = np.concatenate((self.candidates, np.zeros((dif, 5))), axis=0)

                if self.candidates is None:
                    poszFlatten = self.space.posZValid.reshape(-1)
                    selectedIndex = np.argsort(poszFlatten)[0: self.selectedAction]
                    ROT,X,Y = np.unravel_index(selectedIndex, (self.rotNum, self.rangeX_A, self.rangeY_A))
                    H = poszFlatten[selectedIndex]
                    V = self.space.naiveMask.reshape(-1)[selectedIndex]
                    self.candidates = np.concatenate([ROT.reshape(-1, 1), X.reshape(-1, 1),
                                                     Y.reshape(-1, 1), H.reshape(-1, 1), V.reshape(-1, 1)], axis=1)

                result = np.concatenate((self.candidates.reshape(-1), result))
        else:
            self.next_k_item_ID = self.item_creator.preview(self.previewNum)
            result = np.concatenate((np.array(self.next_k_item_ID), self.space.heightmapC.reshape(-1)))

        self.action_stop = time.time()
        return result

    def action_to_position(self, action):
        if self.chooseItem and not self.hierachical:
            self.orderAction = action
            self.next_item_ID = self.next_k_item_ID[action]
            self.space.get_possible_position(self.next_item_ID, self.shapeDict[self.next_item_ID], self.selectedAction)
            rotIdx, lx, ly = self.space.get_heuristic_action(0, 'DBLF', self.next_item_ID, self.shapeDict[self.next_item_ID])
        elif self.lineAction:
            rotIdx = action // self.rangeX_A
            lx = action %  self.rangeX_A
            ly = np.argmin(self.space.posZmap[rotIdx,lx])
        elif self.rotAction:
            rotIdx = action
            index = np.argmin(self.space.posZmap[rotIdx])
            lx, ly = np.unravel_index(index, (self.rangeX_A, self.rangeY_A))
        elif self.heuAction:
            heuIdx = np.unravel_index(action, (len(self.heuristicPool),))[0]
            rotIdx, lx, ly = self.space.get_heuristic_action(0, self.heuristicPool[heuIdx], self.next_item_ID, self.shapeDict[self.next_item_ID])
        else:
            if self.actionType == 'UniformTuple':
                rotIdx, lx, ly = action
            elif self.selectedAction:
                rotIdx, lx, ly = self.candidates[action][0:3].astype(np.int)
            else:
                rotIdx, lx, ly = np.unravel_index(action, (self.rotNum, self.rangeX_A, self.rangeY_A))
        return rotIdx, np.round((lx * self.resolutionAct, ly * self.resolutionAct, self.bin_dimension[2]), decimals=6), (lx,ly)

    def get_possible_position(self, next_item_ID):

        naiveMask = np.zeros((self.rotNum, self.rangeX_A, self.rangeX_A))

        for rotIdx in range(self.rotNum):
            extents = self.infoDict[next_item_ID][rotIdx]['extents']
            boundingSize = np.round(extents, decimals=6)
            boundingSizeInt = np.ceil(boundingSize / self.resolutionAct).astype(np.int32)
            rangeX_O, rangeY_O = boundingSizeInt[0], boundingSizeInt[1]
            naiveMask[rotIdx, 0:self.rangeX_A - rangeX_O + 1, 0:self.rangeX_A - rangeY_O + 1] = 1

        return naiveMask

    def prejudge(self, rotIdx, translation, naiveMask):
        extents = self.shapeDict[self.next_item_ID][rotIdx].extents
        if np.round(translation[0] + extents[0] - self.bin_dimension[0], decimals=6)  > 0 \
            or np.round(translation[1] + extents[1] - self.bin_dimension[1], decimals=6) > 0:
            return False
        if np.sum(naiveMask) == 0:
            return False
        return True

    # Note the transform between Ra coord and Rh coord
    def step(self, action):
        if self.stability and not self.finished[0]:
            return self.nullObs, 0.0, False, {'Valid': False, 'MINZ': np.argmin(self.space.posZValid)}

        if self.stability and self.finished[0] and self.nowTask:
            heightMapShape = self.space.heightmapC.shape
            info = {'counter': self.item_idx,
                    'ratio': self.get_ratio(),
                    'bottom_area': np.mean(self.areaSum),
                    'poseDist': np.mean(self.dists),
                    'Occupancy': np.sum(self.space.heightmapC)/ (heightMapShape[0] * heightMapShape[1] * self.bin_dimension[2]),
                    'Valid': True,
                    'MINZ': np.argmin(self.space.posZValid)}
            self.nowTask = False
            info['stability'] = self.score[0]
            reward = 1e2 * info['stability']
            observation = self.cur_observation()
            return observation, reward, True, info

        rotIdx, targetFLB, coordinate = self.action_to_position(action)
        self.startSimulation = time.time()

        rotation = self.transformation[int(rotIdx)]

        valid = False
        succeeded = self.prejudge(rotIdx, targetFLB, self.space.naiveMask)
        color = [1,0,0,1] if not succeeded else None
        id = self.interface.addObject(self.dicPath[self.next_item_ID][0:-4], targetFLB = targetFLB, rotation = rotation,
                                      linearDamping = 0.5, angularDamping = 0.5, color = color)
        if self.useHeightMap:
            # height = self.space.get_lowest_z(self.next_item_ID, self.shapeDict[self.next_item_ID], rotIdx, coordinate)
            height = self.space.posZmap[rotIdx, coordinate[0], coordinate[1]]
            self.interface.adjustHeight(id , height + self.tolerance)
        else:
            assert False

        if succeeded:
            if self.simulation:
                succeeded, valid = self.interface.simulateToQuasistatic(givenId=id,
                                                                        linearTol = 0.01,
                                                                        angularTol = 0.01,
                                                                        maxBatch=self.maxBatch)
            else:
                succeeded, valid = self.interface.simulateHeight(id)

            if not self.globalView:
                # self.interface.secondSimulation(maxBatch=1)
                self.interface.disableObject(id)

        bounds = self.interface.get_wraped_AABB(id, inner=False)
        positionT, orientationT = self.interface.get_Wraped_Position_And_Orientation(id, inner=False)

        self.packed.append([self.next_item_ID, self.dicPath[self.next_item_ID], positionT, orientationT])
        self.packedId.append(id)


        if not succeeded:

            if self.globalView and self.test:
                for replayIdx, idNow in enumerate(self.packedId):
                    positionT, orientationT = self.interface.get_Wraped_Position_And_Orientation(idNow, inner=False)
                    self.packed[replayIdx][2] = positionT
                    self.packed[replayIdx][3] = orientationT

            if self.calFreq:
                self.dicRecorder[self.dicPath[self.next_item_ID]] += 1
                freq = np.array(list(self.dicRecorder.values())).astype(np.float)
                freq /= np.sum(freq)
                name = list(self.dicRecorder.keys())
                order = np.argsort(freq)
                print('######################################')
                for index in order:
                    print('{}: {}'.format(name[index], freq[index]))
                name = self.objPath.split('/')[-1]
                # np.save(name + '.npy', self.dicRecorder)

            reward = 0.0
            if self.stability:
                self.finished[0] = False
                subProcess = threading.Thread(target=stabilityScore, args=(self.score, self.finished, self.interface))
                subProcess.start()
                self.nowTask = True
                return self.nullObs, 0.0, False, {'Valid': False, 'MINZ': np.argmin(self.space.posZValid)}

            info = {'counter': self.item_idx,
                    'ratio': self.get_ratio(),
                    'bottom_area': np.mean(self.areaSum),
                    'poseDist': np.mean(self.dists),
                    'Occupancy': self.get_occupancy(),
                    'Valid': True,
                    'MINZ': np.argmin(self.space.posZValid)}
            observation = self.cur_observation()
            print('candidate number min:{}, max:{}, mean:{}'.format(np.min(self.candidates_num), np.max(self.candidates_num), np.mean(self.candidates_num)))
            return observation, reward, True, info

        if valid:
            if self.useHeightMap:
                if self.globalView:
                    self.space.shot_whole()
                else:
                    # self.space.place_item(bounds, self.interface.defaultScale[0])
                    self.space.place_item_trimesh(self.shapeDict[self.next_item_ID][0], (positionT, orientationT), (bounds, self.next_item_ID))
            # draw_heatmap(self.space.heightmapC)
            self.item_vec[self.item_idx, 0] = self.next_item_ID
            self.item_vec[self.item_idx, -1] = 1

            if self.rewardType == 'ratio':
                item_ratio = self.get_item_ratio(self.next_item_ID)
                reward = item_ratio * 10
            elif self.rewardType == 'aabb':
                aabb_ratio = np.prod(bounds[1] - bounds[0]) / np.prod(self.bin_dimension)
                reward = aabb_ratio * 10 # wrong design, vmax is not right
            elif self.rewardType == 'number':
                reward = 1 # wrong design, vmax is not right

            # Side tasks
            bottom_area = np.prod((bounds[1] - bounds[0])[0:2])
            self.areaSum.append(bottom_area)

            # FLB2centroid = np.array(positionT) - bounds[0]
            # centroidPosition = np.array((*translation[0:2], height)) + FLB2centroid
            # dist = pose_distance((centroidPosition, rotation), (positionT, orientationT))
            dist = 0
            self.dists.append(dist)
            if self.poseDist:
                reward = max(reward - dist * 0.1, 1e-3)

            reward -= np.var(self.space.heightmapC)

            self.item_idx += 1
            self.item_creator.update_item_queue(self.orderAction)
            self.item_creator.generate_item()  # add a new box to the list
            observation = self.cur_observation()
            return observation, reward, False, {'Valid': True, 'MINZ': np.argmin(self.space.posZValid)}
        else:
            print('Invalid call')
            self.packed.pop()
            self.packedId.pop()

            delId = self.interface.objs.pop()
            self.interface.removeBody(delId)
            self.item_creator.update_item_queue(self.orderAction)
            self.item_creator.generate_item()  # Add a new box to the list
            observation = self.cur_observation()
            return observation, 0.0, False, {'Valid': False, 'MINZ': np.argmin(self.space.posZValid)}