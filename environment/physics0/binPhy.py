import time
import gym
import numpy as np
from torch import load
from tools import getRotationMatrix
import transforms3d
from .Interface import Interface
from .IRcreator import RandomItemCreator, LoadItemCreator, RandomInstanceCreator, RandomCateCreator
from .space import Space
from .cvTools import getConvexHullActions
import random
import threading

def non_blocking_simulation(interface, finished, id, non_blocking_result):
    succeeded, valid = interface.simulateToQuasistatic(givenId=id,
                                    linearTol=0.01,
                                    angularTol=0.01)

    finished[0] = True
    non_blocking_result[0] = [succeeded, valid]
class PackingGame(gym.Env):
    def __init__(self,
                 args
    ):
        args = vars(args)
        self.resolutionAct = args['resolutionA']
        self.resolutionH   = args['resolutionH']
        self.bin_dimension = args['bin_dimension']
        self.scale         = args['scale']
        self.objPath       = args['objPath']
        self.meshScale     = args['meshScale']
        self.shapeDict     = args['shapeDict']
        self.infoDict      = args['infoDict']
        self.dicPath       = load(args['dicPath'])
        self.ZRotNum       = args['ZRotNum']
        self.heightMapPre  = args['heightMap']
        self.globalView    = not args['only_simulate_current']
        self.selectedAction= args['selectedAction']
        self.bufferSize    = args['bufferSize']
        self.chooseItem    = self.bufferSize > 1
        self.simulation    = args['simulation']
        self.evaluate      = args['evaluate']
        self.maxBatch      =  args['maxBatch']
        self.heightResolution   = args['resolutionZ']
        self.dataSample    = args['dataSample']
        self.dataname      = args['test_name']
        self.visual        = args['visual']
        self.non_blocking  = args['non_blocking']
        self.time_limit    = args['time_limit']


        self.interface = None
        self.item_vec = np.zeros((1000, 9))
        self.rangeX_A, self.rangeY_A = np.ceil(self.bin_dimension[0:2] / self.resolutionAct).astype(np.int32)
        self.space = Space(self.bin_dimension, self.resolutionAct, self.resolutionH, False,   self.ZRotNum,
                           args['shotInfo'], self.scale)

        if self.evaluate and self.dataname is not None:
            self.item_creator = LoadItemCreator(data_name=self.dataname)
        else:
            if self.dataSample == 'category':
                self.item_creator = RandomCateCreator(np.arange(0, len(self.shapeDict.keys())), self.dicPath)
            elif self.dataSample == 'instance':
                self.item_creator = RandomInstanceCreator(np.arange(0, len(self.shapeDict.keys())), self.dicPath)
            else:
                assert self.dataSample == 'pose'
                self.item_creator = RandomItemCreator(np.arange(0, len(self.shapeDict.keys())))

        self.next_item_vec = np.zeros((9))

        self.item_idx = 0

        self.transformation = []
        DownFaceList, ZRotList = getRotationMatrix(1, self.ZRotNum)
        for d in DownFaceList:
            for z in ZRotList:
                quat = transforms3d.quaternions.mat2quat(np.dot(z, d)[0:3, 0:3])
                self.transformation.append([quat[1],quat[2],quat[3],quat[0]]) # Saved in xyzw
        self.transformation = np.array(self.transformation)

        self.rotNum = self.ZRotNum
        self.act_len = self.selectedAction

        if self.chooseItem:
            self.act_len = self.bufferSize

        if not self.chooseItem:
            self.obs_len = len(self.next_item_vec.reshape(-1))

            if self.selectedAction:
                self.obs_len += self.selectedAction * 5
            else:
                self.obs_len += self.act_len
        else:
            self.obs_len = self.bufferSize

        if self.heightMapPre:
            self.obs_len += self.space.heightmapC.size

        self.observation_space = gym.spaces.Box(low=0.0, high=self.bin_dimension[2],
                                                shape=(self.obs_len,))
        self.action_space = gym.spaces.Discrete(self.act_len)

        self.tolerance = 0 # defalt 0.002

        self.episodeCounter = 0
        self.updatePeriod = 500
        self.trajs = []
        self.orderAction = 0
        self.hierachical = False

        if self.non_blocking:
            self.nullObs = np.zeros((self.obs_len))
            self.finished = [True]
            self.non_blocking_result = [None]
            self.nowTask = False

    def seed(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]

    def close(self):
        self.interface.close()

    def reset(self, index = None):
        self.space.reset()

        self.episodeCounter = (self.episodeCounter + 1) % self.updatePeriod
        if self.episodeCounter == 0 or self.interface is None:
            if self.interface is not None:
                self.interface.close()
                del self.interface
            self.interface = Interface(bin=self.bin_dimension, foldername=self.objPath, visual=self.visual,
                                       scale=self.scale, simulationScale=self.meshScale, maxBatch=self.maxBatch,)
        else:
            self.interface.reset()
        self.item_creator.reset(index)
        self.packed = []
        self.packedId = []
        self.next_item_vec[:] = 0
        self.item_idx = 0
        self.item_vec[:] = 0
        self.id = None
        return self.cur_observation()

    def get_ratio(self):
        totalVolume = 0
        for idx in range(self.item_idx):
            totalVolume += self.infoDict[int(self.item_vec[idx][0])][0]['volume']
        return totalVolume / np.prod(self.bin_dimension)

    def get_item_ratio(self, next_item_ID):
        return self.infoDict[next_item_ID][0]['volume'] / np.prod(self.bin_dimension)

    def gen_next_item_ID(self):
        return self.item_creator.preview(1)[0]

    def get_action_candidates(self, orderAction):
        self.hierachical = True
        self.next_item_ID = self.next_k_item_ID[orderAction]
        self.space.get_possible_position(self.next_item_ID, self.shapeDict[self.next_item_ID], self.selectedAction)
        self.chooseItem = False
        locObservation  = self.cur_observation(genItem = False)
        self.chooseItem = True
        self.orderAction = orderAction
        return locObservation

    def get_all_possible_observation(self):
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
        if self.item_idx != 0:
            positions, orientations = self.interface.getAllPositionAndOrientation(inner=False)
            self.item_vec[0:self.item_idx, 1:4] = np.array([positions[0:self.item_idx]])
            self.item_vec[0:self.item_idx, 4:8] = np.array([orientations[0:self.item_idx]])
        if not self.chooseItem:
            if genItem:
                self.next_item_ID = self.gen_next_item_ID()
            self.next_item_vec[0] = self.next_item_ID

            naiveMask = self.space.get_possible_position(self.next_item_ID, self.shapeDict[self.next_item_ID], self.selectedAction)


            result = self.next_item_vec.reshape(-1)

            if not self.selectedAction:
                result = np.concatenate((self.next_item_vec.reshape(-1),
                                     naiveMask.reshape(-1)))

            if self.heightMapPre:
                result = np.concatenate((result, self.space.heightmapC.reshape(-1)))
            if self.selectedAction:
                self.candidates = None
                self.candidates= getConvexHullActions(self.space.posZValid, self.space.naiveMask,
                                                             self.heightResolution)
                if self.candidates is not None:
                    if len(self.candidates) > self.selectedAction:
                        # sort with height
                        selectedIndex = np.argsort(self.candidates[:,3])[0: self.selectedAction]
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
                    H[:] = self.bin_dimension[-1]
                    self.candidates = np.concatenate([ROT.reshape(-1, 1), X.reshape(-1, 1),
                                                     Y.reshape(-1, 1), H.reshape(-1, 1), V.reshape(-1, 1)], axis=1)

                result = np.concatenate((self.candidates.reshape(-1), result))
        else:
            self.next_k_item_ID = self.item_creator.preview(self.bufferSize)
            result = np.concatenate((np.array(self.next_k_item_ID), self.space.heightmapC.reshape(-1)))

        return result

    def action_to_position(self, action):
        rotIdx, lx, ly = self.candidates[action][0:3].astype(np.int)
        return rotIdx, np.round((lx * self.resolutionAct, ly * self.resolutionAct, self.bin_dimension[2]), decimals=6), (lx,ly)

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
        if self.non_blocking and not self.finished[0]:
            return self.nullObs, 0.0, False, {'Valid': False}

        if self.non_blocking and self.finished[0] and self.nowTask:
            success, sim_suc = self.non_blocking_result[0]
            self.nowTask = False
            self.non_blocking_result[0] = None

        else:
            rotIdx, targetFLB, coordinate = self.action_to_position(action)
            rotation = self.transformation[int(rotIdx)]

            sim_suc = False
            success = self.prejudge(rotIdx, targetFLB, self.space.naiveMask)
            self.id = self.interface.addObject(self.dicPath[self.next_item_ID][0:-4], targetFLB = targetFLB, rotation = rotation,
                                          linearDamping = 0.5, angularDamping = 0.5)

            height = self.space.posZmap[rotIdx, coordinate[0], coordinate[1]]
            self.interface.adjustHeight(self.id , height + self.tolerance)

            if success:
                if self.simulation:
                    if self.non_blocking:
                        self.finished[0] = False
                        subProcess = threading.Thread(target=non_blocking_simulation, args=(self.interface, self.finished, self.id, self.non_blocking_result))
                        subProcess.start()
                        self.nowTask = True
                        start_time = time.time()
                        end_time   = start_time
                        while end_time - start_time < self.time_limit:
                            end_time = time.time()
                            if self.finished[0]:
                                break
                        if not self.finished[0]:
                            return self.nullObs, 0.0, False, {'Valid': False}
                    else:
                        success, sim_suc = self.interface.simulateToQuasistatic(givenId=self.id,
                                                                            linearTol = 0.01,
                                                                            angularTol = 0.01)
                else:
                    success, sim_suc = self.interface.simulateHeight(self.id)

        if not self.globalView:
            self.interface.disableObject(self.id)

        bounds = self.interface.get_wraped_AABB(self.id, inner=False)
        positionT, orientationT = self.interface.get_Wraped_Position_And_Orientation(self.id, inner=False)
        self.packed.append([self.next_item_ID, self.dicPath[self.next_item_ID], positionT, orientationT])
        self.packedId.append(self.id)

        if not success:
            if self.globalView and self.evaluate:
                for replayIdx, idNow in enumerate(self.packedId):
                    positionT, orientationT = self.interface.get_Wraped_Position_And_Orientation(idNow, inner=False)
                    self.packed[replayIdx][2] = positionT
                    self.packed[replayIdx][3] = orientationT
            reward = 0.0
            info = {'counter': self.item_idx,
                    'ratio': self.get_ratio(),
                    'Valid': True,
                    }
            observation = self.cur_observation()
            return observation, reward, True, info

        if sim_suc:
            if self.globalView:
                self.space.shot_whole()
            else:
                self.space.place_item_trimesh(self.shapeDict[self.next_item_ID][0], (positionT, orientationT), (bounds, self.next_item_ID))

            self.item_vec[self.item_idx, 0] = self.next_item_ID
            self.item_vec[self.item_idx, -1] = 1
            item_ratio = self.get_item_ratio(self.next_item_ID)
            reward = item_ratio * 10
            self.item_idx += 1
            self.item_creator.update_item_queue(self.orderAction)
            self.item_creator.generate_item()  # add a new box to the list
            observation = self.cur_observation()
            return observation, reward, False, {'Valid': True}
        else:
            # Invalid call
            self.packed.pop()
            self.packedId.pop()
            delId = self.interface.objs.pop()
            self.interface.removeBody(delId)
            self.item_creator.update_item_queue(self.orderAction)
            self.item_creator.generate_item()  # Add a new box to the list
            observation = self.cur_observation()
            return observation, 0.0, False, {'Valid': False}