#--coding:utf-8--
import numpy as np
from tools import gen_ray_origin_direction, shot_after_item_placement, getRotationMatrix, extendMat, shot_item
from matplotlib import pyplot as plt
import transforms3d
import pybullet as p

def draw_heatmap(heightMap, vmin = 0, vmax = 0.32):
    plt.imshow(heightMap,  cmap=plt.cm.hot, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()

# Record heightMap for heuristic things.
class Space(object):
    def __init__(self, bin_dimension, resolutionAct, resolutionH, boxPack = False, DownRotNum = None, ZRotNum = None, shotInfo = None, scale = None):
        self.bin_dimension = bin_dimension
        self.resolutionH = resolutionH
        self.resolutionAct = resolutionAct
        self.stepSize = int(self.resolutionAct / self.resolutionH)
        assert self.stepSize == self.resolutionAct / self.resolutionH
        self.rotNum = DownRotNum * ZRotNum
        self.scale = scale
        self.rangeX_C, self.rangeY_C = np.ceil(bin_dimension[0:2] / resolutionH).astype(np.int32)
        self.rangeX_A, self.rangeY_A = np.ceil(bin_dimension[0:2] / resolutionAct).astype(np.int32)

        self.heightmapC = np.zeros((self.rangeX_C, self.rangeY_C))
        self.ray_origins, self.ray_directions = \
            gen_ray_origin_direction(self.rangeX_C, self.rangeY_C, resolutionH, boxPack, shift = 0.001)

        # self.pack_meshes = []
        self.shotInfo = shotInfo

        self.transformation = []
        DownFaceList, ZRotList = getRotationMatrix(DownRotNum, ZRotNum)

        for d in DownFaceList:
            for z in ZRotList:
                self.transformation.append(np.dot(z, d).reshape(-1))
        self.transformation = np.array(self.transformation)

        # Some auxiliary variables
        self.posZmap = np.zeros((self.rotNum, self.rangeX_A, self.rangeY_A))
        self.posZValid = np.zeros((self.rotNum, self.rangeX_A, self.rangeY_A))
        bottom = np.arange(0, self.rangeX_A * self.rangeY_A).reshape((self.rangeX_A, self.rangeY_A))
        self.coors = np.zeros((self.rangeX_A, self.rangeY_A, 2))
        self.coors[:, :, 0] = bottom // self.rangeY_A
        self.coors[:, :, 1] = bottom % self.rangeY_A

    def reset(self):
        self.heightmapC[:] = 0
        self.item_idx = 0
        self.scene = []

    # def shot_heightMap(self, ray_origins_ini, bounds, xRange=20, yRange=20, start=[0, 0, 0]):  # xRange, yRange the grid range.
    # 
    #     batchNum = int(np.ceil(xRange * yRange / (p.MAX_RAY_INTERSECTION_BATCH_SIZE - 1)))
    #     maxXStep = int(np.floor(xRange / batchNum))
    #     XStart = start[0]
    # 
    #     heightMapHList = []
    #     maskHList = []
    # 
    #     while XStart < start[0] + xRange:
    #         step = min(start[0] + xRange - XStart, maxXStep)
    # 
    #         ray_origins = ray_origins_ini[XStart: XStart + step, start[1]: start[1] + yRange].copy().reshape((-1, 3)) * self.scale
    #         ray_ends    = ray_origins_ini[XStart: XStart + step, start[1]: start[1] + yRange].copy().reshape((-1, 3)) * self.scale
    # 
    #         ray_origins[:, 2] = bounds[1][2] * self.scale[2]
    #         ray_ends[:, 2]    = 0
    #         intersections = p.rayTestBatch(ray_origins, ray_ends, numThreads=16)
    #         intersections = np.array(intersections, dtype=object)
    # 
    #         maskH = intersections[: , 0]
    #         maskH = np.where(maskH >= 0, 1, 0)
    # 
    #         if np.sum(maskH) != 0:
    #             fractions = intersections[:, 2]
    #             heightMapH = ray_origins[:, 2] + (ray_ends[:, 2] - ray_origins[:, 2]) * fractions
    #             heightMapH *= maskH
    #         else:
    #             heightMapH = np.zeros(step * yRange)
    #             heightMapH[:] = bounds[1][2] * self.scale[2]
    #             maskH[:] = 1
    # 
    #         heightMapH = heightMapH.reshape((step, yRange)) / self.scale[2]
    #         maskH = maskH.reshape((step, yRange))
    # 
    #         heightMapHList.append(heightMapH)
    #         maskHList.append(maskH)
    #         XStart += maxXStep
    # 
    #     heightMapF = np.concatenate(heightMapHList, axis=0)
    #     maskF = np.concatenate(maskHList, axis=0)
    #     return heightMapF, maskF

    def shot_whole(self):

        ray_origins = self.ray_origins.reshape((-1, 3)) * self.scale
        ray_ends    = ray_origins.copy().reshape((-1, 3))

        ray_origins[:, 2] = self.bin_dimension[2] * self.scale[2] * 2
        ray_ends[:, 2] = 0

        intersections = p.rayTestBatch(ray_origins, ray_ends, numThreads=16)
        intersections = np.array(intersections, dtype=object)

        maskH = intersections[:, 0]
        maskH = np.where(maskH >= 0, 1, 0)

        fractions = intersections[:, 2]
        heightMapH = ray_origins[:, 2] + (ray_ends[:, 2] - ray_origins[:, 2]) * fractions
        heightMapH *= maskH

        heightMapH = heightMapH.reshape((self.rangeX_C, self.rangeY_C)) / self.scale[2]
        self.heightmapC = heightMapH.astype(np.float)


    # def place_item(self, bounds, scale): # 这个是截取移动后的heightMap, 和移动前的heightMap不一样的
    #     # assert scale == 1
    #     minBoundsInt    = np.floor(np.maximum(bounds[0], [0,0,0])/ self.resolutionH).astype(np.int32)
    #     maxBoundsInt    = np.ceil(np.minimum(bounds[1], self.bin_dimension) / self.resolutionH).astype(np.int32)
    #
    #     boundingSizeInt = maxBoundsInt - minBoundsInt
    #     rangeX_O, rangeY_O = boundingSizeInt[0], boundingSizeInt[1]
    #
    #     heightMapH, maskH = self.shot_heightMap(self.ray_origins, bounds, rangeX_O, rangeY_O, start=minBoundsInt)
    #
    #     # Hdraw = np.zeros(heightMapH.shape)
    #     # Hdraw[:] = heightMapH
    #     # draw_heatmap(Hdraw)
    #
    #     # if np.max(heightMapH) > self.bin_dimension[2]:
    #     #     return False
    #
    #     coorX, coorY = minBoundsInt[0:2]
    #     self.heightmapC[coorX:coorX + rangeX_O, coorY:coorY + rangeY_O] = \
    #         np.maximum(self.heightmapC[coorX:coorX + rangeX_O, coorY:coorY + rangeY_O], heightMapH)


    def place_item_trimesh(self, mesh, poseT, debugInfo): # 这个是截取移动后的heightMap, 和移动前的heightMap不一样的
        meshT = mesh.copy()
        # meshT.apply_scale(scale)
        positionT, orientationT = poseT
        meshT.apply_transform(extendMat(transforms3d.euler.quat2mat([orientationT[3], *orientationT[0:3]]))) # OT quat XYZW
        meshT.apply_translation(-meshT.bounds[0])
        meshT.apply_translation(positionT)
        bounds = np.round(meshT.bounds, decimals=6)

        minBoundsInt = np.floor(np.maximum(bounds[0], [0, 0, 0]) / self.resolutionH).astype(np.int32)
        maxBoundsInt = np.ceil(np.minimum(bounds[1], self.bin_dimension) / self.resolutionH).astype(np.int32)
        boundingSizeInt = maxBoundsInt - minBoundsInt
        rangeX_O, rangeY_O = boundingSizeInt[0], boundingSizeInt[1]
        if rangeY_O <= 0 or rangeX_O <= 0:
            print('bounds:{}\nminBoundsInt{}\nmaxBoundsInt{}\nDebugInfo{}'.format(bounds, minBoundsInt, maxBoundsInt, debugInfo))
        heightMapH, maskH = shot_after_item_placement(meshT, self.ray_origins, self.ray_directions, rangeX_O, rangeY_O, start=minBoundsInt)

        coorX, coorY = minBoundsInt[0:2]
        self.heightmapC[coorX:coorX + rangeX_O, coorY:coorY + rangeY_O] = \
            np.maximum(self.heightmapC[coorX:coorX + rangeX_O, coorY:coorY + rangeY_O], heightMapH)

        # draw_heatmap(heightMapH)
        # self.scene.append(meshT)
        # wholeScene = trimesh.Scene(self.scene)
        # wholeScene.show()

    # def get_lowest_z(self, next_item_ID, next_item, rotIdx, coordinate):
    #     boundingSize = next_item[rotIdx].extents
    #     boundingSizeInt = np.ceil(boundingSize / self.resolutionH).astype(np.int32)
    #     rangeX_O, rangeY_O = boundingSizeInt[0], boundingSizeInt[1]
    #     heightMapT, heightMapB, maskH, maskB = self.shotInfo[next_item_ID][rotIdx]  # 这个操作很省运算量，之后也可以考虑用进来
    #     coorX, coorY = coordinate
    #     interceptMap = self.heightmapC[coorX: coorX + rangeX_O, coorY: coorY + rangeY_O]
    #     posZ = np.max((interceptMap - heightMapB) * maskB) # 这里面要把可能为0的部分删去
    #     return posZ

    # 动作设计，还没想好怎么做(感觉这玩意还挺关键的，因为动作空间会很大)
    def get_possible_position(self, next_item_ID, next_item, selectedAction):

        rotNum = len(next_item)
        naiveMask = np.zeros((rotNum, self.rangeX_A, self.rangeY_A))
        self.posZmap[:] = 1e3
        for rotIdx in range(rotNum):
            boundingSize = np.round(next_item[rotIdx].extents, decimals=6)
            rangeX_OH, rangeY_OH = np.ceil(boundingSize[0:2] / self.resolutionH).astype(np.int32)
            rangeX_OA, rangeY_OA = np.ceil(boundingSize[0:2] / self.resolutionAct).astype(np.int32)
            if self.shotInfo is not None:
                heightMapT, heightMapB, maskH, maskB = self.shotInfo[next_item_ID][rotIdx] # 这个操作很省运算量，之后也可以考虑用进来
            else:
                heightMapT, heightMapB, maskH, maskB = shot_item(next_item[rotIdx],
                                                                 self.ray_origins,
                                                                 self.ray_directions,
                                                                 rangeX_OH, rangeY_OH)

            for X in range(self.rangeX_A - rangeX_OA + 1):
                for Y in range(self.rangeY_A - rangeY_OA + 1):
                    coorX, coorY = X * self.stepSize, Y * self.stepSize
                    posZ = np.max((self.heightmapC[coorX: coorX + rangeX_OH, coorY: coorY + rangeY_OH]
                                  - heightMapB) * maskB)
                    if np.round(posZ + boundingSize[2] - self.bin_dimension[2], decimals=6) <= 0:
                        naiveMask[rotIdx, X, Y] = 1
                    self.posZmap[rotIdx, X, Y] = posZ


        # if naiveMask.sum() == 0:
        #     if selectedAction is not None:
        #         posZmap = self.posZmap.reshape(-1)
        #         index = np.argsort(posZmap.reshape(-1))[0:selectedAction]
        #         naiveMask.reshape(-1)[index] = 1
        #     else:
        #         for rotIdx in range(rotNum):
        #             boundingSize = np.round(next_item[rotIdx].extents, decimals=6)
        #             rangeX_OH, rangeY_OH = np.ceil(boundingSize[0:2] / self.resolutionH).astype(np.int32)
        #             naiveMask[rotIdx, 0:self.rangeX_A - rangeX_OH + 1, 0:self.rangeX_A - rangeY_OH + 1] = 1

        self.naiveMask = naiveMask.copy()
        invalidIndex = np.where(naiveMask==0)
        self.posZValid[:] = self.posZmap[:]
        self.posZValid[invalidIndex] = 1e3

        return naiveMask

    def get_possible_position_custom(self, next_item, rotIdx = 0):

        rotNum = 1
        naiveMask = np.zeros((rotNum, self.rangeX_A, self.rangeY_A))
        self.posZmap[:] = 1e3

        if True:
            boundingSize = np.round(next_item.extents, decimals=6)
            rangeX_OH, rangeY_OH = np.ceil(boundingSize[0:2] / self.resolutionH).astype(np.int32)
            rangeX_OA, rangeY_OA = np.ceil(boundingSize[0:2] / self.resolutionAct).astype(np.int32)
            heightMapT, heightMapB, maskH, maskB = shot_item(next_item,
                                                                 self.ray_origins,
                                                                 self.ray_directions,
                                                                 rangeX_OH, rangeY_OH)

            for X in range(self.rangeX_A - rangeX_OA + 1):
                for Y in range(self.rangeY_A - rangeY_OA + 1):
                    coorX, coorY = X * self.stepSize, Y * self.stepSize
                    posZ = np.max((self.heightmapC[coorX: coorX + rangeX_OH, coorY: coorY + rangeY_OH]
                                  - heightMapB) * maskB)
                    if np.round(posZ + boundingSize[2] - self.bin_dimension[2], decimals=6) <= 0:
                        naiveMask[rotIdx, X, Y] = 1
                    self.posZmap[rotIdx, X, Y] = posZ

        self.naiveMask = naiveMask.copy()
        invalidIndex = np.where(naiveMask==0)
        self.posZValid[:] = self.posZmap[:]
        self.posZValid[invalidIndex] = 1e3

        return naiveMask

    def get_heuristic_action(self, dirIdx, method, next_item_ID, next_item):
        if dirIdx == 0:   Xflip, Yflip = False, False
        elif dirIdx == 1: Xflip, Yflip = False, True
        elif dirIdx == 2: Xflip, Yflip = True, False
        else: Xflip, Yflip = True, True
        assert dirIdx <= 3
        if method == 'MINZ':
            invalidIndex = np.where(self.naiveMask == 0)
            score = self.posZmap.copy()
            score[invalidIndex] = 1e6
            score = np.round(score, decimals=6)
            index = np.argmin(score)
            rotIdx, lx, ly = np.unravel_index(index, score.shape)
        elif method == 'DBLF':
            invalidIndex = np.where(self.naiveMask == 0)
            coorsX = self.coors[:,:,0] if not Xflip else self.rangeX_A - self.coors[:,:,0]
            coorsY = self.coors[:,:,1] if not Yflip else self.rangeY_A - self.coors[:,:,1]
            score = coorsX + coorsY
            score = score.reshape((1, -1)).repeat(self.rotNum, axis = 0).reshape(self.naiveMask.shape)
            score = score * self.resolutionAct + 100 * self.posZmap
            score[invalidIndex] = 1e6
            score = np.round(score, decimals=6)
            index = np.argmin(score)
            rotIdx, lx, ly = np.unravel_index(index, score.shape)
        elif method == 'FIRSTFIT':
            invalidIndex = np.where(self.naiveMask == 0)
            coorsX = self.coors[:,:,0] if not Xflip else self.rangeX_A - self.coors[:,:,0]
            coorsY = self.coors[:,:,1] if not Yflip else self.rangeY_A - self.coors[:,:,1]
            score = coorsX + coorsY
            score = score.reshape((1, -1)).repeat(self.rotNum, axis = 0).reshape(self.naiveMask.shape)
            score[invalidIndex] = 1e6
            score = np.round(score, decimals=6)
            index = np.argmin(score)
            rotIdx, lx, ly = np.unravel_index(index, score.shape)
        elif method == 'HM':
            invalidIndex = np.where(self.naiveMask == 0)
            coorsX = self.coors[:,:,0] if not Xflip else self.rangeX_A - self.coors[:,:,0]
            coorsY = self.coors[:,:,1] if not Yflip else self.rangeY_A - self.coors[:,:,1]
            score = (coorsX + coorsY) * self.resolutionAct
            score = score.reshape((1, -1)).repeat(self.rotNum, axis = 0).reshape(self.naiveMask.shape)
            score[invalidIndex] = 1e6
            for rotIdx in range(self.rotNum):
                heightMapT, heightMapB, maskH, maskB = self.shotInfo[next_item_ID][rotIdx]
                boundingSize = np.round(next_item[rotIdx].extents, decimals=6)
                rangeX_OH, rangeY_OH = np.ceil(boundingSize[0:2] / self.resolutionH).astype(np.int32)
                for coorX in range(self.rangeX_A):
                    for coorY in range(self.rangeY_A):
                        if self.naiveMask[rotIdx, coorX, coorY] == 0:
                            continue
                        posZ = self.posZmap[rotIdx, coorX, coorY]
                        X, Y = coorX * self.stepSize, coorY * self.stepSize
                        heightmapC_Prime = np.max(((heightMapT + posZ) * maskH, self.heightmapC[X:X + rangeX_OH, Y:Y + rangeY_OH]), axis=0)
                        mapSum = np.sum(heightmapC_Prime)
                        score[rotIdx, coorX, coorY] += mapSum * 100
            score = np.round(score, decimals=6)
            index = np.argmin(score)
            rotIdx, lx, ly = np.unravel_index(index, score.shape)
        else:
            assert method == 'RANDOM'
            validIndex = np.where(self.naiveMask.reshape(-1) == 1)
            if validIndex is not None:
                index = np.random.choice(validIndex)
            else:
                index = np.random.randint(len(self.naiveMask.reshape(-1)))
            rotIdx, lx, ly = np.unravel_index(index, self.naiveMask.shape)
        return rotIdx, lx,ly

    # def heuristic_method_old(self, next_item_ID, next_item, resolutionAct, method = 'HM'):
    #     heuristicC = 1
    # 
    #     delta_x = resolutionAct
    #     delta_y = resolutionAct
    #     candicadates = []
    #     rotNum = len(next_item)
    # 
    #     for rotIdx in range(rotNum):
    #         boundingSize = np.round(next_item[rotIdx].extents, decimals=6)
    #         rangeX_OH, rangeY_OH = np.ceil(boundingSize[0:2] / self.resolutionH).astype(np.int32)
    #         heightMapT, heightMapB, maskH, maskB = self.shotInfo[next_item_ID][rotIdx]
    # 
    #         posX = 0
    #         while posX <= self.bin_dimension[0] - boundingSize[0]:
    #             posY = 0
    #             while posY <= self.bin_dimension[1] - boundingSize[1]:
    #                 X, Y = int(posX / self.resolutionAct), int(posY / self.resolutionAct)  # 对应heightmap的坐标变化
    #                 coorX, coorY = int(posX / self.resolutionH), int(posY / self.resolutionH)  # 对应heightmap的坐标变化
    #                 posZ = np.max((self.heightmapC[coorX: coorX + rangeX_OH, coorY: coorY + rangeY_OH]
    #                                - heightMapB) * maskB)  # the lowest collision-free Z
    # 
    #                 heightmapC_Prime = np.max(
    #                     ((heightMapT + posZ) * maskH, self.heightmapC[coorX:coorX + rangeX_OH, coorY:coorY + rangeY_OH]), axis=0)
    #                 # maxHeight = np.max(heightmapC_Prime)
    #                 maxHeight = boundingSize[2] + posZ
    #                 # print(maxHeight)
    #                 if np.around(maxHeight - self.bin_dimension[2], decimals=6) > 0:
    #                     posY = np.round(delta_y + posY, decimals=3)
    #                     continue
    #                 mapSum = np.sum(heightmapC_Prime)
    #                 scoreHM = heuristicC * (posX + posY) + mapSum * 100
    #                 scoreDBLF = heuristicC * (posX + posY) + posZ * 100
    #                 candicadates.append([X, Y, posZ, scoreHM, scoreDBLF, rotIdx])
    # 
    #                 posY = np.round(delta_y + posY, decimals=3)
    # 
    #             posX = np.round(delta_x + posX, decimals=3)
    # 
    #     if len(candicadates) != 0:
    #         candicadates = np.round(np.array(candicadates), decimals=6)
    #         if method == 'RANDOM':
    #             action = candicadates[np.random.randint(0, len(candicadates))]
    #             return action.astype(np.int)
    #         elif method == 'DBLF':
    #             action = candicadates[np.argmin(candicadates[:, 4])]
    #             return action.astype(np.int)
    #         elif method == 'HM':
    #             action = candicadates[np.argmin(candicadates[:, 3])]
    #             return action.astype(np.int)
    #         elif method == 'MINZ':
    #             action = candicadates[np.argmin(candicadates[:, 2])]
    #             return action.astype(np.int)
    #         elif method == 'FIRSTFIT':
    #             action = candicadates[np.argmin(candicadates[:, 0] + candicadates[:, 1])]
    #             return action.astype(np.int)
    #     else:
    #         return None

    def heuristic_method(self, next_item_ID, next_item, resolutionAct, method = 'HM'):
        heuristicC = 1

        delta_x = resolutionAct
        delta_y = resolutionAct
        candicadates = []
        rotNum = len(next_item)


        for rotIdx in range(rotNum):
            boundingSize = np.round(next_item[rotIdx].extents, decimals=6)
            rangeX_OH, rangeY_OH = np.ceil(boundingSize[0:2] / self.resolutionH).astype(np.int32)
            heightMapT, heightMapB, maskH, maskB = self.shotInfo[next_item_ID][rotIdx]

            posX = 0
            while posX <= self.bin_dimension[0] - boundingSize[0]:
                posY = 0
                while posY <= self.bin_dimension[1] - boundingSize[1]:
                    X, Y = int(posX / self.resolutionAct), int(posY / self.resolutionAct)  # 对应heightmap的坐标变化
                    coorX, coorY = int(posX / self.resolutionH), int(posY / self.resolutionH)  # 对应heightmap的坐标变化
                    posZ = np.max((self.heightmapC[coorX: coorX + rangeX_OH, coorY: coorY + rangeY_OH]
                                   - heightMapB) * maskB)  # the lowest collision-free Z

                    heightmapC_Prime = np.max(
                        ((heightMapT + posZ) * maskH, self.heightmapC[coorX:coorX + rangeX_OH, coorY:coorY + rangeY_OH]), axis=0)
                    # maxHeight = np.max(heightmapC_Prime)
                    maxHeight = boundingSize[2] + posZ
                    # print(maxHeight)
                    if np.around(maxHeight - self.bin_dimension[2], decimals=6) > 0:
                        posY = np.round(delta_y + posY, decimals=3)
                        continue

                    scoreHM = 0
                    scoreDBLF = 0
                    if method == 'HM':
                        mapSum = np.sum(heightmapC_Prime)
                        scoreHM = heuristicC * (posX + posY) + mapSum * 100
                    if method == 'DBLF':
                        scoreDBLF = heuristicC * (posX + posY) + posZ * 100
                    candicadates.append([X, Y, posZ, scoreHM, scoreDBLF, rotIdx])

                    posY = np.round(delta_y + posY, decimals=3)

                posX = np.round(delta_x + posX, decimals=3)

        if len(candicadates) != 0:
            candicadates = np.round(np.array(candicadates), decimals=6)
            if method == 'RANDOM':
                action = candicadates[np.random.randint(0, len(candicadates))]
                return action.astype(np.int)
            elif method == 'DBLF':
                action = candicadates[np.argmin(candicadates[:, 4])]
                return action.astype(np.int)
            elif method == 'HM':
                action = candicadates[np.argmin(candicadates[:, 3])]
                return action.astype(np.int)
            elif method == 'MINZ':
                action = candicadates[np.argmin(candicadates[:, 2])]
                return action.astype(np.int)
            elif method == 'FIRSTFIT':
                action = candicadates[np.argmin(candicadates[:, 0] + candicadates[:, 1])]
                return action.astype(np.int)
        else:
            return None