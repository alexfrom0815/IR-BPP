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
    def __init__(self, bin_dimension, resolutionAct, resolutionH, boxPack = False,  ZRotNum = None, shotInfo = None, scale = None):
        self.bin_dimension = bin_dimension
        self.resolutionH = resolutionH
        self.resolutionAct = resolutionAct
        self.stepSize = int(self.resolutionAct / self.resolutionH)
        assert self.stepSize == self.resolutionAct / self.resolutionH
        self.rotNum = ZRotNum
        self.scale = scale
        self.rangeX_C, self.rangeY_C = np.ceil(bin_dimension[0:2] / resolutionH).astype(np.int32)
        self.rangeX_A, self.rangeY_A = np.ceil(bin_dimension[0:2] / resolutionAct).astype(np.int32)

        self.heightmapC = np.zeros((self.rangeX_C, self.rangeY_C))
        self.ray_origins, self.ray_directions = \
            gen_ray_origin_direction(self.rangeX_C, self.rangeY_C, resolutionH, boxPack, shift = 0.001)

        # self.pack_meshes = []
        self.shotInfo = shotInfo

        self.transformation = []
        DownFaceList, ZRotList = getRotationMatrix(1, ZRotNum)

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

    def place_item_trimesh(self, mesh, poseT, debugInfo):
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