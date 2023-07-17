import trimesh,os,math
import pybullet as p
import numpy as np

def extendMat(mat3, translation = None):
    mat4 = np.eye(4)
    mat4[0:3,0:3] = mat3
    if translation is not None:
        mat4[0:3,3] = translation
    return mat4

class Interface:
    
    def __init__(self, bin = [10, 10, 5],
                 foldername = '../dataset/datas/128',
                 visual = False,
                 scale = [1.0,1.0,1.0],
                 simulationScale = None,
                 maxBatch = 2,
                 ):
        self.foldername = foldername
        if not os.path.exists(self.foldername):
            os.mkdir(foldername)

        cid = p.connect(p.SHARED_MEMORY)
        self.visual = visual

        if (cid < 0):
            if self.visual:
                p.connect(p.GUI)
            else:
                p.connect(p.DIRECT)

        self.defaultScale = scale.copy()
        if simulationScale is None:
            self.simulationScale = 1
        else:
            self.simulationScale = simulationScale
        self.bin = np.array(bin)
        self.bin = np.round(self.bin * self.defaultScale, decimals=6)
        self.shapeMap = {}
        self.objs = []
        self.objsDynamic = []
        self.g = [0.0, 0.0, -10.0]
        p.setGravity(self.g[0], self.g[1], self.g[2])
        p.setPhysicsEngineParameter(constraintSolverType=p.CONSTRAINT_SOLVER_LCP_PGS, globalCFM = 0.0001, numSolverIterations=10)

        if self.visual:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

        self.containerFolder = self.foldername + '/../box_{}_{}_{}'.format(*self.bin)
        if not os.path.exists(self.containerFolder):
            os.mkdir(self.containerFolder)
        self.addBox(self.bin, [1,1,1], [0, 0, 0])

        if self.visual:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        # p.setRealTimeSimulation(1)

        self.AABBCompensation = np.array([0.002, 0.002, 0.002])
        self.cameraForRecord()
        self.meshDict = {}
        self.maxBatch = maxBatch
    def close(self):
        p.disconnect()

    def removeBody(self, delId):
        p.removeBody(delId)
        if delId in self.objsDynamic:
            self.objsDynamic.remove(delId)

    def reset(self):
        for id in set(self.objs + self.objsDynamic):
            p.removeBody(id)
        bodyNum = p.getNumBodies()

        if bodyNum > self.boxNum:
            for i in range(bodyNum):
                itemId = p.getBodyUniqueId(i)
                if itemId > self.boxNum-1:
                    p.removeBody(itemId)

        self.objs = []
        self.objsDynamic = []
        self.meshDict = {}

    def getAllPositionAndOrientation(self, inner = True):
        positions = []
        orientations = []

        for id in self.objs:
            position, orientation = self.get_Wraped_Position_And_Orientation(id, inner)
            positions.append(position)
            orientations.append(orientation)

        return positions, orientations

    def makeBox(self, bin, color, thick = 1):
        box = []

        left = trimesh.primitives.Box(extents = np.array([thick, bin[1]+thick*2, bin[2]]))
        left.apply_translation(np.array([-thick/2, bin[1]/2, bin[2]/2]))
        left.visual.face_colors = color
        
        right = trimesh.primitives.Box(extents = np.array([thick, bin[1]+thick*2, bin[2]]))
        right.apply_translation(np.array([bin[0]+thick/2, bin[1]/2, bin[2]/2]))
        right.visual.face_colors = color
        
        front = trimesh.primitives.Box(extents = np.array([bin[0], thick, bin[2]]))
        front.apply_translation(np.array([bin[0]/2, bin[1]+thick/2, bin[2]/2]))
        front.visual.face_colors = color
        
        back = trimesh.primitives.Box(extents = np.array([bin[0], thick, bin[2]]))
        back.apply_translation(np.array([bin[0]/2, -thick/2, bin[2]/2]))
        back.visual.face_colors = color
        
        bottom = trimesh.primitives.Box(extents = np.array([bin[0]+thick*2, bin[1]+thick*2, thick]))
        bottom.apply_translation(np.array([bin[0]/2, bin[1]/2, -thick/2]))
        bottom.visual.face_colors = color

        box.append(bottom)
        box.append(left)
        box.append(right)
        box.append(front)
        box.append(back)
        return box

    def addBox(self, bin, scale, shift):
        box = self.makeBox(bin, color = [0.6, 0.3, 0.1, 1])

        counter = 0

        for index, side in enumerate(box):
            if index == 0:
                repeat = 5
            else:
                repeat = 1

            boxPath = os.path.join(self.containerFolder, 'Box' + str(index) + '.obj')
            if not os.path.exists(boxPath):
                side.export(boxPath)

            if self.visual:
                visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                      fileName=boxPath,
                                                      rgbaColor = [0.6, 0.3, 0.1, 1],
                                                      specularColor = [0.4, .4, 0],
                                                      visualFramePosition=shift,
                                                      # meshScale=scale
                                                      )

            collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                        fileName=boxPath,
                                                        collisionFramePosition=shift,
                                                        flags = 1,
                                                        )

            for _ in range(repeat):
                if self.visual:
                    boxID = p.createMultiBody(baseMass=0,
                                  baseInertialFramePosition=[0, 0, 0],
                                  baseCollisionShapeIndex=collision_shape_id,
                                  baseVisualShapeIndex=visual_shape_id,
                                  useMaximalCoordinates=True)
                else:
                    boxID = p.createMultiBody(baseMass=0,
                                  baseInertialFramePosition=[0, 0, 0],
                                  baseCollisionShapeIndex=collision_shape_id,
                                  useMaximalCoordinates=True)

                p.changeDynamics(boxID, -1,
                                 contactProcessingThreshold = 0,
                                 )
                counter += 1
        self.boxNum = counter

    def overlap2d(self, minC, maxC, minC2, maxC2):
        for d in range(2):
            if minC2[d] > maxC[d] or minC[d]  > maxC2[d]:
                return False
        return True

    def adjustHeight(self, newId, height):
        height = height * self.defaultScale[2]
        self.reset_Height(newId, height)

    def addObject(self, name,
                  targetFLB = [0.0, 0.0, 0.0],
                  rotation = [0.0, 0.0, 0.0],
                  scale = None,
                  density = 1.0,
                  linearDamping = 0.1,
                  angularDamping = 0.1,
                  path = None,
                  color = None
                  ):
        if scale is None: scale = self.defaultScale

        targetFLB = np.array(targetFLB) * scale
        if name in self.shapeMap:
            mesh, visual_shape_id, collision_shape_id = self.shapeMap[name]
        else:

            objPath = path if path is not None else self.foldername+ "/" + name + ".obj"
            mesh = trimesh.load(objPath)
            mesh.apply_scale(scale[0]*self.simulationScale)
            # mass = mesh.volume * density
            if self.visual:
                visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                      fileName=objPath,
                                                      meshScale=scale * self.simulationScale,
                                                      )
            else:
                visual_shape_id = None

            collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                  fileName=objPath,
                                                  collisionFramePosition=[0.0, 0.0, 0.0],
                                                  meshScale=scale * self.simulationScale)
            self.shapeMap[name] = (mesh, visual_shape_id, collision_shape_id)
        if self.visual and color is not None:
                objPath = path if path is not None else self.foldername + "/" + name + ".obj"
                visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                      fileName=objPath,
                                                      meshScale=scale * self.simulationScale,
                                                      rgbaColor=color)

        assert len(rotation) == 3 or len(rotation) == 4
        if len(rotation) == 3:
            lenRot = math.sqrt(rotation[0] * rotation[0] + rotation[1] * rotation[1] + rotation[2] * rotation[2])
            rotation[0] *= math.sin(lenRot/2) * lenRot
            rotation[1] *= math.sin(lenRot/2) * lenRot
            rotation[2] *= math.sin(lenRot/2) * lenRot
            rotation += [math.cos(lenRot/2)]

        mass = mesh.volume
        if self.visual:
            id = p.createMultiBody(baseMass=mass,
                                   basePosition=[-100, -100, -100],
                                   baseOrientation=rotation,
                                   baseCollisionShapeIndex=collision_shape_id,
                                   baseVisualShapeIndex=visual_shape_id,
                                   useMaximalCoordinates=True)
        else:
            id = p.createMultiBody(baseMass=mass,
                                   basePosition=[-100,-100,-100],
                                   baseOrientation=rotation,
                                   baseCollisionShapeIndex=collision_shape_id,
                                   useMaximalCoordinates=True)
        self.meshDict[id] = mesh
        self.reset_Wraped_Position_And_Orientation(id, targetFLB)

        p.changeDynamics(id, -1,
                         linearDamping = linearDamping,
                         angularDamping = angularDamping,
                         contactProcessingThreshold = 0,
                         )

        if density>0:
            self.objsDynamic.append(id)
        self.objs.append(id)
        return id

    def simulatePlain(self, batch = 1.0, dt = 0.01, maxBatch = 1):
        for _ in range(maxBatch):
            for i in range(int(batch/dt)):
                p.stepSimulation()

    def simulateToQuasistatic(self, givenId = None, linearTol = 0.001, angularTol = 0.001, batch = 1.0, dt = 0.01, maxBatch = 5):
        end = False
        linearTolSqr = linearTol * linearTol
        angularTolSqr = angularTol * angularTol
        batchCounter = 0

        # while not end:
        for _ in range(maxBatch):
            if end: break
            batchCounter += 1
            # simulation a batch
            for i in range(int(batch/dt)):
                p.stepSimulation()
            # test
            end = True

            if givenId is not None:
                id_List = [givenId]
            else:
                id_List = self.objsDynamic

            for id in id_List:
                linear, angular = p.getBaseVelocity(id)

                if linear[0] * linear[0] + linear[1] * linear[1] + linear[2] * linear[2] > linearTolSqr:
                    end = False
                if angular[0] * angular[0] + angular[1] * angular[1] + angular[2] * angular[2] > angularTolSqr:
                    end = False

                minC, maxC = self.get_wraped_AABB(id)

                midC = (maxC - minC) / 2 + minC
                if midC[0] <= 0 or midC[0] - self.bin[0] >= 0 or \
                   midC[1] <= 0 or midC[1] - self.bin[1] >= 0 or \
                   midC[2] <= 0:
                    # print('midC out of bounds', midC)
                    return True, False


        return True, True

    def simulateToQuasistaticRecord(self, givenId = None, linearTol = 0.001,
                              angularTol = 0.001, batch = 1.0, dt = 0.01, maxBatch = 5,
                                    id_List = [], returnRecord = True):
        end = False
        linearTolSqr = linearTol * linearTol
        angularTolSqr = angularTol * angularTol
        batchCounter = 0

        recordList = []

        # while not end:
        for _ in range(maxBatch):
            if end: break
            batchCounter += 1
            # simulation a batch
            for i in range(int(batch/dt)):
                p.stepSimulation()
                recordForThisTime = []
                print(len(id_List))
                for id in id_List:
                    positionT, orientationT = self.get_Wraped_Position_And_Orientation(id, inner=False)
                    recordForThisTime.append([positionT, orientationT])
                recordList.append(recordForThisTime)
        return recordList


    def secondSimulation(self, linearTol = 0.001, angularTol = 0.001, batch = 1.0, dt = 0.01, maxBatch = 5):
        self.enableObjects()

        end = False
        linearTolSqr = linearTol * linearTol
        angularTolSqr = angularTol * angularTol
        batchCounter = 0

        # while not end:
        for _ in range(maxBatch):
            if end: break
            batchCounter += 1
            # simulation a batch
            for i in range(int(batch/dt)):
                p.stepSimulation()
            # test
            end = True
            for id in self.objsDynamic:
                linear, angular = p.getBaseVelocity(id)
                if linear[0] * linear[0] + linear[1] * linear[1] + linear[2] * linear[2] > linearTolSqr:
                    end = False
                if angular[0] * angular[0] + angular[1] * angular[1] + angular[2] * angular[2] > angularTolSqr:
                    end = False

        self.disableAllObject()


    def simulateHeight(self, id):
        minC, maxC = self.get_wraped_AABB(id)
        if np.round(maxC[2] - self.bin[2], decimals=6) > 0:
                return False, True
        return True, True

    def disableObject(self, id, targetZ = None):
        if targetZ is not None:
            self.reset_Height(id, targetZ)
        p.changeDynamics(id, -1, mass = 0.0)
        self.objsDynamic.remove(id)

    def enableObjects(self):
        for id in self.objs:
            p.changeDynamics(id, -1, self.meshDict[id].volume)
            self.objsDynamic.append(id)

    def disableAllObject(self):
        for id in self.objsDynamic:
            p.changeDynamics(id, -1, mass=0.0)
            self.objsDynamic.remove(id)

    def cameraForRecord(self):
        target = [self.bin[0] / 2, self.bin[1] / 2, self.bin[2] / 2]
        self.setupCamera(target = target, dist = 0.38 * self.defaultScale[0],
                         yaw = 90.0, pitch = -85.0)

    def setupCamera(self, target, position = None, dist = None, yaw = None, pitch = None):
        if position is not None:
            dir = [target[0] - position[0], target[1] - position[1], target[2] - position[2]]
            dist = math.sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
            distxy = math.sqrt(dir[0] * dir[0] + dir[1] * dir[1])
            yaw = math.atan2(dir[1] ,dir[0]) * 180 / math.pi
            pitch = math.atan2(dir[2], distxy) * 180 / math.pi
        p.resetDebugVisualizerCamera(cameraDistance = dist, 
                                     cameraYaw = yaw, 
                                     cameraPitch = pitch, 
                                     cameraTargetPosition = target)
        return dist, yaw, pitch, target

    def get_wraped_AABB(self, id, inner = True):
        return self.get_trimesh_AABB(id, inner)

    def get_Wraped_Position_And_Orientation(self, id, inner = True, getPosBase = False):
        return self.get_trimesh_Position_And_Orientation(id, inner, getPosBase)


    def reset_Wraped_Position_And_Orientation(self, id, targetFLB, targetOrientation = None):
        self.reset_trimesh_Position_And_Orientation(id, targetFLB, targetOrientation)


    def reset_Height(self, id, targetHeight):
        self.reset_trimesh_height(id, targetHeight)

    def get_trimesh_AABB(self, id, inner = True):
        positionBase, orientationT = p.getBasePositionAndOrientation(id)
        mesh = self.meshDict[id].copy()
        mat = p.getMatrixFromQuaternion(orientationT)
        mesh.apply_transform(extendMat(np.array(mat).reshape((3,3)), positionBase))
        bounds = mesh.bounds
        if not inner:
            bounds = bounds / self.defaultScale
        return bounds

    def get_trimesh_Position_And_Orientation(self, id, inner = True, getPosBase = False):

        positionBase, orientationT = p.getBasePositionAndOrientation(id)
        mesh = self.meshDict[id].copy()
        mat = p.getMatrixFromQuaternion(orientationT)
        mesh.apply_transform(extendMat(np.array(mat).reshape((3,3)), positionBase))
        bounds = mesh.bounds

        if not inner:
            bounds = bounds / self.defaultScale

        positionFLB = bounds[0]
        returnList = [np.array(positionFLB), np.array(orientationT)]
        if getPosBase:
            returnList.append(np.array(positionBase))

        return returnList

    def reset_trimesh_Position_And_Orientation(self, id, targetFLB, targetOrientation = None):
        if targetOrientation is not None:
            p.resetBasePositionAndOrientation(id, [-100,-100,-100], targetOrientation)
        positionFLB, orientationT, positionBase = self.get_trimesh_Position_And_Orientation(id, inner=True, getPosBase=True)
        positionTarget = targetFLB - positionFLB + positionBase
        p.resetBasePositionAndOrientation(id, positionTarget, orientationT)


    def reset_trimesh_height(self, id, targetHeight):
        positionFLB, orientationT, positionBase = self.get_trimesh_Position_And_Orientation(id, inner=True, getPosBase=True)
        positionHeight = targetHeight - positionFLB[2] + positionBase[2]
        p.resetBasePositionAndOrientation(id, [*positionBase[0:2], positionHeight], orientationT)

    def reset_trimesh_Position_And_Orientation_new(self, id, targetFLB, targetOrientation = None):
        mesh = self.meshDict[id].copy()
        mat = p.getMatrixFromQuaternion(targetOrientation)
        mesh.apply_transform(extendMat(np.array(mat).reshape((3,3))))
        positionTarget = targetFLB - mesh.bounds[0]
        p.resetBasePositionAndOrientation(id, positionTarget, targetOrientation)
