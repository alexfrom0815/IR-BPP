#--coding:utf-8--
import copy
import time

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
from pointnet import ResnetPointnet
import pybullet as p
import math
import gym
import torch.nn as nn

def load_mesh(path, init = 'BoundingBox', minArea = True):
    assert False
    mesh = trimesh.load(path)
    mesh.remove_unreferenced_vertices()
    transforms, probs = mesh.compute_stable_poses() # Computes stable orientations of a mesh and their quasi-static probabilities.
    mesh.apply_transform(transforms[0])

    if minArea:
        areaList = np.zeros((360))
        meshList = []
        Tz = extendMat(transforms3d.euler.euler2mat(0, 0, np.pi * 1 / 180, 'sxyz'))
        for i in range(360):
            areaList[i] = mesh.extents[0] * mesh.extents[1]
            meshList.append(mesh.copy())
            mesh.apply_transform(Tz)
        mesh = meshList[int(np.argmin(areaList))]

    if init == 'BoundingBox':  # Place the front-left-bottom point of object bounding box at origin.
        mesh.apply_translation(- mesh.bounds[0])
    else:
        assert False
    return [mesh]

def load_mesh_plain(path, DownRotNum, ZRotNum, init = 'Centroid', scale = 1):
    mesh = trimesh.load(path)
    # print('len', len(mesh.vertices))
    if scale != 1:
        mesh.apply_scale(scale)
    mesh.apply_translation(- mesh.centroid)
    meshList = []
    DownFaceList, ZRotList = getRotationMatrix(DownRotNum, ZRotNum)

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

def load_mesh_with_rotation(path, DownRotNum, ZRotNum, init = 'BoundingBox', minArea = True):
    assert False
    mesh = trimesh.load(path)
    mesh.remove_unreferenced_vertices()
    transforms, probs = mesh.compute_stable_poses() # Computes stable orientations of a mesh and their quasi-static probabilities.
    mesh.apply_transform(transforms[0])

    if minArea:
        areaList = np.zeros((360))
        meshList = []
        Tz = extendMat(transforms3d.euler.euler2mat(0, 0, np.pi * 1 / 180, 'sxyz'))
        for i in range(360):
            areaList[i] = mesh.extents[0] * mesh.extents[1]
            meshList.append(mesh.copy())
            mesh.apply_transform(Tz)
        mesh = meshList[int(np.argmin(areaList))]

    mesh.apply_translation(- mesh.centroid)

    meshList = []
    DownFaceList, ZRotList = getRotationMatrix(DownRotNum, ZRotNum)

    for d in DownFaceList:
        for z in ZRotList:
            tmpObj = mesh.copy()
            Transform = np.dot(z, d)
            tmpObj.apply_transform(Transform)

            if init == 'BoundingBox':  # Place the front-left-bottom point of object bounding box at origin.
                tmpObj.apply_translation(- tmpObj.bounds[0])
            else:
                assert False
            meshList.append(tmpObj)

    return meshList

def create_box(extents, init = 'BoundingBox'):
    mesh = trimesh.primitives.Box(extents=extents)
    mesh.apply_translation(- mesh.bounding_box.vertices[0])
    if init == 'BoundingBox':  # Place the front-left-bottom point of object bounding box at origin.
        mesh.apply_translation(- mesh.bounding_box.vertices[0])
    elif init == 'MassCenter': # Place the object's center of mass at origin
        mesh.apply_translation(- mesh.center_mass)
        assert False
    return mesh

def gen_ray_origin_direction(xRange, yRange, resolution_h, boxPack, shift = 0.001):

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

# Raycasting is used to build heightmaps
# heightMapB  a bottom-up heightmap of the object to be placed
# heightMapH  a top-down heightmap  of the object to be placed
# 这个函数还是存在一些问题的，比如mask是不对的
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

    # ray_visualize = trimesh.load_path(np.hstack((ray_origins, ray_origins + ray_directions * 1.1)).reshape(-1, 2, 3))
    # scene = trimesh.Scene([mesh, ray_visualize])
    # scene.show()
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

    # B_mask = np.zeros( xRange * yRange )
    # H_mask = np.zeros( xRange * yRange )
    # B_mask[index_rayB] = 1
    # H_mask[index_rayH] = 1
    # objectHeight = (heightMapH - heightMapB) * (B_mask * H_mask).reshape(xRange, yRange)
    # maskH = np.where(heightMapH > 0, 1, 0)

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

def add_bullet_object(
                  path,
                  translation = [0.0, 0.0, 0.0],
                  rotation = [0.0, 0.0, 0.0],
                  linearDamping = 0.1,
                  angularDamping = 0.1,
                  visual = False):

    translation = np.array(translation)

    if visual:
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                              fileName=path)
    else:
        visual_shape_id = None

    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName=path,
                                                collisionFramePosition=[0.0, 0.0, 0.0],
                                                meshScale=[1.0, 1.0, 1.0])

    assert len(rotation) == 3 or len(rotation) == 4
    if len(rotation) == 3:
        lenRot = math.sqrt(rotation[0] * rotation[0] + rotation[1] * rotation[1] + rotation[2] * rotation[2])
        rotation[0] *= math.sin(lenRot / 2) * lenRot
        rotation[1] *= math.sin(lenRot / 2) * lenRot
        rotation[2] *= math.sin(lenRot / 2) * lenRot
        rotation += [math.cos(lenRot / 2)]

    if visual:
        id = p.createMultiBody(basePosition=[0,0,0],
                               baseOrientation=rotation,
                               baseCollisionShapeIndex=collision_shape_id,
                               baseVisualShapeIndex=visual_shape_id,
                               useMaximalCoordinates=True)
    else:
        id = p.createMultiBody(basePosition=[0,0,0],
                               baseOrientation=rotation,
                               baseCollisionShapeIndex=collision_shape_id,
                               useMaximalCoordinates=True)

    AABBCompensation = np.array([0.002, 0.002, 0.002])
    AABB = p.getAABB(id)
    position, orientation = p.getBasePositionAndOrientation(id)
    assert False
    cen2FLB = position - (AABB[0] + AABBCompensation[0])
    p.resetBasePositionAndOrientation(id, translation + cen2FLB, orientation)
    p.changeDynamics(id, -1, linearDamping=linearDamping, angularDamping=angularDamping)
    return id

def shot_item_with_bullet(args, itemID, ray_origins_ini, start = [0,0,0]): # xRange, yRange the grid range.

    dicPath = args.dicPath
    shapeDict = torch.load(dicPath)

    loadPath = os.path.join(args.objPath, shapeDict[itemID])
    objID = add_bullet_object(loadPath)
    bounds = np.array(p.getAABB(objID))
    AABBCompensation = np.array([0.002, 0.002, 0.002])

    boundingSize = bounds[1] - bounds[0] - 2 * AABBCompensation
    xRange, yRange = np.ceil(boundingSize[0:2] / args.resolutionH).astype(np.int32)

    ray_origins = ray_origins_ini[start[0] : start[0] + xRange, start[1] : start[1] + yRange].copy().reshape((-1,3))
    ray_ends    = ray_origins_ini[start[0] : start[0] + xRange, start[1] : start[1] + yRange].copy().reshape((-1,3))
    ray_origins[:, 2] = bounds[1][2]
    ray_ends[:, 2] = 0

    intersections = p.rayTestBatch(ray_origins, ray_ends, numThreads=16)
    intersections = np.array(intersections, dtype=object)

    maskH = intersections[:, 0]
    maskH = np.where(maskH >= 0, 1, 0)

    heightMapH = np.zeros(xRange * yRange)
    if np.sum(maskH) != 0:
        fractions = intersections[:, 2]
        H = ray_origins[:, 2] + (ray_ends[:, 2] - ray_origins[:, 2]) * fractions
        heightMapH[:] = H
        heightMapH *= maskH
    else:
        heightMapH[:] = bounds[1][2] - AABBCompensation[2]
        maskH[:] = 1

    heightMapH = heightMapH.reshape((xRange, yRange))
    maskH = maskH.reshape((xRange, yRange))
    p.removeBody(objID)
    return heightMapH, maskH, boundingSize


def draw_container(bin_dimension):
    containerEdge = []

    containerEdge.append([0, 0, 0, bin_dimension[0], 0, 0])
    containerEdge.append([bin_dimension[0], 0, 0, bin_dimension[0], bin_dimension[1], 0])
    containerEdge.append([bin_dimension[0], bin_dimension[1], 0, 0, bin_dimension[1], 0])
    containerEdge.append([0, bin_dimension[1], 0, 0, 0, 0, ])

    containerEdge.append([0, 0, bin_dimension[2], bin_dimension[0], 0, bin_dimension[2]])
    containerEdge.append([bin_dimension[0], 0, bin_dimension[2], bin_dimension[0], bin_dimension[1], bin_dimension[2]])
    containerEdge.append([bin_dimension[0], bin_dimension[1], bin_dimension[2], 0, bin_dimension[1], bin_dimension[2]])
    containerEdge.append([0, bin_dimension[1], bin_dimension[2], 0, 0, bin_dimension[2]])

    containerEdge.append([0, 0, 0, 0, 0, bin_dimension[2]])
    containerEdge.append([bin_dimension[0], 0, 0, bin_dimension[0], 0, bin_dimension[2]])
    containerEdge.append([bin_dimension[0], bin_dimension[1], 0, bin_dimension[0], bin_dimension[1], bin_dimension[2]])
    containerEdge.append([0, bin_dimension[1], 0, 0, bin_dimension[1], bin_dimension[2]])

    ray_visualize = trimesh.load_path(np.array(containerEdge).reshape(-1, 2, 3))
    return ray_visualize

# def interp(index_rayB, objectXRange, objectYRange): # 插值很简单, 把最外面扩大一段, 然后上下左右移动一下, 加一圈就可以了
#     pass

# def score_candidates(**kwargs):
#     pass

def backup(timeStr, args):
    if args.evaluate:
        targetDir = os.path.join('./logs/evaluation', timeStr)
    else:
        targetDir = os.path.join('./logs/experiment', timeStr)

    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    copyfile('agent.py',  os.path.join(targetDir, 'agent.py'))
    copyfile('arguments.py',  os.path.join(targetDir, 'arguments.py'))
    copyfile('config.py',    os.path.join(targetDir, 'config.py'))
    copyfile('testconfig.py',    os.path.join(targetDir, 'testconfig.py'))
    copyfile('envs.py',    os.path.join(targetDir, 'envs.py'))
    copyfile('evaluation.py',    os.path.join(targetDir, 'evaluation.py'))
    copyfile('graph_encoder.py',    os.path.join(targetDir, 'graph_encoder.py'))

    copyfile('main.py',   os.path.join(targetDir, 'main.py'))
    copyfile('model.py',   os.path.join(targetDir, 'model.py'))
    copyfile('tools.py', os.path.join(targetDir, 'tools.py'))
    copyfile('trainer.py', os.path.join(targetDir, 'trainer.py'))
    copyfile('memory.py', os.path.join(targetDir, 'memory.py'))
    copyfile('pointnet.py', os.path.join(targetDir, 'pointnet.py'))

    gymPath = './environment'
    import config
    envName = config.envName.split('-v')
    envName = envName[0].lower() + envName[1]
    envPath = os.path.join(gymPath, envName)
    copytree(envPath, os.path.join(targetDir, envName))

def init(module, weight_init, bias_init, gain=1):
      weight_init(module.weight.data, gain=gain)
      bias_init(module.bias.data)
      return module

def registration_envs():
    register(
        id='Ir-v0',                                  # Format should be xxx-v0, xxx-v1
        entry_point='environment.ir0:PackingGame',  # Expalined in envs/__init__.py
    )
    register(
        id='Boxpack-v0',                                  # Format should be xxx-v0, xxx-v1
        entry_point='environment.boxpack0:PackingGame',  # Expalined in envs/__init__.py
    )
    register(
        id='Physics-v0',                                  # Format should be xxx-v0, xxx-v1
        entry_point='environment.physics0:PackingGame',   # Expalined in envs/__init__.py
    )
    register(
        id='Physics-v1',                                  # This repo change the FLB action to CenterAction
        entry_point='environment.physics1:PackingGame',
    )

def save_memory(memory, memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'wb') as pickle_file:
      pickle.dump(memory, pickle_file)
  else:
    with bz2.open(memory_path, 'wb') as zipped_pickle_file:
      pickle.dump(memory, zipped_pickle_file)

def load_memory(memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'rb') as pickle_file:
      return pickle.load(pickle_file)
  else:
    with bz2.open(memory_path, 'rb') as zipped_pickle_file:
      return pickle.load(zipped_pickle_file)

def load_policy(load_path, dqn):
    print(load_path)
    assert os.path.exists(load_path), 'File does not exist'
    load_dict = torch.load(load_path, map_location='cpu')
    dqn.load_state_dict(load_dict, strict=True)
    print('Loading pre-train upper model', load_path)
    return dqn

def draw_points(points):
    mesh = trimesh.points.PointCloud(points, [255,0,0,255])
    return mesh

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
        if args.shapePreType == 'Triangle':
            triangleNum = len(shapeDict[0][0].triangles)
            rotationNum = len(shapeDict[0])
            shapeArray = np.zeros((len(shapeDict), rotationNum, triangleNum, 12))
            for shapeIdx in shapeDict.keys():
                for rotIdx in range(rotationNum):
                    minP = shapeDict[shapeIdx][rotIdx].bounds[0]
                    newTriangles = shapeDict[shapeIdx][rotIdx].triangles.copy() - minP
                    for trindex in range(len(newTriangles)):
                        newTriangles[trindex] = sorted(newTriangles[trindex], key=lambda ems: (ems[2], ems[1], ems[0]), reverse=False)
                    triangles = newTriangles.reshape((1, triangleNum, -1))
                    face_normals = shapeDict[shapeIdx][rotIdx].face_normals.reshape((1, triangleNum, -1))
                    shapes = np.concatenate((triangles, face_normals), axis=2)
                    shapeArray[shapeIdx][rotIdx] = shapes
        else: # Present the current shape with pointNet
            if args.shapePreType == 'PreTrain':
                assert False # 目前初始化的方法没有确定
                shapeDict = torch.load(args.dicPath)
                pointCloudPath = args.pointCloud
                encoderName = args.encoderPath.split('/')[-1]
                if os.path.exists(os.path.join(args.pointCloud, encoderName)):
                    shapeArray = torch.load(os.path.join(args.pointCloud, encoderName))
                else:
                    pointsLen = 512
                    rotationNum = 1
                    shapeArray = np.zeros((len(shapeDict), rotationNum, pointsLen))
                    model = ResnetPointnet().to(args.device)
                    model.load_state_dict(torch.load(args.encoderPath))
                    for shapeIdx in shapeDict.keys():
                        for rotIdx in range(rotationNum):
                            # data = shapeDict[0].replace('.obj', '.npz')
                            data = shapeDict[0][0:-4] + '.npz'
                            data = np.load(os.path.join(pointCloudPath, data))['points']
                            data = torch.tensor(data).to(args.device).unsqueeze(0).to(torch.float32)
                            feature = model(data).cpu().detach().numpy()
                            shapeArray[shapeIdx][rotIdx] = feature
                    torch.save(shapeArray, os.path.join(args.pointCloud, encoderName))
            elif args.shapePreType == 'MeshVertices':
                maxVerticeNum = 0
                for shapeIdx in shapeDict.keys():
                    maxVerticeNum = max(len(shapeDict[shapeIdx][0].vertices), maxVerticeNum)
                rotationNum = len(shapeDict[0])
                shapeArray = np.zeros((len(shapeDict), rotationNum, maxVerticeNum, 3))
                for shapeIdx in shapeDict.keys():
                    for rotIdx in range(rotationNum):
                        minP = shapeDict[shapeIdx][rotIdx].bounds[0]
                        verticeNum = len(shapeDict[shapeIdx][rotIdx].vertices)
                        newVertices = shapeDict[shapeIdx][rotIdx].vertices.copy() - minP
                        extendVer = newVertices[0:1].repeat(maxVerticeNum - verticeNum, axis = 0)
                        vertices = np.concatenate([newVertices, extendVer], axis=0).reshape((1, maxVerticeNum, -1))
                        shapeArray[shapeIdx][rotIdx] = vertices
            elif args.shapePreType == 'SurfacePoints':
                VerticeNum = 1024
                rotationNum = len(shapeDict[0])
                shapeArray = np.zeros((len(shapeDict), rotationNum, VerticeNum, 3))
                for shapeIdx in shapeDict.keys():
                    for rotIdx in range(rotationNum):
                        minP = shapeDict[shapeIdx][rotIdx].bounds[0]
                        samplePoints = shapeDict[shapeIdx][rotIdx].sample(VerticeNum) - minP
                        shapeArray[shapeIdx][rotIdx] = samplePoints
            else:
                assert args.shapePreType == 'SurfacePointsRandom' or args.shapePreType == 'SurfacePointsEncode'
                shapeDict = torch.load(args.dicPath)
                pointCloudPath = args.pointCloud

                pointsNum = 100000
                shapeArray = np.zeros((len(shapeDict), pointsNum, 3))
                for shapeIdx in shapeDict.keys():
                    # data = shapeDict[shapeIdx].replace('.obj', '.npz')
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
    if not args.boxPack:
        shapeDict = torch.load(dicPath)
        for k in shapeDict.keys():
            if k >= args.categories: break
            loadPath = os.path.join(objPath, shapeDict[k])

            if args.enable_rotation:
                if args.envName == 'Ir-v0':
                    backDict[k] = load_mesh_with_rotation(loadPath, args.DownRotNum, args.ZRotNum, 'BoundingBox', )
                elif args.envName == 'Physics-v0':
                    backDict[k] = load_mesh_plain(loadPath, args.DownRotNum, args.ZRotNum, 'BoundingBox', scale)
            else:
                if args.envName == 'Ir-v0':
                    backDict[k] = load_mesh(loadPath, 'BoundingBox')
                elif args.envName == 'Physics-v0':
                    backDict[k] = load_mesh_plain(loadPath, args.DownRotNum, args.ZRotNum, 'BoundingBox', scale)

            infoDict[k] = []
            for idx in range(len(backDict[k])):
                infoDict[k].append({'volume': backDict[k][idx].volume, 'extents': backDict[k][idx].extents})
    else:
        x_range = int(args.bin_dimension[0] / (args.resolutionA * 2))
        y_range = int(args.bin_dimension[1] / (args.resolutionA * 2))
        z_range = int(args.bin_dimension[2] / (args.resolutionA * 2))
        extents = []
        for i in range(x_range):
            for j in range(y_range):
                for k in range(z_range):
                    extents.append([(i + 1) * args.resolutionA, (j + 1) * args.resolutionA, (k + 1) * args.resolutionA])
        for k in range(len(extents)):
            backDict[k] = [create_box(extents[k], 'BoundingBox')]
    if returnInfo:
        return backDict, infoDict
    else:
        return backDict

# def pose_distance(pose1, pose2, a = None):
#     # Di Gregorio R (2008) A novel point of view to define the distance
#     # between two rigid-body poses. In: Advances in Robot Kinematics:
#     # Analysis and Design, Springer, p 361–369
#     # http://www.boris-belousov.net/2016/12/01/quat-dist/
#     pose1 = np.array(pose1)
#     pose2 = np.array(pose2)
#
#     pos1, xyz1 = pose1[:3], pose1[3:]
#     pos2, xyz2 = pose2[:3], pose2[3:]
#
#     if a is None:
#         a = 0.28059 / 2.0
#     b = 1
#
#     m1 = transforms3d.euler.euler2mat(*xyz1, axes='sxyz')
#     m2 = transforms3d.euler.euler2mat(*xyz2, axes='sxyz')
#
#     diff_m = m1.transpose() * m2
#
#     r_diff = np.arccos( (diff_m.trace() - 1) / 2 )
#     t_diff = np.linalg.norm(pos1 - pos2)
#
#     r_diff = a * r_diff
#     t_diff = b * t_diff
#
#     diff = r_diff + t_diff
#
#     return diff

def pose_distance(pose1, pose2, a = None):
    # Di Gregorio R (2008) A novel point of view to define the distance
    # between two rigid-body poses. In: Advances in Robot Kinematics:
    # Analysis and Design, Springer, p 361–369
    # http://www.boris-belousov.net/2016/12/01/quat-dist/

    pos1, orien1 = pose1 # xyzw
    pos2, orien2 = pose2

    if a is None:
        a = 0.28059 / 2.0
    b = 1

    m1 = transforms3d.quaternions.quat2mat([orien1[3], *orien1[0:3]]) # wxyz
    m2 = transforms3d.quaternions.quat2mat([orien2[3], *orien2[0:3]])

    diff_m = m1.transpose() * m2

    r_diff = np.arccos( (diff_m.trace() - 1) / 2 )
    t_diff = np.linalg.norm(np.array(pos1) - np.array(pos2))

    r_diff = a * r_diff
    t_diff = b * t_diff

    diff = r_diff + t_diff

    return diff

def shotInfoPre(args, meshScale = 1):
    shapeDict = args.shapeDict
    rangeX_C = int(np.ceil(args.bin_dimension[0] / args.resolutionH))
    rangeY_C = int(np.ceil(args.bin_dimension[1] / args.resolutionH))
    ray_origins, ray_directions = gen_ray_origin_direction(rangeX_C, rangeY_C, args.resolutionH, args.boxPack)
    shotInfo = {}
    data_name = args.objPath.split('/')[-2]
    dicPath = args.dicPath.replace('.pt', '')
    dicPath = dicPath.split('/')[-1]
    if meshScale != 1:
        dataStorePath = os.path.join('./data/shotInfo', '{}_{}_{}_{}'.format(data_name, dicPath,  args.resolutionH, meshScale))
    else:
        dataStorePath = os.path.join('./data/shotInfo', '{}_{}_{}'.format(data_name, dicPath,  args.resolutionH))
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

def shotInfoPreArgs(args, meshScale = 1):
    shapeDict = args['shapeDict']
    rangeX_C = int(np.ceil(args['bin_dimension'][0] / args['resolutionH']))
    rangeY_C = int(np.ceil(args['bin_dimension'][1] / args['resolutionH']))
    ray_origins, ray_directions = gen_ray_origin_direction(rangeX_C, rangeY_C, args['resolutionH'], args['boxPack'])
    shotInfo = {}
    data_name = args['objPath'].split('/')[-2]
    dicPath = args['objPath'].replace('.pt', '')
    dicPath = dicPath.split('/')[-1]
    if meshScale != 1:
        dataStorePath = os.path.join('./data/shotInfo', '{}_{}_{}_{}'.format(data_name, dicPath,  args['resolutionH'], meshScale))
    else:
        dataStorePath = os.path.join('./data/shotInfo', '{}_{}_{}'.format(data_name, dicPath,  args['resolutionH']))
    if not os.path.exists(dataStorePath):
        os.makedirs(dataStorePath)
    for k in shapeDict.keys():
        if k >= args['categories']:
            break
        next_item = shapeDict[k]
        shotInfo[k] = []
        for rotIdx in range(len(next_item)):
            boundingSize = np.round(next_item[rotIdx].extents, decimals=6)
            rangeX_O, rangeY_O = np.ceil(boundingSize[0:2] / args['resolutionH']).astype(np.int32)
            subdataPath = os.path.join(dataStorePath, '{}_{}.pt'.format(k, rotIdx))
            if os.path.exists(subdataPath):
                heightMapT, heightMapB, maskH, maskB = torch.load(subdataPath)
            else:
                heightMapT, heightMapB, maskH, maskB = shot_item(next_item[rotIdx], ray_origins,
                                                             ray_directions, rangeX_O, rangeY_O)
                torch.save([heightMapT, heightMapB, maskH, maskB], subdataPath)
            shotInfo[k].append((heightMapT, heightMapB, maskH, maskB))
    return shotInfo

def shotInfoPreBullet(args):
    shapeDict = args.shapeDict
    rangeX_C = int(np.ceil(args.bin_dimension[0] / args.resolutionH))
    rangeY_C = int(np.ceil(args.bin_dimension[1] / args.resolutionH))
    ray_origins, ray_directions = gen_ray_origin_direction(rangeX_C, rangeY_C, args.resolutionH, args.boxPack)
    shotInfo = {}
    for k in shapeDict.keys():
        if k >= args.categories:
            break
        shotInfo[k] = []
        heightMapT, maskH, extent = shot_item_with_bullet(args, k, ray_origins)
        shotInfo[k].append((heightMapT, maskH, extent))
    return shotInfo

def decode_physic_only_with_heightmap(observation, args):
    next_item = observation[0]
    masks = observation[9: 9 + args.action_space]
    heightMap = observation[9 + args.action_space : ]
    return int(next_item), masks,  heightMap

def combine_new_observation(OldObservation, Newheightmap, Newmask, args):
    NewObservation = OldObservation.reshape((-1)).copy()
    NewObservation[9 + args.action_space : ] = Newheightmap
    if Newmask is not None:
        NewObservation[9: 9 + args.action_space] = Newmask
    return NewObservation

def substitute_masks_with_1(OldObservation, args):
    NewObservation = OldObservation.reshape((-1)).copy()
    NewObservation[9: 9 + args.action_space] = 1
    return NewObservation

# Flip alone X and Y axis, no mask operation
def dataAugmentation(state, action, args):
    rotNum = args.ZRotNum * args.DownRotNum
    next_item, masks, heightMap = decode_physic_only_with_heightmap(state, args)
    next_item = args.shapeDict[next_item]

    bin_dimension = args.bin_dimension
    rangeX_A, rangeY_A = np.ceil(bin_dimension[0:2] / args.resolutionA).astype(np.int32)
    rangeX_C, rangeY_C = np.ceil(bin_dimension[0:2] / args.resolutionH).astype(np.int32)

    assert rangeX_A == rangeY_A
    actionShape = (rangeX_A, rangeY_A)
    mapShape = (rangeX_C, rangeY_C)

    augStates = [state]
    augActions = [action]

    heightMap = heightMap.reshape(mapShape)

    # 先只考虑filp
    for XFlipFlag in range(2):
        mapFlipX = np.flip(heightMap.copy(), axis=0) if XFlipFlag == 1 else heightMap.copy()

        for YFlipFlag in range(2):
            if XFlipFlag == 0 and YFlipFlag == 0:
                continue

            mapFlipXY = np.flip(mapFlipX.copy(), axis=1) if YFlipFlag == 1 else mapFlipX.copy()

            # TODO: The action also need flip and rotation
            subAction = np.array(np.unravel_index(action, (rotNum, *actionShape)))
            if XFlipFlag == 1:
                subAction[1] = rangeX_A - 1 - subAction[1]
            if YFlipFlag == 1:
                subAction[2] = rangeY_A - 1 - subAction[2]

            for rotIdx in range(rotNum):
                boundingSize = np.round(next_item[rotIdx].extents, decimals=6)
                boundingSizeInt = np.ceil(boundingSize / args.resolutionA).astype(np.int32)

                # TODO, NOT COMPLETE
                FLBTransitionNeed = np.array([0, 0])
                if XFlipFlag == 1:
                    FLBTransitionNeed[0] = 1 - boundingSizeInt[0]
                if YFlipFlag == 1:
                    FLBTransitionNeed[1] = 1 - boundingSizeInt[1]

                if subAction[0] == rotIdx:
                    subAction[1:3] += FLBTransitionNeed

            subState = combine_new_observation(state, mapFlipXY.reshape(-1), None, args)
            subState = substitute_masks_with_1(subState, args)
            augStates.append(subState)
            subAction = np.ravel_multi_index(subAction, (rotNum, *actionShape))
            augActions.append(subAction)

    return augStates, augActions

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def get_mask_from_state(state, args, previewNum):
    actionNum = args.action_space

    if previewNum > 1:
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
def test(args, dqn, printInfo = False, videoName = None, timeStr = None, times = ''):
    env = make_eval_env(args)
    T_rewards, T_lengths, T_ratio, T_ratio_local = [], [], [], []
    all_episodes = []
    actionNum = env.action_space.n
    print('Evaluation Start')
    # Test performance over several episodes
    done = True
    dqn.online_net.eval()
    assert not dqn.online_net.training
    if videoName is not None:
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, videoName)

    placementTime = 0
    placementCounter = 0
    simulationTime = 0
    networkTime = 0
    for _ in range(args.evaluation_episodes):
        while True:
            s = time.time()
            if done:
                # state, reward_sum, done, episode_length = env.reset(), 0, False, 0
                state, reward_sum, done, episode_length = env.reset(), 0, False, 0
            state = torch.FloatTensor(state).reshape((1, -1)).to(args.device)
            mask = get_mask_from_state(state, args, args.previewNum)
            net_s = time.time()
            action = dqn.act_e_greedy(state, mask, -1)
            net_e = time.time()
            state, reward, done, _ = env.step(action.item())  # Step

            e = time.time()
            placementTime = placementTime + e - s
            networkTime = networkTime + net_e - net_s
            simulationTime = simulationTime + env.endSimulation - env.startSimulation
            placementCounter += 1
            print(placementTime / placementCounter, (placementTime - simulationTime) / placementCounter,
                  (placementTime - simulationTime - networkTime) / placementCounter, networkTime / placementCounter)

            reward_sum += reward
            episode_length += 1

            if done:
                ratio = env.get_ratio()
                T_ratio.append(ratio)
                occupancy = env.get_occupancy()
                T_ratio_local.append(ratio / occupancy)
                T_rewards.append(reward_sum)
                T_lengths.append(episode_length)
                if printInfo:
                    print('avg_reward:', np.mean(T_rewards))
                    print('avg_length:', np.mean(T_lengths))
                    print('var_reward:', np.var(T_rewards))
                    print('var_length:', np.var(T_lengths))

                    print('Mean Ratio:', np.mean(T_ratio))
                    print('Mean Ratio Local:', np.mean(T_ratio_local))
                    print('Var Ratio:', np.var(T_ratio))
                    print('Var Ratio Local:', np.var(T_ratio_local))
                    print('Episode {} Ratio {}'.format(env.item_creator.traj_index, reward_sum))
                all_episodes.append(copy.deepcopy( env.packed))
                np.save(os.path.join('./logs/evaluation', timeStr, 'trajs{}.npy'.format(times)), all_episodes)
                break

    if videoName is not None:
        p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
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
    dqn.online_net.train()
    assert dqn.online_net.training
    # Return average reward and Q-value
    return avg_reward, avg_length

def test_with_given_traj(args, dqn, printInfo = False, videoName = None, timeStr = None, times = ''):
    env = make_eval_env(args)
    env.saveTrajRecord = args.saveTrajRecord
    T_rewards, T_lengths, T_ratio, T_ratio_local = [], [], [], []
    all_episodes = []
    actionNum = env.action_space.n
    print('Evaluation Start')
    # Test performance over several episodes
    done = True
    dqn.online_net.eval()
    assert not dqn.online_net.training
    if videoName is not None:
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, videoName)
    placementTime = 0
    placementCounter = 0
    simulationTime = 0

    candidates = args.candidates
    for trajIdx  in candidates:
        while True:
            s = time.time()
            if done:
                # state, reward_sum, done, episode_length = env.reset(), 0, False, 0
                state, reward_sum, done, episode_length = env.reset(trajIdx + 1), 0, False, 0
            state = torch.FloatTensor(state).reshape((1, -1)).to(args.device)
            mask = get_mask_from_state(state, args, args.previewNum)
            action = dqn.act_e_greedy(state, mask, -1)
            state, reward, done, _ = env.step(action.item())  # Step

            e = time.time()
            placementTime = placementTime + e - s
            simulationTime = simulationTime + env.endSimulation - env.startSimulation
            placementCounter += 1
            print(placementTime / placementCounter, (placementTime - simulationTime) / placementCounter)

            reward_sum += reward
            episode_length += 1

            if done:
                ratio = env.get_ratio()
                T_ratio.append(ratio)
                occupancy = env.get_occupancy()
                T_ratio_local.append(ratio / occupancy)
                T_rewards.append(reward_sum)
                T_lengths.append(episode_length)
                if printInfo:
                    print('avg_reward:', np.mean(T_rewards))
                    print('avg_length:', np.mean(T_lengths))
                    print('var_reward:', np.var(T_rewards))
                    print('var_length:', np.var(T_lengths))

                    print('Mean Ratio:', np.mean(T_ratio))
                    print('Mean Ratio Local:', np.mean(T_ratio_local))
                    print('Var Ratio:', np.var(T_ratio))
                    print('Var Ratio Local:', np.var(T_ratio_local))
                    print('Episode {} Ratio {}'.format(env.item_creator.traj_index, reward_sum))
                all_episodes.append(copy.deepcopy( env.packed))
                np.save(os.path.join('./logs/evaluation', timeStr, 'trajs{}.npy'.format(times)), all_episodes)
                break

    if videoName is not None:
        p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
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
    dqn.online_net.train()
    assert dqn.online_net.training
    # Return average reward and Q-value
    return avg_reward, avg_length

# Test DQN
def test_hierachical_with_given_traj(args, dqns, printInfo = False, videoName = None, timeStr = None, times = ''):
    env = make_eval_env(args)
    env.saveTrajRecord = args.saveTrajRecord
    T_rewards, T_lengths, T_ratio, T_ratio_local = [], [], [], []
    all_episodes = []
    print('Evaluation Start')
    # Test performance over several episodes
    done = True

    for dqn in dqns:
        dqn.online_net.eval()
        assert not dqn.online_net.training
    orderDQN, locDQN = dqns

    if videoName is not None:
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, videoName)
    placementTime = 0
    placementCounter = 0
    simulationTime = 0


    candidates = args.candidates
    for trajIdx  in candidates:
        while True:
            s = time.time()
            if done:
                orderState, reward_sum, done, episode_length = env.reset(trajIdx + 1), 0, False, 0
            orderState = torch.FloatTensor(orderState).reshape((1, -1)).to(args.device)

            s3 = time.time()
            orderAction = orderDQN.act(orderState, None)
            s4 = time.time()

            locState = env.get_action_candidates(orderAction.cpu().numpy().astype(np.int)[0])
            locState = torch.from_numpy(np.array(locState)).float().to(args.device).reshape((1, -1))
            mask = get_mask_from_state(locState, args, 1)
            locAction = locDQN.act_e_greedy(locState, mask, -1)
            orderState, reward, done, _ = env.step(locAction.item())  # Step

            e = time.time()
            placementTime = placementTime + e - s
            # placementTime = placementTime + e - s - (s4 - s3)
            simulationTime = simulationTime + env.endSimulation - env.startSimulation
            placementCounter += 1
            print(placementTime / placementCounter, (placementTime - simulationTime) / placementCounter)

            reward_sum += reward
            episode_length += 1

            if done:
                ratio = env.get_ratio()
                T_ratio.append(ratio)
                occupancy = env.get_occupancy()
                T_ratio_local.append(ratio / occupancy)
                T_rewards.append(reward_sum)
                T_lengths.append(episode_length)
                all_episodes.append(copy.deepcopy( env.packed))
                if printInfo:
                    print('avg_reward:', np.mean(T_rewards))
                    print('avg_length:', np.mean(T_lengths))
                    print('var_reward:', np.var(T_rewards))
                    print('var_length:', np.var(T_lengths))

                    print('Mean Ratio:', np.mean(T_ratio))
                    print('Mean Ratio Local:', np.mean(T_ratio_local))
                    print('Var Ratio:', np.var(T_ratio))
                    print('Var Ratio Local:', np.var(T_ratio_local))
                    print('Episode {} Ratio {}'.format(env.item_creator.traj_index, reward_sum))

                np.save(os.path.join('./logs/evaluation', timeStr, 'trajs{}.npy'.format(times)), all_episodes)
                break

    if videoName is not None:
        p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
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

# Test DQN
def test_hierachical(args, dqns, printInfo = False, videoName = None, timeStr = None, times = ''):
    env = make_eval_env(args)
    T_rewards, T_lengths, T_ratio, T_ratio_local = [], [], [], []
    all_episodes = []
    print('Evaluation Start')
    # Test performance over several episodes
    done = True

    for dqn in dqns:
        dqn.online_net.eval()
        assert not dqn.online_net.training
    orderDQN, locDQN = dqns

    if videoName is not None:
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, videoName)
    placementTime = 0
    placementCounter = 0
    simulationTime = 0
    decisionTime = 0

    # todo should be a parameter
    select_item_with_one_dqn = args.select_item_with_one_dqn

    for _ in range(args.evaluation_episodes):
        while True:
            s = time.time()
            if done:
                orderState, reward_sum, done, episode_length = env.reset(), 0, False, 0
            orderState = torch.FloatTensor(orderState).reshape((1, -1)).to(args.device)

            if select_item_with_one_dqn:
                all_observations = env.get_all_possible_observation()
                all_observations = torch.from_numpy(np.array(all_observations)).to(args.device).reshape((args.previewNum,  -1))
                all_masks = []
                for obs in all_observations:
                    obs = obs.reshape((1,-1))
                    mask = get_mask_from_state(obs, args, 1)
                    all_masks.append(mask)
                all_masks = torch.cat(all_masks, dim=0)
                itemValue = locDQN.evalutate_item(all_observations.float(), all_masks.float())
                orderAction = torch.argmax(itemValue[0])
                locAction = itemValue[1][orderAction.item()]
            else:
                s1 = time.time()
                orderAction = orderDQN.act(orderState, None)
                s2 = time.time()

            locState = env.get_action_candidates(orderAction.cpu().numpy().astype(np.int)[0] if len(orderAction.shape) > 0 else orderAction.item())
            if not select_item_with_one_dqn:
                locState = torch.from_numpy(np.array(locState)).float().to(args.device).reshape((1, -1))
                mask = get_mask_from_state(locState, args, 1)
                locAction = locDQN.act_e_greedy(locState, mask, -1)
            s3 = time.time()

            orderState, reward, done, _ = env.step(locAction.item())  # Step

            # e = time.time()

            # placementTime = placementTime + e - s
            # placementTime = placementTime + e - s - (s4 - s3)
            # simulationTime = simulationTime + env.endSimulation - env.startSimulation

            decisionTime = decisionTime + s3 - s + env.action_stop - env.action_start
            # decisionTime = decisionTime + s3 - s + env.action_stop - env.action_start - (s2 - s1)

            placementCounter += 1
            print(decisionTime / placementCounter)
            # print(1, placementTime, simulationTime)
            # print(placementTime / placementCounter, (placementTime - simulationTime) / placementCounter)

            reward_sum += reward
            episode_length += 1

            if done:
                ratio = env.get_ratio()
                T_ratio.append(ratio)
                occupancy = env.get_occupancy()
                T_ratio_local.append(ratio / occupancy)
                T_rewards.append(reward_sum)
                T_lengths.append(episode_length)
                all_episodes.append(copy.deepcopy( env.packed))
                if printInfo:
                    print('avg_reward:', np.mean(T_rewards))
                    print('avg_length:', np.mean(T_lengths))
                    print('var_reward:', np.var(T_rewards))
                    print('var_length:', np.var(T_lengths))

                    print('Mean Ratio:', np.mean(T_ratio))
                    print('Mean Ratio Local:', np.mean(T_ratio_local))
                    print('Var Ratio:', np.var(T_ratio))
                    print('Var Ratio Local:', np.var(T_ratio_local))
                    print('Episode {} Ratio {}'.format(env.item_creator.traj_index, reward_sum))

                np.save(os.path.join('./logs/evaluation', timeStr, 'trajs{}.npy'.format(times)), all_episodes)
                break

    if videoName is not None:
        p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
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
    if 'Physics' in args.envName:
        env = gym.make(args.envName,
                       objPath=args.objPath,
                       resolutionAct=args.resolutionA,
                       resolutionH=args.resolutionH,
                       shapeDict=args.shapeDict,
                       infoDict=args.infoDict,
                       dicPath=args.dicPath,
                       enable_rotation=args.enable_rotation,
                       categories=args.categories,
                       bin_dimension=args.bin_dimension,
                       packed_holder=args.packed_holder,
                       boxPack=args.boxPack,
                       DownRotNum=args.DownRotNum,
                       ZRotNum=args.ZRotNum,
                       heightMap=args.heightMap,
                       useHeightMap=args.useHeightMap,
                       visual=args.visual,
                       globalView=args.globalView,
                       stability=args.stability,
                       poseDist=args.poseDist,
                       shotInfo=args.shotInfo,
                       rewardType=args.rewardType,
                       actionType=args.actionType,
                       elementWise=args.elementWise,
                       test=True,
                       dataname=args.test_name,
                       simulation = args.simulation,
                       scale = args.scale,
                       selectedAction = args.selectedAction,
                       convexAction = args.convexAction,
                       previewNum = args.previewNum,
                       dataSample = args.dataSample,
                       meshScale = args.meshScale,
                       heightResolution = args.heightResolution,
                       maxBatch = args.maxBatch,
                       timeStr = args.timeStr
        )

    # elif config.envName == 'Ir-v0':
    #     env = gym.make(config.envName,
    #                objPath=config.objPath,
    #                resolution_h=config.resolutionA,
    #                enable_rotation=config.enable_rotation,
    #                categories=config.categories,
    #                bin_dimension=config.bin_dimension,
    #                packed_holder=config.packed_holder,
    #                boxPack=config.boxPack,
    #                DownRotNum = 1,
    #                ZRotNum = 8
    #                )
    # elif config.envName == 'Boxpack-v0':
    #     env = gym.make(config.envName,
    #                container_size=config.bin_dimension,
    #                boxPack=config.boxPack
    #                )
    return env

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, device = 'cuda'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
