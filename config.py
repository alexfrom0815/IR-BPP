import numpy as np
import torch

device = 0
# envName = "Boxpack-v0"
# envName = "Ir-v0"
envName = "Physics-v0"
triangleNum = 256
seed = 6
num_processes = 2
boxset = []
useHeightMap = False
visual = False
globalView = False
poseDist = False
distributed = False

# simulation = False
dataAugmentation = False
load_memory_path = None
save_memory_path = None
scale = [100, 100, 100] # fix it! don't change it!

# dataSample = 'pose' # instance, category, pose

# data_name = 'IR_apc_mass'
# data_name = 'tetris3D_tolerance_large_mass'

# data_name = 'BoxMeshRebuttal'
# data_name = 'IR_mix_mass'
data_name = 'tetris3D_tolerance_middle_mass'
# data_name = 'IR_concaveArea3_mass'
# data_name = 'IR_abc_good'

# data_name = 'IR_mix_nomin'
# data_name = 'tetris3D_tolerance_middle_nomin'
# data_name = 'IR_concaveArea3_nomin'

objPath = './data/final_data/{}/vhacd_with_pose'.format(data_name)
pointCloud  = './data/final_data/{}/pointCloud_with_pose'.format(data_name)
if 'IR_mix' in data_name:
    dicPath = './data/final_data/{}/dicPathHalf.pt'.format(data_name)
else:
    dicPath = './data/final_data/{}/dicPath.pt'.format(data_name)
# dataSample = 'pose' # instance, category, pose
if 'concave' in data_name:
    dataSample = 'category'
else:
    dataSample = 'instance'  # instance, category, pose

# dataSample = 'category' # instance, category, pose
# dataSample = 'instance' # instance, category, pose

if 'concave' in data_name:
    assert  dataSample == 'category'
if 'IR_mix' in data_name:
    assert 'dicPathHalf' in dicPath

# dicPath = './data/final_data/{}/dicPathGen.pt'.format(data_name)

# part = 0.01 # 0.8 0.5, 0.3 0.1 0.05 0.01
# dicPath = './data/final_data/{}/dicPathPart_{}.pt'.format(data_name, part)

if 'abc_mass' in data_name:
    meshScale = 0.8
else:
    meshScale = 1

fullData =  True
boundingBoxVec = False
boxPack = False
objVecLen = 9

# if 'Box' in dicPath:
#     bin_dimension = [0.3,  0.3, 0.3]
# elif 'tetris' in dicPath:
#     bin_dimension = [0.32, 0.32, 0.30]
# else:
bin_dimension = [0.32, 0.32, 0.30]

# resolutionA = 0.04
resolutionA = 0.02
# resolutionA = 0.03
# resolutionA = 0.01
# resolutionH = 0.005
# resolutionH = 0.002

if  'Block'  in dicPath:
# if  'Block' in dicPath or 'Box' in dicPath or 'tetris3D' in dicPath:
    resolutionA = 0.04

resolutionH = 0.01  # A / H should be a integer.

categories = 4
if fullData > 0:
    categories = len(torch.load(dicPath))
# categories = 20
# DownRotNum = 6 # Max: 6
DownRotNum = 1 # Max: 6

if 'IR' in dicPath:
    ZRotNum    = 8 # Max: 4/8
elif 'Box' in dicPath:
    ZRotNum    = 2 # Max: 4/8
else:
    ZRotNum    = 4 # Max: 4/8
# ZRotNum = ZRotNum * 2
heightMap = True
useHeightMap = True
globalView = False
stability  = False
poseDist   = False
shapePreType = 'SurfacePointsRandom' # MeshVertices, SurfacePoints, PreTrain, Index, Triangle, SurfacePointsRandom, SurfacePointsEncode, GlobalIndices
actionType = 'Uniform' # Uniform, RotAction, LineAction, HeuAction
rewardType = 'ratio'  # number, aabb, ratio
distributed = True
elementWise = False
if stability: assert globalView
simulation = True
# encoderPath = './checkpoints/encoder/encoder_official.pt'
# encoderPath = './checkpoints/encoder/encoder_onet_Best521.pt'
# encoderPath = './checkpoints/encoder/encoder_onet256meshpointsPretrained_Best6_8.pt'
# encoderPath = './checkpoints/encoder/encoder_uniform_ratio_1.pt'
# encoderPath = './checkpoints/encoder/encoder_complete_parts.pt'
# encoderPath = './checkpoints/encoder/encoder_complete_parts_contrastive.pt'
# encoderPath = './checkpoints/encoder/contrastive_only_no_noise.pt'
encoderPath = './checkpoints/encoder/encoder_complete_parts_contrastive_normalize_no_noise_final.pt'
# encoderPath = './checkpoints/encoder/encoder_complete_parts_new.pt'
# encoderPath = None
# pointCloud  = './data/pointClouds/tetris3D_tolerance_middle_pointcloud/'
# pointCloud  = './data/pointClouds/complete_parts/'
# pointCloud  = './data/pointClouds/my_sample_mesh/'
dataAugmentation = False
if dataAugmentation: assert objPath == './data/datas/BoxMeshTolerance'
assert resolutionA >= 0.001
heuristicExplore = False # This does not help the performance, forget it.
load_memory_path = None
save_memory_path = None
selectedAction = 500
# selectedAction = False
if selectedAction: assert  actionType == 'Uniform'
model = None
# model = './checkpoints/BlockLInUse_mass_vhacd_another-2022.08.01-17-47-45_30.pt'
originShape = False

hierachical = False
previewNum = 1
convexAction = 'ConvexVertex' # HullVertices, Contour, Defects, None, ConvexVertex

samplePointsNum = 1024
# samplePointsNum = 2048

# only used for debug
test = False
if dataSample == 'pose':
    test_name = './data/dataset/random_index_{}.pt'.format(categories)
else:
    # test_name = './data/final_data/{}/random_cate.pt'.format(data_name)
    test_name = './data/final_data/{}/random_cate_half.pt'.format(data_name)
maxBatch = 1
randomConvex = False
LFSS = False
heightResolution = 0.01

packed_holder = 100
enable_rotation = True
bin_dimension = np.round(bin_dimension, decimals=6)
