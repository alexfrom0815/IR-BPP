import torch

device = 1
# device = 'cpu'
# device = 0
envName = "Physics-v0"
triangleNum = 256
seed = 6
num_processes = 4
boxset = []
useHeightMap = False
globalView = False
poseDist = False
distributed = False


model = None

video = False
evaluation_episodes = 100
visual = False
rewardType = 'ratio' # number, aabb, ratio
# actionType = 'Uniform'  # Uniform, RotAction, LineAction, HeuAction, UniformTuple
actionType = 'Uniform'  # Uniform, RotAction, LineAction, HeuAction, UniformTuple
# dicPath = './data/dicPath/IrDicPath.pt'
# dicPath = './data/dicPath/BoxMeshToleranceDicPath.pt'
# dicPath = './data/dicPath/BlockMeshTolerance.pt'
dicPath = './data/dicPath/BlockLMiddle.pt'
# dicPath = './data/dicPath/tetris3D23.pt'
simulation = False
scale = [100, 100, 100]

if envName == 'Physics-v0':
    fullData = True

    boundingBoxVec = False
    boxPack = False
    objVecLen = 9
    categories = 4


    # data_name = 'IR_mix_mass'
    # data_name = 'blockout'
    # data_name = 'IR_concaveArea3_mass'
    data_name = 'IR_abc_good'

    globalView = True
    points_sigma = 0.0 # 0.00 0.01 0.03 0.05 0.1

    objPath = './data/final_data/{}/vhacd_with_pose'.format(data_name)
    if points_sigma!= 0:
        pointCloud  = './data/final_data/{}/pointCloud_with_pose_{}'.format(data_name, int(points_sigma * 100))
    else:
        pointCloud  = './data/final_data/{}/pointCloud_with_pose'.format(data_name)

    if 'abc_mass' in data_name:
        meshScale = 0.8
    else:
        meshScale = 1

    if 'IR_mix' in data_name:
        dicPath = './data/final_data/{}/dicPathHalf.pt'.format(data_name)
    else:
        dicPath = './data/final_data/{}/dicPath.pt'.format(data_name)

    # dataSample = 'pose' # instance, category, pose
    if 'concave' in data_name:
        dataSample = 'category'
    else:
        dataSample = 'instance'

    if fullData:
        categories = len(torch.load(dicPath))
    if dataSample == 'pose':
        test_name = './data/dataset/random_index_{}.pt'.format(categories)
    else:
        if 'IR_mix' in data_name:
            test_name = './data/final_data/{}/random_cate_half.pt'.format(data_name)
        else:
            test_name = './data/final_data/{}/random_cate.pt'.format(data_name)

    if 'IR_mix' in data_name:
        assert 'dicPathHalf' in dicPath
        assert  'half' in test_name
    if  dicPath == './data/dicPath/BlockLMiddle.pt':
        bin_dimension = [0.32, 0.32, 0.035]
    else:
        bin_dimension = [0.32, 0.32, 0.30]

    resolutionA = 0.02
    resolutionH = 0.01

    if  'Block' in dicPath or 'Box' in dicPath:
    # if  'Block' in dicPath or 'Box' in dicPath or 'tetris3D' in dicPath:
        resolutionA = 0.04

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

    stability = False
    poseDist = False
    distributed = False
    shapePreType = 'SurfacePointsRandom' # MeshVertices, SurfacePoints, PreTrain, Index, Triangle, SurfacePointsRandom
    # shapePreType = 'MeshVertices' # MeshVertices, SurfacePoints, PreTrain, Index, Triangle, SurfacePointsRandom
    elementWise = False
    simulation = True
    encoderPath = './checkpoints/encoder/encoderBest.pt'
    if stability: assert globalView
    # selectedAction = False
    selectedAction = 100
    convexAction = 'ConvexVertex' # HullVertices, Contour, Defects, None
    originShape = False

    hierachical = True # False 的话，就是 pi_s + dblf, 如果要求下层做决策的话，那么 hiera 要设置成true
    select_item_with_one_dqn = True
    previewNum = 10
    samplePointsNum = 1024
    heightResolution = 0.01
    if hierachical:
        orderModelPath = None
        locModelPath = './checkpoints/IR_abc_good_convex_var_20_12.pt'



packed_holder = 100
enable_rotation = True
if video: visual = True