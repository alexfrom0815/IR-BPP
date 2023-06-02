import os
# os.chdir('../')
print(os.getcwd())
import numpy as np
import trimesh
import time
import transforms3d
from environment.physics0.binPhy import PackingGame
# import sys
from arguments import get_args
import torch
from tools import load_shape_dict, shotInfoPre, draw_heatmap

def extendMat(mat3, translation = None):
    mat4 = np.eye(4)
    mat4[0:3,0:3] = mat3
    if translation is not None:
        mat4[0:3,3] = translation
    return mat4

bin_dimension = [0.32, 0.32, 0.30]

taskName = 'IR_mix_no_vhacd'
# taskName = 'tetris3D_tolerance_middle_tri_to_quad'
# taskName = 'IR_concaveArea3'
# taskName = 'IR_abc_good'
# taskName = 'all'

# 39 40 41 42 43 44 sigma nogen
# # 只有能画出来的是对的


if taskName == 'IR_mix_no_vhacd':
    folderList = [
        'IR_mix_mass_no_leak_dblf-2022.10.08-13-49-04',  # 0.0048223581442409826
        'IR_mix_mass_no_leak_first-2022.10.08-13-49-14', # 0.0030513569935337447
        'IR_mix_mass_no_leak_hm-2022.10.08-13-49-31',    # 0.0020609540140081924
        'IR_mix_mass_no_leak_minz-2022.10.08-13-49-39',  # 0.0015997817265853565
        'IR_mix_mass_no_leak_random-2022.10.08-13-49-51',# 0.0013499231866454194
        'IR_mix_mass_candi-2022.10.17-15-25-19',         # 0.00127953572570857
        'IR_mix_mass_convex_var_28_02-2022.10.11-16-06-10', # 0.0049128951115592736
        'IR_mix_mass_500_43_13-2022.09.16-15-40-41', # 0.005025015505252453
    ]

elif taskName == 'tetris3D_tolerance_middle_tri_to_quad':
    folderList = [
        # 'tetris3D_tolerance_middle_mass_convex_var_32_52-2022.10.11-16-06-50', # 0.002516526472746125
        # 'tetris3D_tolerance_middle_mass_no_leak_dblf-2022.10.08-13-50-05', # 0.0030863505439093653
        # 'tetris3D_tolerance_middle_mass_no_leak_first-2022.10.08-13-50-17', # 0.0016978695779041298
        # 'tetris3D_tolerance_middle_mass_no_leak_hm-2022.10.08-13-50-31', # 0.0014431839096910016
        # 'tetris3D_tolerance_middle_mass_no_leak_minz-2022.10.08-13-51-16', # 0.0011496681143806885
        # 'tetris3D_tolerance_middle_mass_no_leak_random-2022.10.08-13-51-28', # 0.0008764399918347559
        # 'tetris3D_tolerance_middle_mass_candi-2022.10.17-15-25-25', # 0.0007115306174566354
        # 'tetris3D_tolerance_middle_mass_500_38_48-2022.09.16-15-45-56', # 0.0006308721549010112
    ]

elif taskName == 'IR_concaveArea3':
    folderList = [
        # 'IR_concaveArea3_mass_convex_var_00_12-2022.10.11-16-05-25', # 0.0036460680665597344
        # 'IR_concaveArea3_mass_no_leak_dblf-2022.10.08-13-47-12',     # 0.002866939413597154
        # 'IR_concaveArea3_mass_no_leak_first-2022.10.08-13-47-25',    # 0.0029074638181500094
        # 'IR_concaveArea3_mass_no_leak_hm-2022.10.08-13-47-34',       # 0.002810334260259481
        # 'IR_concaveArea3_mass_no_leak_minz-2022.10.08-13-47-47',     # 0.0023794462279707195
        # 'IR_concaveArea3_mass_no_leak_random-2022.10.08-13-48-03',   # 0.0023004380702242116
        # 'IR_concaveArea3_mass_candi-2022.10.17-15-25-52',            # 0.0036808073661833557
        # 'IR_concaveArea3_mass_500_30_34-2022.09.16-15-46-30',        # 0.003669060428359324
    ]

elif taskName == 'IR_abc_good':
    folderList = [
        # 'IR_abc_good_convex_var_20_12-2022.10.11-16-04-40', # 0.003160004976268176
        # 'IR_abc_good_no_leak_dblf-2022.10.08-13-45-26',     # 0.0027608582316902552
        # 'IR_abc_good_no_leak_first-2022.10.08-13-45-42',    # 0.0019204666774423032
        # 'IR_abc_good_no_leak_hm-2022.10.08-13-45-56',       # 0.001668315698678117
        # 'IR_abc_good_no_leak_minz-2022.10.08-13-46-07',     # 0.0015032642560507878
        # 'IR_abc_good_no_leak_random-2022.10.08-13-46-18',   # 0.0012621059239487226
        # 'IR_abc_good_candi-2022.10.17-15-26-31',            # 0.0011701164734347952
        # 'IR_abc_good_500_43_57-2022.09.16-15-47-00',        # 0.001144684102966497
    ]


# f = open('results_{}_{}.txt'.format(taskName, timeStr),'a+')

envName = 'Physics-v0'
args = get_args()
args.envName = envName


triangleNum = 256
times = 2000
resolutionH = 0.01
resolutionAct = 0.01

args.enable_rotation = True
bin_dimension = [0.32, 0.32, 0.30]

args.boxPack = False

args.resolutionH = resolutionH
args.resolutionA = resolutionAct

if taskName == 'IR_mix_no_vhacd':
    data_name = 'IR_mix_mass'
    objPath = './final_data/IR_mix_mass/vhacd_with_pose'
elif taskName == 'tetris3D_tolerance_middle_tri_to_quad':
    data_name = 'tetris3D_tolerance_middle_mass'
    objPath = './final_data/tetris3D_tolerance_middle_mass/vhacd_with_pose'
elif taskName == 'IR_concaveArea3':
    data_name = 'IR_concaveArea3_mass'
    objPath = './final_data/IR_concaveArea3_mass/vhacd_with_pose'
elif taskName == 'IR_abc_good':
    data_name = 'IR_abc_good'
    objPath = './final_data/IR_abc_good/vhacd_with_pose'


args.objPath = './data/final_data/{}/vhacd_with_pose'.format(data_name)

if 'IR_mix' in data_name:
    args.dicPath = './data/final_data/{}/dicPathHalf.pt'.format(data_name)
else:
    args.dicPath = './data/final_data/{}/dicPath.pt'.format(data_name)

if 'concave' in data_name:  # 这里其实也不用, 因为数据集摆在这里
    dataSample = 'category'
else:
    dataSample = 'instance'

if 'IR_mix' in data_name:
    assert 'dicPathHalf' in args.dicPath

args.categories = len(torch.load(args.dicPath))
if dataSample == 'pose':
    test_name = './data/dataset/random_index_{}.pt'.format(args.categories)
else:
    if 'IR_mix' in data_name:
        test_name = './data/final_data/{}/random_cate_half.pt'.format(data_name)
    else:
        test_name = './data/final_data/{}/random_cate.pt'.format(data_name)

args.DownRotNum = 1
if 'IR' in args.dicPath:
    args.ZRotNum = 8  # Max: 4/8
elif 'Box' in args.dicPath:
    args.ZRotNum = 2  # Max: 4/8
else:
    args.ZRotNum = 4  # Max: 4/8

shapeDict, infoDict = load_shape_dict(args, True)

visual = False
test = False
simulation = True
dataname = None
packed_holder = 1

timeStr =  time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

args.shapeDict = shapeDict
args.bin_dimension = bin_dimension
args.resolutionH = resolutionH
args.actionType = 'Uniform'
shotInfo = shotInfoPre(args)

heightMap = True
useHeightMap = True

game = PackingGame(
    resolutionAct=resolutionAct,
    resolutionH=resolutionH,
    bin_dimension=bin_dimension,
    objPath=args.objPath,
    shapeDict=shapeDict,
    infoDict=infoDict,
    dicPath=args.dicPath,
    test=True,
    dataname=test_name,
    DownRotNum=args.DownRotNum,
    ZRotNum=args.ZRotNum,
    packed_holder=100,
    useHeightMap=useHeightMap,
    visual=visual,
    timeStr=timeStr,
    globalView=True,
    stability=False,
    shotInfo=shotInfo,
    simulation=simulation,
    heightMap=heightMap,
    actionType=args.actionType,
    scale=[100,100,100],
    dataSample=dataSample,
    selectedAction=100,
    convexAction='Defects',

)

# f = open('results_{}_{}.txt'.format(taskName, timeStr),'a+')
allTrajLength = []
for folder in folderList:
    print(folder + '\n')
    dataPath = './logs/evaluation/{}/trajs.npy'.format(folder)
    if os.path.exists(dataPath):
        print('File exists')
        # for objPath in objPathList:
        if True:
            try:
                trajs = np.load(dataPath, allow_pickle=True)

                print(folder, len(trajs))
                allEpisodeRatios = []
                allEpisodeLength = []
                allVar = []
                trajLengths = []

                # for trajIdx in range(len(trajs)):
                for trajIdx in range(100):

                    traj = trajs[trajIdx]

                    # for itemidx, item in enumerate(traj):
                    #     game.next_item_ID = item[0]
                    #     id = game.interface.addObject(game.dicPath[game.next_item_ID][0:-4],
                    #                                   targetFLB=np.array(item[2]),
                    #                                   rotation=np.array(item[3]),
                    #                                   linearDamping=0.5, angularDamping=0.5)
                    #
                    # game.space.shot_whole()
                    # var = np.var(game.space.heightmapC)
                    # allVar.append(var)

                    trajLengths.append(len(traj))

                # f.write(folder + '\n')
                # f.write(objPath + '\n')
                # f.write('{} {}\n'.format(
                #                              len(allEpisodeRatios),
                #                              np.mean(allVar) ))
                print(folder, objPath, np.mean(allVar))
                allTrajLength.append(trajLengths)

            except ValueError as e:
                pass

allTrajLength = np.array(allTrajLength)
meanTrajLength = np.mean(allTrajLength, 1)

print(np.mean(allTrajLength))
trajLable = np.where(allTrajLength > np.min(meanTrajLength), 1, 0)
trajLable = np.prod(trajLable, 0)
trajIndexs = np.where(trajLable > 0)[0]
print(trajLable)


for folder in folderList:
    print(folder + '\n')
    dataPath = './logs/evaluation/{}/trajs.npy'.format(folder)
    if os.path.exists(dataPath):
        print('File exists')
        if True:
            try:
                trajs = np.load(dataPath, allow_pickle=True)
                print(folder, len(trajs))
                allEpisodeRatios = []
                allEpisodeLength = []
                allVar = []

                # for trajIdx in range(100):
                for trajIdx in trajIndexs:
                    traj = trajs[trajIdx]

                    thisTrajLength = np.min(allTrajLength[:, trajIdx])

                    for itemidx in range(thisTrajLength):
                        item = traj[itemidx]
                        game.next_item_ID = item[0]
                        id = game.interface.addObject(game.dicPath[game.next_item_ID][0:-4],
                                                      targetFLB=np.array(item[2]),
                                                      rotation=np.array(item[3]),
                                                      linearDamping=0.5, angularDamping=0.5)

                    game.space.shot_whole()
                    var = np.var(game.space.heightmapC)
                    allVar.append(var)

                # f.write(folder + '\n')
                # f.write(objPath + '\n')
                # f.write('{} {}\n'.format(
                #                              len(allEpisodeRatios),
                #                              np.mean(allVar) ))
                print(folder, objPath, np.mean(allVar))

            except ValueError as e:
                pass