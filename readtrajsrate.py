import os

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
        # 'IR_mix_mass_500_43_13_1000-2022.11.03-15-38-55',  # baseline
        # 'IR_mix_mass_convex_var_28_02_1000-2022.11.03-15-39-43'  # var
        
        # 'IR_mix_mass_convex_var_hier_10_50_53_10-2022.10.11-16-10-30',
        # 'IR_mix_mass_convex_var_packit_18_23_10-2022.10.11-15-49-44',
        # 'IR_mix_mass_convex_var_lfss_40_45_10-2022.10.11-15-46-53',

        # 'IR_mix_mass_500_43_13_1000-2022.11.03-15-38-55', # baseline
        # 'IR_mix_mass_convex_var_28_02_1000-2022.11.03-15-39-43'  # var

        # 'IR_mix_mass_convex_var_28_02_static-2022.11.03-10-55-45',  # baseline
        # 'IR_mix_mass_convex_var_28_02_static-2022.11.03-10-54-32'  # var

        # 'IR_mix_mass_no_leak_dblf-2022.10.08-13-49-04', # 0.0048223581442409826
        # 'IR_mix_mass_no_leak_first-2022.10.08-13-49-14', # 0.0030513569935337447
        # 'IR_mix_mass_no_leak_hm-2022.10.08-13-49-31', # 0.0020609540140081924
        # 'IR_mix_mass_no_leak_minz-2022.10.08-13-49-39', # 0.0015997817265853565
        # 'IR_mix_mass_no_leak_random-2022.10.08-13-49-51', # 0.0013499231866454194
        # 'IR_mix_mass_candi-2022.10.17-15-25-19', # 0.00127953572570857

        # 'IR_mix_mass_convex_var_28_02-2022.10.11-16-06-10', # 0.0049128951115592736
        # 'IR_mix_mass_500_43_13-2022.09.16-15-40-41', # 0.005025015505252453
    ]

elif taskName == 'tetris3D_tolerance_middle_tri_to_quad':
    folderList = [
        'tetris3D_tolerance_middle_mass_convex_var_hier_10_52_21_10-2022.10.11-16-12-21',
        'tetris3D_tolerance_middle_mass_convex_var_packit_37_53_10-2022.10.11-15-50-55',
        'tetris3D_tolerance_middle_mass_var_lfss_46_33_10-2022.10.15-09-12-58',

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

        'IR_concaveArea3_mass_convex_var_lfss_39_51_10-2022.10.11-15-58-46',
        'IR_concaveArea3_mass_convex_var_hier_10_08_02_10-2022.10.11-16-09-10',
        'IR_concaveArea3_mass_convex_var_packit_30_50_10-2022.10.11-15-53-31',

        # 'IR_concaveArea3_mass_convex_var_00_12-2022.10.11-16-05-25', # 0.0036460680665597344
        # 'IR_concaveArea3_mass_no_leak_dblf-2022.10.08-13-47-12',     # 0.002866939413597154
        # 'IR_concaveArea3_mass_no_leak_first-2022.10.08-13-47-25',    # 0.0029074638181500094
        # 'IR_concaveArea3_mass_no_leak_hm-2022.10.08-13-47-34',       # 0.002810334260259481
        # 'IR_concaveArea3_mass_no_leak_minz-2022.10.08-13-47-47',     #
        # 'IR_concaveArea3_mass_no_leak_random-2022.10.08-13-48-03',   #
        # 'IR_concaveArea3_mass_candi-2022.10.17-15-25-52',            #
        # 'IR_concaveArea3_mass_500_30_34-2022.09.16-15-46-30',        #
    ]

elif taskName == 'IR_abc_good':
    folderList = [
        'IR_abc_good_convex_var_20_12-2022.10.11-16-04-40',
        'IR_abc_good_no_leak_dblf-2022.10.08-13-45-26',
        'IR_abc_good_no_leak_first-2022.10.08-13-45-42',
        'IR_abc_good_no_leak_hm-2022.10.08-13-45-56',
        'IR_abc_good_no_leak_minz-2022.10.08-13-46-07',
        'IR_abc_good_no_leak_random-2022.10.08-13-46-18',
        'IR_abc_good_candi-2022.10.17-15-26-31',
        'IR_abc_good_500_43_57-2022.09.16-15-47-00',
    ]
elif taskName == 'all':
    folderList = [
        ###############################################################
        # 'IR_concaveArea3_mass_convex_var_lfss_39_51_10-2022.10.11-15-58-46',
        # 'IR_concaveArea3_mass_convex_var_lfss_39_51_3-2022.10.11-15-59-04',
        # 'IR_concaveArea3_mass_convex_var_lfss_39_51_5-2022.10.11-15-58-54',
        #
        # 'tetris3D_tolerance_middle_mass_var_lfss_46_33_10-2022.10.15-09-12-58',
        # 'tetris3D_tolerance_middle_mass_var_lfss_46_33_5-2022.10.15-09-13-10',
        # 'tetris3D_tolerance_middle_mass_var_lfss_46_33_3-2022.10.15-09-13-19',

        # 'IR_concaveArea3_mass_convex_var_packit_30_50_10-2022.10.11-15-53-31',
        # 'IR_concaveArea3_mass_convex_var_packit_30_50_5-2022.10.11-15-54-35',
        # 'IR_concaveArea3_mass_convex_var_packit_30_50_3-2022.10.11-15-54-52',
        #
        # 'IR_mix_mass_convex_var_lfss_40_45_3-2022.10.11-15-47-28',
        # 'IR_mix_mass_convex_var_lfss_40_45_5-2022.10.11-15-47-08',
        # 'IR_mix_mass_convex_var_lfss_40_45_10-2022.10.11-15-46-53',
        # 'IR_mix_mass_convex_var_packit_18_23_3-2022.10.11-15-50-09',
        # 'IR_mix_mass_convex_var_packit_18_23_5-2022.10.11-15-50-00',
        # 'IR_mix_mass_convex_var_packit_18_23_10-2022.10.11-15-49-44',
        #
        # 'tetris3D_tolerance_middle_mass_convex_var_packit_37_53_3-2022.10.11-15-51-15',
        # 'tetris3D_tolerance_middle_mass_convex_var_packit_37_53_5-2022.10.11-15-51-06',
        # 'tetris3D_tolerance_middle_mass_convex_var_packit_37_53_10-2022.10.11-15-50-55',
        #
        # 'IR_concaveArea3_mass_convex_var_00_12_sigma_1-2022.10.13-13-33-58',
        # 'IR_concaveArea3_mass_convex_var_00_12_sigma_3-2022.10.13-13-34-08',
        # 'IR_concaveArea3_mass_convex_var_00_12_sigma_5-2022.10.13-13-34-17',
        # 'IR_concaveArea3_mass_convex_var_00_12_sigma_10-2022.10.13-13-34-37',
        #
        # 'tetris3D_tolerance_middle_mass_convex_var_32_52_sigma_1-2022.10.13-13-37-09',
        # 'tetris3D_tolerance_middle_mass_convex_var_32_52_sigma_3-2022.10.13-13-37-20',
        # 'tetris3D_tolerance_middle_mass_convex_var_32_52_sigma_5-2022.10.13-13-37-29',
        # 'tetris3D_tolerance_middle_mass_convex_var_32_52_sigma_10-2022.10.13-13-37-43',
        #
        # 'IR_concaveArea3_mass_convex_var_gen_07_44-2022.10.11-16-01-53',
        # 'IR_mix_mass_convex_var_1000_07_51-2022.10.14-09-38-57',
        # 'IR_mix_mass_convex_var_2048_points_22_26-2022.10.14-09-38-16',
        #
        # 'IR_mix_mass_convex_var_28_02_sigma_1-2022.10.11-23-15-06',
        # 'IR_mix_mass_convex_var_28_02_sigma_3-2022.10.11-23-14-56',
        # 'IR_mix_mass_convex_var_28_02_sigma_5-2022.10.11-23-14-40',
        # 'IR_mix_mass_convex_var_28_02_sigma_10-2022.10.11-23-14-28',
        # 'IR_mix_mass_convex_var_deltaz_005_07_53-2022.10.14-09-37-31',
        # 'IR_mix_mass_convex_var_double_rot_33_23-2022.10.14-16-01-53',
        # 'IR_mix_mass_convex_var_gen_04_01-2022.10.11-16-02-44',
        # 'IR_mix_mass_convex_var_no_distri_22_59-2022.10.11-23-16-56',
        # 'IR_mix_mass_convex_var_no_para_56_09-2022.10.11-23-17-13',
        # 'IR_mix_mass_convex_var_rainbow_47_19-2022.10.14-16-02-44',
        # 'IR_mix_mass_convex_var_resolutionA_04_11-2022.10.14-16-03-10',
        # 'IR_mix_mass_convex_var_resolutionH_43_34-2022.10.14-16-03-30',
        # 'IR_mix_nomin_convex_var_19_08-2022.10.11-23-17-46',
        # 'tetris3D_tolerance_middle_mass_convex_var_gen_10_14-2022.10.11-16-02-22',
        #
        # 'IR_abc_good_convex_var_hier_10_51_53_3-2022.10.11-16-08-19',
        # 'IR_abc_good_convex_var_hier_10_51_53_5-2022.10.11-16-08-06',
        # 'IR_abc_good_convex_var_hier_10_51_53_10-2022.10.11-16-07-56',
        #
        # 'IR_mix_mass_convex_var_hier_10_50_53_3-2022.10.11-16-10-46',
        # 'IR_mix_mass_convex_var_hier_10_50_53_5-2022.10.11-16-10-38',
        # 'IR_mix_mass_convex_var_hier_10_50_53_10-2022.10.11-16-10-30',
        #
        # 'IR_concaveArea3_mass_convex_var_hier_10_08_02_3-2022.10.11-16-09-28',
        # 'IR_concaveArea3_mass_convex_var_hier_10_08_02_5-2022.10.11-16-09-18',
        # 'IR_concaveArea3_mass_convex_var_hier_10_08_02_10-2022.10.11-16-09-10',
        #
        # 'tetris3D_tolerance_middle_mass_convex_var_hier_10_52_21_3-2022.10.11-16-12-47',
        # 'tetris3D_tolerance_middle_mass_convex_var_hier_10_52_21_5-2022.10.11-16-12-33',
        # 'tetris3D_tolerance_middle_mass_convex_var_hier_10_52_21_10-2022.10.11-16-12-21',
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
objPath = './data/final_data/{}/vhacd_with_pose'.format(data_name)


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
    convexAction='Defects'
)

f = open('results_{}_{}.txt'.format(taskName, timeStr),'a+')

selected = 1000
lengthVector = np.zeros((len(folderList), selected))

# for i in range(len(folderList)):
#     folder = folderList[i]
#     dataPath = './logs/evaluation/{}/trajs.npy'.format(folder)
#     trajs = np.load(dataPath, allow_pickle=True)
#     for trajIdx in range(selected):
#         traj = trajs[trajIdx]
#         lengthVector[i][selected] = len(traj)

meanLength = np.mean(lengthVector, axis=0)

fullEpisodeRate = []
for folder in folderList:
    dataPath = './logs/evaluation/{}/trajs.npy'.format(folder)
    if os.path.exists(dataPath):
                trajs = np.load(dataPath, allow_pickle=True)
                allEpisodeRatios = []
                allEpisodeLength = []
                allRatio = []

                for trajIdx in range(selected):
                    traj = trajs[trajIdx]

                    thisEpisodeRatio = []
                    thisEpisodeVolume = []
                    thisEpisodeBullet = []

                    for itemid in range(len(traj)-1):
                        item = traj[itemid]
                        game.next_item_ID = item[0]

                        id = game.interface.addObject(game.dicPath[game.next_item_ID][0:-4],
                                                      targetFLB=np.array(item[2]),
                                                      rotation=np.array(item[3]),
                                                      linearDamping=0.5, angularDamping=0.5)
                        thisEpisodeBullet.append(id)
                        name = item[1]
                        mesh = trimesh.load(os.path.join(objPath, name))
                        thisEpisodeVolume.append(mesh.volume)

                        game.space.shot_whole()
                        binvolume = np.prod(np.array(bin_dimension))
                        # ratio = np.sum(game.space.heightmapC) / (np.prod(game.space.heightmapC.shape) * bin_dimension[2])
                        ratio = np.minimum(game.space.heightmapC / bin_dimension[2], 1)
                        ratio = np.mean(ratio)

                        relativeRatio = np.mean(game.space.heightmapC / bin_dimension[2])
                        thisRatio = np.sum(thisEpisodeVolume) / binvolume

                        fillRatio = thisRatio / relativeRatio

                        variance = np.var(game.space.heightmapC)
                        thisEpisodeRatio.append([ratio, fillRatio, variance])

                        print([ratio, fillRatio])

                    for rmId in thisEpisodeBullet:
                        game.interface.removeBody(rmId)

                    print(np.max(game.space.heightmapC))
                    print(thisEpisodeRatio[-1])
                    allRatio.append(thisEpisodeRatio)

                # fullEpisodeRate.append(allRatio)

                torch.save(allRatio, 'ratio_{}.pt'.format(folder))

                f.write(folder + '\n')
                f.write(objPath + '\n')
                f.write('{} {}\n'.format(len(allEpisodeRatios),
                                         np.mean(allRatio) ))
                print(folder, objPath, np.mean(allRatio))

