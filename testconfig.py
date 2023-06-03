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

# model = './checkpoints/resolution_0002_8rot-2022.05.07-08-44-53_10.pt'
# model = './checkpoints/r001_rot-2022.05.12-14-58-40_3.pt'
# model = './checkpoints/BoxMeshPack2Rot-2022.05.14-18-42-19.pt'
# model = './checkpoints/BoxMeshPack2Rot-2022.05.14-18-42-19-20.pt'
# model = './checkpoints/BlockMeshToleranceLarge-2022.05.17-21-01-45_21.pt'
# model = './checkpoints/lineRescale_025_050-2022.05.14-16-03-16.pt'
# model = './checkpoints/BlockMeshToleranceMiddle-2022.05.19-18-33-57_5.pt'
# model = './checkpoints/BlockMeshToleranceMiddle-2022.05.19-18-33-57_14.pt'
# model = './checkpoints/BlockMeshToleranceMiddleNoSim-2022.05.20-11-21-22_11.pt'
# model = './checkpoints/BlockMeshTolerance-2022.05.17-20-16-20.pt'
# model = './checkpoints/blockLretry-2022.05.24-18-55-45_13.pt'
# model = './logs/experiment/r001_rot-2022.05.12-14-58-40/checkpoint8.pth'
# model = './logs/experiment/r001_line-2022.05.12-14-57-50/checkpoint8.pth'
# model = './logs/experiment/r001_line-2022.05.12-14-57-50/checkpoint8.pth'
# model = './logs/experiment/r001_line-2022.05.12-14-57-50/checkpoint8.pth'
# model = None
# model = './checkpoints/blockVHACD_convex_16processes-2022.06.05-15-15-33_9.pt'
# model = './checkpoints/256MeshVHACD_convex_16processes-2022.06.06-16-47-01_5.pt'
# model = './checkpoints/256MeshVHACD_convex_16processes-2022.06.06-16-47-01_4.pt'
# model = './checkpoints/BlockLContourRetry-2022.06.09-21-56-38_3.pt'
# model = './checkpoints/BlockLDefects004-2022.06.10-15-24-09_6.pt'
# model = './checkpoints/256_sample_points_4096-2022.06.28-10-32-06.pt'
# model = './checkpoints/SurfacePointsRandom_complete_parts_agian-2022.07.05-15-25-05.pt'
# model = './checkpoints/SurfacePointsRandom-2022.06.29-12-32-49.pt'
# model = './checkpoints/block_uni_flb_defects-2022.06.23-19-27-02.pt'
# model = './checkpoints/IR_apc_ycb-2022.07.19-13-03-57_1.pt'

# model = './checkpoints/tetris3D_tolerance_middle_vhacd-2022.07.16-13-07-40_18.pt'
# model = './checkpoints/tetris3D_tolerance_large_vhacd-2022.07.16-13-08-38_23.pt'
# model = './checkpoints/IR_apc_ycb-2022.07.19-13-03-57_5.pt'
# model = './checkpoints/IR_mix_pointCloud-2022.07.21-10-21-53_1.pt'
# model = './checkpoints/IR_mix_half_pointCloud-2022.07.23-13-38-40_2.pt'
# model = './checkpoints/new_tetris3D_mass_middle_30-2022.07.27-19-02-35_56.pt'
# model = './checkpoints/IR_concaveArea2_mass_category-2022.07.28-14-47-06_45.pt'
# model = './checkpoints/BlockLInUse_mass_vhacd-2022.07.31-20-44-17_41.pt'
# model = './checkpoints/BlockLInUse_mass_vhacd_another-2022.08.01-17-47-45_2.pt'
# model = './checkpoints/BlockLInUse_mass_vhacd_another-2022.08.01-17-47-45_31.pt'
# model = './checkpoints/IR_mix_mass_pcd_half-2022.07.27-18-38-50_100.pt'
# model = './checkpoints/IR_mix_mass_pcd_half-2022.07.27-18-38-50_101.pt'
# model = './checkpoints/IR_mix_mass_pcd_half-2022.07.27-18-38-50_102.pt'
# model = './checkpoints/IR_mix_mass_pcd_half-2022.07.27-18-38-50_103.pt'
# model = './checkpoints/IR_mix_mass_pcd_half-2022.07.27-18-38-50_120.pt'
# model = './checkpoints/IR_concaveArea3_mass_category-2022.07.30-19-56-27_53.pt'
# model = './checkpoints/new_tetris3D_mass_middle_30_hier_100_lowest-2022.08.01-23-09-11_37.pt'
# model = './checkpoints/IR_concaveArea3_mass_category-2022.07.30-19-56-27_75.pt'
# model = './checkpoints/new_tetris3D_mass_middle_grid_action-2022.08.01-23-10-30_36.pt'
# model = './checkpoints/IR_mix_mass_pcd_100_lowest-2022.08.02-22-35-43_34.pt'
# model = './checkpoints/IR_mix_mass_pcd_grid-2022.08.02-22-35-00_29.pt'
# model = './checkpoints/IR_concaveArea3_mass_category-2022.07.30-19-56-27_94.pt'
# model = './checkpoints/IR_mix_mass_pcd_grid-2022.08.02-22-35-00_29.pt'
# model = './checkpoints/exp7-2022.08.08-10-21-01_52_30.pt'
# model = './checkpoints/exp11_43_49.pt'
# model = './checkpoints/exp36Continue_57_13.pt'
# model = './checkpoints/exp12_29_38.pt'
# model = './checkpoints/IR_mix_mass_resolutionH_0005_20_35.pt'
# model = './checkpoints/IR_concaveArea3_mass_resolutionH_0005_21_08.pt'

model = None

# model = './checkpoints/IR_mix_mass_500_43_13.pt'
# model = './checkpoints/tetris3D_tolerance_middle_mass_500_38_48.pt'
# model = './checkpoints/IR_concaveArea3_mass_500_30_34.pt'
# model = './checkpoints/IR_abc_good_500_43_57.pt'

# model = './checkpoints/tetris3D_tolerance_middle_nomin_00_04.pt'
# model = './checkpoints/IR_mix_mass_500_gen_42_47.pt'
# model = './checkpoints/tetris3D_tolerance_middle_mass_500_gen_53_57.pt'
# model = './checkpoints/IR_concaveArea3_mass_500_gen_20_08.pt'

# model = './checkpoints/IR_mix_mass_500_packit_22_57.pt'
# model = './checkpoints/tetris3D_tolerance_middle_mass_500_packit_30_37.pt'
# model = './checkpoints/IR_concaveArea3_mass_500_packit_20_30.pt'

# model = './checkpoints/IR_mix_mass_500_2048_points_45_00.pt' # no
# model = './checkpoints/IR_mix_mass_1000_action_11_22.pt'
# model = './checkpoints/IR_mix_mass_finer_heightmap_42_57.pt' # heightmap
# model = './checkpoints/IR_mix_mass_500_double_rot_07_48.pt' # no
# model = './checkpoints/IR_mix_mass_finer_action_001_46_25.pt'
# model = './checkpoints/IR_mix_mass_500_deltaz_005_48_43.pt'

# model = './checkpoints/IR_mix_mass_naive_500_50_21.pt'
# model = './checkpoints/IR_mix_mass_500_no_distributed_learner_44_56.pt' # no
# model = './checkpoints/IR_mix_mass_500_no_batch_sim_39_34.pt' # no
# model = './checkpoints/IR_mix_nomin_500_26_54.pt' # no

# model = './checkpoints/IR_mix_mass_var_43_26.pt'
# model = './checkpoints/IR_concaveArea3_mass_var_07_01.pt'
# model = './checkpoints/IR_abc_good_var_41_36.pt'

# model = './checkpoints/IR_mix_mass_convex_var_packit_18_23.pt'
# model = './checkpoints/tetris3D_tolerance_middle_mass_convex_var_packit_37_53.pt'
# model = './checkpoints/IR_concaveArea3_mass_convex_var_packit_30_50.pt'

# 这些是比较好的model
# model = './checkpoints/IR_mix_mass_500_43_13.pt'
# model = './checkpoints/tetris3D_tolerance_middle_mass_500_38_48.pt'
# model = './checkpoints/IR_concaveArea3_mass_500_30_34.pt'
# model = './checkpoints/IR_abc_good_500_43_57.pt'

# model = './checkpoints/IR_mix_mass_convex_var_28_02.pt'
# model = './checkpoints/tetris3D_tolerance_middle_mass_convex_var_32_52.pt'
# model = './checkpoints/IR_concaveArea3_mass_convex_var_00_12.pt'
# model = './checkpoints/IR_abc_good_convex_var_20_12.pt'

# model = './checkpoints/IR_mix_mass_convex_var_gen_04_01.pt'
# model = './checkpoints/tetris3D_tolerance_middle_mass_convex_var_gen_10_14.pt'
# model = './checkpoints/IR_concaveArea3_mass_convex_var_gen_07_44.pt'


# model = './checkpoints/IR_mix_mass_convex_var_rainbow_47_19.pt'
# model = './checkpoints/IR_mix_mass_convex_var_resolutionA_04_11.pt'
# model = './checkpoints/IR_mix_mass_convex_var_resolutionH_43_34.pt'
# model = './checkpoints/IR_mix_mass_convex_var_double_rot_33_23.pt'
# model = './checkpoints/IR_mix_mass_convex_var_deltaz_005_07_53.pt'
# model = './checkpoints/IR_mix_mass_convex_var_2048_points_22_26.pt'
# model = './checkpoints/IR_mix_mass_convex_var_1000_07_51.pt'

# model = './checkpoints/IR_mix_mass_part_01_39_37.pt'
# model = './checkpoints/IR_mix_mass_part_03_36_28.pt'
# model = './checkpoints/IR_mix_mass_part_05_36_59.pt'
# model = './checkpoints/IR_mix_mass_part_08_07_03.pt'

# model = './checkpoints/IR_mix_mass_part_001_15_08.pt'
# model = './checkpoints/IR_mix_mass_part_005_32_07.pt'

# model = './checkpoints/IR_mix_mass_part_01_14_47.pt'

# model = './checkpoints/IR_mix_mass_part_01_1day.pt'
# model = './checkpoints/IR_mix_mass_part_01_3day.pt'


video = False
# evaluation_episodes = 1000
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
    # data_name = 'tetris3D_tolerance_middle_mass'
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
    # convexAction = None # HullVertices, Contour, Defects, None
    originShape = False

    hierachical = True # False 的话，就是 pi_s + dblf, 如果要求下层做决策的话，那么 hiera 要设置成true
    select_item_with_one_dqn = True
    previewNum = 10
    samplePointsNum = 1024
    heightResolution = 0.01
    # heightResolution = 0.005
    if hierachical:
        # orderModelPath = './checkpoints/new_tetris3D_mass_middle_30_hier_10-2022.07.27-19-03-18_order_40.pt'
        # orderModelPath = './checkpoints/new_tetris3D_mass_middle_30_hier_5-2022.07.30-13-17-51_order_50.pt'
        # orderModelPath = './checkpoints/IR_mix_mass_pcd_half_hier_10-2022.07.27-18-39-28_order_60.pt'
        # orderModelPath = './checkpoints/IR_concaveArea3_mass_hier_10_continue-2022.08.06-21-09-25_orderCheckpoint2022.08.07-00-25-47.pt'

        # locModelPath = './checkpoints/new_tetris3D_mass_middle_30_hier_10-2022.07.27-19-03-18_loc_40.pt'
        # locModelPath = './checkpoints/new_tetris3D_mass_middle_30_hier_5-2022.07.30-13-17-51_loc_50.pt'
        # locModelPath   = './checkpoints/IR_concaveArea3_mass_hier_10_continue-2022.08.06-21-09-25_locCheckpoint2022.08.07-00-25-47.pt'

        # orderModelPath = './checkpoints/IR_mix_mass_500action_hier10_order_37_55.pt'
        # orderModelPath = './checkpoints/tetris3D_tolerance_middle_mass_500action_hier10_order_02_58.pt'
        # orderModelPath = './checkpoints/IR_concaveArea3_mass_500action_hier10_order_38_20.pt'
        # orderModelPath = './checkpoints/IR_abc_good_500action_hier10_order.pt'

        # locModelPath   = './checkpoints/IR_mix_mass_500action_hier10_loc_37_55.pt'
        # locModelPath   = './checkpoints/tetris3D_tolerance_middle_mass_500action_hier10_loc_02_58.pt'
        # locModelPath   = './checkpoints/IR_concaveArea3_mass_500action_hier10_loc_38_20.pt'
        # locModelPath   = './checkpoints/IR_abc_good_500action_hier10_loc.pt'
        # locModelPath   = './checkpoints/IR_mix_mass_500_lfss_00_09.pt'
        # locModelPath   = './checkpoints/tetris3D_tolerance_middle_mass_500_lfss_51_52.pt'
        # locModelPath   = './checkpoints/IR_concaveArea3_mass_lfss_16_54.pt'

        orderModelPath = None

        # locModelPath = './checkpoints/IR_mix_mass_convex_var_28_02.pt'
        # locModelPath = './checkpoints/tetris3D_tolerance_middle_mass_convex_var_32_52.pt'
        # locModelPath = './checkpoints/IR_concaveArea3_mass_convex_var_00_12.pt'
        locModelPath = './checkpoints/IR_abc_good_convex_var_20_12.pt'

        # locModelPath   = './checkpoints/IR_mix_mass_convex_var_lfss_40_45.pt'
        # locModelPath   = './checkpoints/tetris3D_tolerance_middle_mass_var_lfss_46_33.pt'
        # locModelPath   = './checkpoints/IR_concaveArea3_mass_convex_var_lfss_39_51.pt'

        # orderModelPath = './checkpoints/IR_mix_mass_convex_var_hier_10_50_53_order.pt'
        # locModelPath   = './checkpoints/IR_mix_mass_convex_var_hier_10_50_53_loc.pt'

        # orderModelPath = './checkpoints/IR_mix_mass_convex_var_hier_3_train_order.pt'
        # locModelPath   = './checkpoints/IR_mix_mass_convex_var_hier_3_train_loc.pt'

        # orderModelPath = './checkpoints/tetris3D_tolerance_middle_mass_convex_var_hier_10_52_21_order.pt'
        # locModelPath   = './checkpoints/tetris3D_tolerance_middle_mass_convex_var_hier_10_52_21_loc.pt'

        # orderModelPath = './checkpoints/IR_concaveArea3_mass_convex_var_hier_10_08_02_order.pt'
        # locModelPath   = './checkpoints/IR_concaveArea3_mass_convex_var_hier_10_08_02_loc.pt'

        # orderModelPath = './checkpoints/IR_abc_good_convex_var_hier_10_51_53_order.pt'
        # locModelPath   = './checkpoints/IR_abc_good_convex_var_hier_10_51_53_loc.pt'

packed_holder = 100
enable_rotation = True
if video: visual = True