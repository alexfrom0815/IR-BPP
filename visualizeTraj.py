import numpy as np
import pybullet as p
from environment.physics0.Interface import Interface
import cv2
import os

def extendMat(mat3, translation = None):
    mat4 = np.eye(4)
    mat4[0:3,0:3] = mat3
    if translation is not None:
        mat4[0:3,3] = translation
    return mat4

bin_dimension = [0.32, 0.32, 0.30]
# bin_dimension = [0.32, 0.32, 0.035]
# objPath = './data/datas/BlockL_vhacd'
objPath = './data/datas/256_vhacd'
# folder = 'debug-2022.06.14-15-26-09'
folder = 'hullaction256RL-2022.06.14-16-15-54'
folderList = ['no_mask_stop-2022.06.14-19-11-54',
              'no_mask_stop_dblf-2022.06.14-19-26-30',
              'no_mask_stop_first-2022.06.14-19-27-36',
              'no_mask_stop_hm-2022.06.14-19-26-49',
              'no_mask_stop_minz-2022.06.14-19-27-18',
              'no_mask_stop_random-2022.06.14-19-26-09']
interface = Interface(bin=bin_dimension, foldername=objPath,
                      visual=True, scale=[1, 1, 1])

for folder in folderList:
    dataPath = '/home/hang/Documents/GitHub/IRBPP/logs/evaluation/{}/trajs.npy'.format(folder)

    trajs = np.load(dataPath, allow_pickle=True)


    figPath = os.path.join('./trajPictures', folder)
    if not os.path.exists(figPath): os.makedirs(figPath)

    for trajIdx in range(len(trajs)):
        subFigPath = os.path.join(figPath, str(trajIdx))
        print(subFigPath)
        if not os.path.exists(subFigPath):
            os.makedirs(subFigPath)

        traj = trajs[trajIdx]
        meshInTraj = []
        for itemIdx in range(len(traj)):
            item = traj[itemIdx]
            positionT, orientationT = item[2:]
            name = item[1].replace('.obj','')
            color = [1,0,0,1] if itemIdx == len(traj) - 1 else None
            id = interface.addObject(name, translation = positionT, rotation=orientationT, FLB = False, color=color)
            # p.setDebugObjectColor(id, -1, objectDebugColorRGB = [1,0,0])
            _, _, img, _, _ = p.getCameraImage(512,512)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img[:,:,0:3]
            cv2.imwrite(subFigPath + '/{}.png'.format(itemIdx), img)
        interface.reset()

