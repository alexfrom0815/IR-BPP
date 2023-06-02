import numpy as np
import cv2
import torch
import time

# approxPolyDP
def find_out_contour(contour, hierarchy):
    # Next, Previous, First_Child, Parent
    validIndexs = []
    inValidIndexs = []
    levelCounter = 0
    thisLevel = np.where(hierarchy[:, -1] == -1)[0]
    validIndexs.extend(thisLevel)
    while len(validIndexs) + len(inValidIndexs) != len(hierarchy):
        next_level = []
        for i in thisLevel:
            child = hierarchy[i, 2]
            if child != -1:
                next_level.append(child)
                pointer = child
                while hierarchy[pointer][0] != -1:
                    next_level.append(hierarchy[pointer][0])
                    pointer = hierarchy[pointer][0]

                pointer = child
                while hierarchy[pointer][1] != -1:
                    next_level.append(hierarchy[pointer][1])
                    pointer = hierarchy[pointer][1]

        if levelCounter % 2 != 0:
            validIndexs.extend(next_level)
        else:
            inValidIndexs.extend(next_level)
        levelCounter += 1
        thisLevel = next_level

    newContour = [contour[i] for i in validIndexs]
    return newContour, validIndexs

def find_convex_vetex(approx):
    length = len(approx)
    if length <= 3:
        return np.arange(length)
    else:
        vertex = np.array((approx))[:, 0, :] # B

        last_vertex = np.ones_like(vertex) # A
        last_vertex[0] = vertex[length-1]
        last_vertex[1:length] = vertex[0:length-1]

        next_vertex = np.ones_like(vertex) # C
        next_vertex[0:length-1] = vertex[1:length]
        next_vertex[length - 1] = vertex[0]

        AB = vertex - last_vertex
        AC = next_vertex - last_vertex
        cross = np.cross(AB, AC)

        return np.where(cross < 0)[0]

# only consider one rot now, I CAN JUDGE THE AREA OF CONVEX HULL BOX, TO JUDGE IF NEEDED
def getConvexHullActions(posZValid, mask, actionType, heightResolution, draw = False):
    allCandidates = []
    save = False
    for rotIdx in range(len(posZValid)):
        allHulls, V, s = convexHulls(posZValid[rotIdx], mask[rotIdx], actionType, heightResolution, draw)
        # if rotIdx == 2:
        #     torch.save([posZValid, mask], 'mask_{}.pt'.format(format(time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time())))))
        if s:
            save = True
        if len(allHulls) != 0:
            H = posZValid[rotIdx][allHulls[:,1], allHulls[:,0]]
            ROT = np.ones((len(allHulls))) * rotIdx
            candidates = np.concatenate([ROT.reshape(-1, 1), allHulls[:,1].reshape(-1, 1), allHulls[:,0].reshape(-1, 1),
                                         H.reshape(-1, 1), V.reshape(-1, 1)], axis=1)
            allCandidates.append(candidates)
    if len(allCandidates)!= 0:
        allCandidates = np.concatenate(allCandidates, axis=0)
        return allCandidates, save
    else:
        return None, save

# Operate on single rotation.
def convexHulls(posZMap, mask, actionType,  heightResolution = 0.01, draw = False):
    mapInt = (posZMap // heightResolution).astype(np.int32)
    mapInt[mask==0] = -1
    uniqueHeight = np.unique(mapInt)
    allCandidates = []
    save = False
    if draw[0]:
        mapO = posZMap // (1 / 255)
        mapO = mapO.astype(np.uint8)
        mapO = cv2.applyColorMap(mapO, cv2.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列

        mapDraw = posZMap * mask
        mapDraw = mapDraw // (1 / 255)
        mapDraw[mask == 0] = -1
        mapDraw = mapDraw.astype(np.uint8)
        mapDraw = cv2.applyColorMap(mapDraw, cv2.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列

    for h in uniqueHeight:
        if h == -1: continue
        check = np.where(mapInt == h, 255, 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(image=check, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        newContour, outIdx = find_out_contour(contours,hierarchy[0])

        for i in range(len(newContour)):
            defects = None
            if actionType == 'HullVertices' or actionType == 'Defects':
                hx, hy, hw, hh = cv2.boundingRect(newContour[i])
                if hw * hh <= 6 and hw !=1 and hh !=1:
                    candidate = sorted(newContour[i].reshape((-1, 2)), key=lambda ems: (ems[1], ems[0]), reverse=False)[0]
                    candidate = candidate.reshape((-1,2))
                else:
                    candidateIdx = cv2.convexHull(newContour[i], returnPoints=False)
                    candidate = newContour[i][candidateIdx].reshape((-1, 2))
                    if actionType == 'Defects':
                        candidateIdx[::-1].sort(axis=0)
                        defects = cv2.convexityDefects(newContour[i], candidateIdx)
                        if defects is not None:
                            defectsIdx = defects[:, 0, 2]
                            defects = newContour[i][defectsIdx]
            elif actionType == 'ConvexVertex':
                approx = cv2.approxPolyDP(newContour[i], 1, True)
                convexIndex = find_convex_vetex(approx)
                candidate = approx[convexIndex].reshape((-1, 2))
            else:
                assert actionType == 'Contour'
                candidate = newContour[i].reshape((-1, 2))

            allCandidates.append(candidate)
            if defects is not None:
                allCandidates.append(defects.reshape(-1,2))

            # if len(candidate) > 6 and draw[1] < 5 and (np.sum(check / 255) / np.prod(check.shape)) > 0.2:
            #     save = True

            if draw[0]:
                origin = cv2.cvtColor(check, cv2.COLOR_GRAY2BGR)
                # Draw contour
                canvasC = cv2.cvtColor(check, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(image=canvasC, contours=newContour, contourIdx=-1, color=(0, 255, 0), thickness=1)

                # Draw hull
                canvasH = cv2.cvtColor(check, cv2.COLOR_GRAY2BGR)
                cv2.polylines(canvasH, [candidate], True, (255, 255, 0), thickness=1)
                for singH in candidate:
                    cv2.circle(mapDraw, tuple(singH.reshape(2).tolist()), radius=1, color=(0,0,255))

                # invalidM = cv2.cvtColor(check, cv2.COLOR_GRAY2BGR)
                # invalidIdx = np.where(mask[(candidate[:, 0], candidate[:, 1])] == 0)[0]
                # for idx in invalidIdx:
                #     cv2.circle(invalidM, tuple(candidate[idx].reshape(2).tolist()), radius=1, color=(0,0,255))
                canThisLayer = cv2.cvtColor(check, cv2.COLOR_GRAY2BGR)
                for singH in candidate:
                    cv2.circle(canThisLayer, tuple(singH.reshape(2).tolist()), radius=1, color=(0, 0, 255))

                if defects is not None:
                    drawD = defects.reshape(-1, 2)
                    for singD in drawD:
                        cv2.circle(canThisLayer, tuple(singD.reshape(2).tolist()), radius=1, color=(0, 255, 0))


                if len(candidate) > 6 and draw[1] < 5 and (np.sum(check / 255) / np.prod(check.shape)) > 0.2 and h == 0:
                    cv2.imshow('all', np.concatenate((mapO, # 这里是 heightmap
                                                      mapDraw, # 包含所有的动作候选， 对应一个旋转下
                                                      origin,  # 感兴趣的区域
                                                      canvasC, # 轮廓
                                                      canvasH, # 凸包
                                                      canThisLayer), axis=1))
                    # torch.save([posZMap, mask], 'debug.pt')
                    cv2.waitKey(0)
    V = None
    if len(allCandidates) != 0:
        allCandidates = np.concatenate(allCandidates, axis=0)
        allCandidates = np.unique(allCandidates,axis=0) # this is img coords, but not suitable for numpy coords
        V = mask[(allCandidates[:, 1], allCandidates[:, 0])]
    return allCandidates, V, save