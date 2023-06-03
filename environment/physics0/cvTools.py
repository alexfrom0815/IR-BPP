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

def getConvexHullActions(posZValid, mask,  heightResolution):
    allCandidates = []
    for rotIdx in range(len(posZValid)):
        allHulls, V= convexHulls(posZValid[rotIdx], mask[rotIdx],  heightResolution)
        if len(allHulls) != 0:
            H = posZValid[rotIdx][allHulls[:,1], allHulls[:,0]]
            ROT = np.ones((len(allHulls))) * rotIdx
            candidates = np.concatenate([ROT.reshape(-1, 1), allHulls[:,1].reshape(-1, 1), allHulls[:,0].reshape(-1, 1),
                                         H.reshape(-1, 1), V.reshape(-1, 1)], axis=1)
            allCandidates.append(candidates)
    if len(allCandidates)!= 0:
        allCandidates = np.concatenate(allCandidates, axis=0)
        return allCandidates
    else:
        return None

def convexHulls(posZMap, mask,   heightResolution = 0.01):
    mapInt = (posZMap // heightResolution).astype(np.int32)
    mapInt[mask==0] = -1
    uniqueHeight = np.unique(mapInt)
    allCandidates = []

    for h in uniqueHeight:
        if h == -1: continue
        check = np.where(mapInt == h, 255, 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(image=check, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        newContour, outIdx = find_out_contour(contours,hierarchy[0])

        for i in range(len(newContour)):
            defects = None
            approx = cv2.approxPolyDP(newContour[i], 1, True)
            convexIndex = find_convex_vetex(approx)
            candidate = approx[convexIndex].reshape((-1, 2))
            allCandidates.append(candidate)
            if defects is not None:
                allCandidates.append(defects.reshape(-1,2))

    V = None
    if len(allCandidates) != 0:
        allCandidates = np.concatenate(allCandidates, axis=0)
        allCandidates = np.unique(allCandidates,axis=0)
        V = mask[(allCandidates[:, 1], allCandidates[:, 0])]
    return allCandidates, V