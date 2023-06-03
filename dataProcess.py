def observation_decode_irregular(observation, args):
    batchSize = observation.shape[0]
    observation = observation.reshape((batchSize, -1))
    actions = observation[:, 0 : args.selectedAction * 5].reshape(batchSize, -1, 5)
    next_item = observation[:, args.selectedAction * 5 : args.selectedAction * 5 + 1].reshape((batchSize, -1))
    actionMasks = actions[:,:, -1]
    actions = actions[:,:, 0:-1]
    heightMap = observation[:, args.selectedAction * 5 + 9:]
    return next_item, actionMasks, heightMap, actions

def observation_decode_irregular_indices(observation, args):
    batchSize = observation.shape[0]
    observation = observation.reshape((batchSize, -1))
    actions = observation[:, 0 : args.selectedAction * 5].reshape(batchSize, -1, 5)
    next_item = observation[:, args.selectedAction * 5 : args.selectedAction * 5 + 1].reshape((batchSize, -1))
    actionMasks = actions[:,:, -1]
    actions = actions[:,:, 0:-1]
    heightMap = observation[:, args.selectedAction * 5 + 9: -args.samplePointsNum]
    indices = observation[:, -args.samplePointsNum:]
    return next_item, actionMasks, heightMap, actions, indices

def observation_decode_irregular_k_shape(observation, args):
    batchSize = observation.shape[0]
    observation = observation.reshape((batchSize, -1))
    shapes = observation[:, 0 : args.previewNum].reshape(batchSize, args.previewNum)
    heightMap = observation[:, args.previewNum:]
    return shapes, heightMap

def observation_decode_irregular_with_k_shape(observation, args):
    # print('observation shape', observation.shape)
    batchSize = observation.shape[0]
    observation = observation.reshape((batchSize, -1))
    actions = observation[:, 0 : args.selectedAction * 5].reshape(batchSize, -1, 5)
    next_item = observation[:, args.selectedAction * 5 : args.selectedAction * 5 + 1].reshape((batchSize, -1))
    actionMasks = actions[:,:, -1]
    actions = actions[:,:, 0:-1]
    heightMap = observation[:, args.selectedAction * 5 + 9: - 10]
    # print('heightMap',heightMap.shape)
    kShape = observation[:, - 10 : ]
    # print('kShape',kShape.shape)
    return next_item, actionMasks, heightMap, actions, kShape