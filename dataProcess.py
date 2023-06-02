import torch

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

def info_nce_loss(features):
    batchSize = int(len(features)/2)
    temperature = 0.07
    n_views = 2

    labels = torch.cat([torch.arange(batchSize) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    logits = logits / temperature
    return logits, labels