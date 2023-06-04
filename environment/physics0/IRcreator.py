import numpy as np
import copy
import torch

# 如果想把dict改掉的话，主要是修改这里的逻辑
class ItemCreator(object): # 存一个查shape的字典， 再存一个记录编号的list
    def __init__(self):
        self.item_dict = {} # 根据编号查shape的字典
        self.item_list = [] # 已经存放的item的编号

    def reset(self, index = None):
        self.item_list.clear()

    def generate_item(self, **kwargs):
        pass

    def preview(self, length):
        while len(self.item_list) < length:
            self.generate_item()
        return copy.deepcopy(self.item_list[:length])

    def update_item_queue(self, index):
        assert len(self.item_list) >= 0
        self.item_list.pop(index)

class RandomItemCreator(ItemCreator):
    def __init__(self, item_set):
        super().__init__()
        self.item_set = item_set
        print(self.item_set)

    def generate_item(self):
        self.item_list.append(np.random.choice(self.item_set))

class RandomInstanceCreator(ItemCreator):
    def __init__(self, item_set, dicPath):
        super().__init__()
        self.inverseDict = {}
        for k in dicPath.keys():
            if dicPath[k][0:-6] not in self.inverseDict.keys():
                self.inverseDict[dicPath[k][0:-6]] = [k]
            else:
                self.inverseDict[dicPath[k][0:-6]].append(k)

        self.item_set = item_set
        print(self.item_set)
        print(self.inverseDict)

    def generate_item(self):
        name = np.random.choice(list(self.inverseDict.keys()))
        self.item_list.append(np.random.choice(self.inverseDict[name]))

class RandomCateCreator(ItemCreator):
    def __init__(self, item_set, dicPath):
        super().__init__()
        self.categories = {'objects': 0.34, 'concave': 0.33, 'board': 0.33}

        self.objCates = {}
        for key in self.categories.keys():
            self.objCates[key] = []

        for k, item in zip(dicPath.keys(), dicPath.values()):
            cate, item = item.split('/')
            self.objCates[cate].append(k)

        self.item_set = item_set
        print(self.item_set)
        print(self.objCates)

    def generate_item(self):
        name = np.random.choice(list(self.categories.keys()))
        self.item_list.append(np.random.choice(self.objCates[name]))

class LoadItemCreator(ItemCreator):
    def __init__(self, data_name=None):
        super().__init__()
        self.data_name = data_name
        self.traj_index = 0
        self.item_index = 0
        print("Load dataset set: {}".format(data_name))
        self.item_trajs = torch.load(self.data_name)
        self.traj_nums = len(self.item_trajs)

    def reset(self, traj_index=None):
        self.item_list.clear()

        if traj_index is None:
            self.traj_index += 1
        else:
            self.traj_index = traj_index

        self.traj = self.item_trajs[self.traj_index]
        self.item_index = 0
        self.item_set = self.traj
        self.item_set.append(None)

    def generate_item(self, **kwargs):
        if self.item_index < len(self.item_set):
            self.item_list.append(self.item_set[self.item_index])
            self.item_index += 1
        else:
            self.item_list.append(None)
            self.item_index += 1
