import numpy as np
import pickle
import json
import random
import math
import IPython

from torch.utils.data import Dataset

class Feeder(Dataset):
    def __init__(self, data_path, label_path, repeat=1, random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        data = np.load('../Oneshot_Action_Recognition/data/NW-UCLA/oneshot_ucla.npz', allow_pickle=True)
        train_data = data['train_data']
        test_data = data['test_data']
        # test_data = data['exem_data']


        if 'val' in label_path:
            self.train_val = 'val'
            self.data_dict = test_data
        else:
            self.train_val = 'train'
            self.data_dict = train_data

        self.nw_ucla_root = '../Oneshot_Action_Recognition/data/NW-UCLA/all_sqe/'
        self.time_steps = 52
        self.bone = [(1, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 5), (7, 6), (8, 7), (9, 3), (10, 9), (11, 10),
                     (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19)]
        self.label = []
        for index in range(len(self.data_dict)):
            info = self.data_dict[index]
            self.label.append(int(info['label']) - 1)

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.repeat = repeat
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        self.data = []
        for data in self.data_dict:
            file_name = data['file_name']
            with open(self.nw_ucla_root + file_name + '.json', 'r') as f:
                json_file = json.load(f)
            skeletons = json_file['skeletons']
            value = np.array(skeletons)
            self.data.append(value)


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.data_dict)*self.repeat

    def __iter__(self):
        return self

    def rand_view_transform(self,X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0, -math.sin(agx),math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0,1,0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
        X0 = np.dot(np.reshape(X,(-1,3)), np.dot(Ry,np.dot(Rx,Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def __getitem__(self, index):
        label = self.label[index % len(self.data_dict)]
        value = self.data[index % len(self.data_dict)]

        if self.train_val == 'train':
            random.random()
            agx = random.randint(-60, 60)
            agy = random.randint(-60, 60)
            s = random.uniform(0.5, 1.5)

            center = value[0,1,:]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue,axis=0)) / (np.max(scalerValue,axis=0) - np.min(scalerValue,axis=0))
            scalerValue = scalerValue*2-1
            scalerValue = np.reshape(scalerValue, (-1, 20, 3))

            data = np.zeros( (self.time_steps, 20, 3) )

            value = scalerValue[:,:,:]
            length = value.shape[0]

            random_idx = random.sample(list(np.arange(length))*100, self.time_steps)
            random_idx.sort()
            data[:,:,:] = value[random_idx,:,:]
            data[:,:,:] = value[random_idx,:,:]

        else:
            random.random()
            agx = 0
            agy = 0
            s = 1.0

            center = value[0,1,:]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue,axis=0)) / (np.max(scalerValue,axis=0) - np.min(scalerValue,axis=0))
            scalerValue = scalerValue*2-1

            scalerValue = np.reshape(scalerValue, (-1, 20, 3))

            data = np.zeros( (self.time_steps, 20, 3) )

            value = scalerValue[:,:,:]
            length = value.shape[0]

            idx = np.linspace(0,length-1,self.time_steps).astype(np.int)
            data[:,:,:] = value[idx,:,:] # T,V,C

        if 'bone' in self.data_path:
            data_bone = np.zeros_like(data)
            for bone_idx in range(20):
                data_bone[:, self.bone[bone_idx][0] - 1, :] = data[:, self.bone[bone_idx][0] - 1, :] - data[:, self.bone[bone_idx][1] - 1, :]
            data = data_bone

        if 'motion' in self.data_path:
            data_motion = np.zeros_like(data)
            data_motion[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]
            data = data_motion
        data = np.transpose(data, (2, 0, 1))
        C,T,V = data.shape
        data = np.reshape(data,(C,T,V,1))

        return data, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()

        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod