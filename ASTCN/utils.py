import h5py
import numpy as np
import os
import time
import datetime
import math
from copy import copy
import pickle
import sys
import torch
import torch.utils.data as data


class OneHot():
    """transform the picture to one hot"""
    def __init__(self, hot_number):
        self.hot_number = hot_number

    def __call__(self, sample):
        all_sample = []
        input_flat = sample.reshape(-1)
        one_hot_flat = torch.zeros(input_flat.shape[0], self.hot_number)
        one_hot_flat[torch.arange(input_flat.shape[0]).tolist(), input_flat.tolist()] = 1
        # reverted = torch.argmax(one_hot_flat, dim=1)
        # assert (input_flat == reverted).all().item()
        output = one_hot_flat.reshape([*sample.shape] + [self.hot_number])
        # output = output.byte()
        # all_sample.append(output.view(1, *output.shape))
        return output


class Binary():
    """transform the picture to binary"""
    def __init__(self):
        pass

    def __call__(self, sample):
        # return torch.where(sample==0, torch.tensor(0), torch.tensor(255))
        sample = sample>0
        return sample.float()

def binary_tensor(input):
    # return torch.where(torch.tensor(input) == 0, torch.tensor(0), torch.tensor(255))
    return torch.tensor(input) > 0




def stat(fname):
    def get_nb_timeslot(f):
        s = f['date'][0]
        e = f['date'][-1]
        year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
        ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
        te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
        nb_timeslot = (time.mktime(te) - time.mktime(ts)) / (0.5 * 3600) + 48
        ts_str, te_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
        return nb_timeslot, ts_str, te_str

    with h5py.File(fname) as f:
        nb_timeslot, ts_str, te_str = get_nb_timeslot(f)
        nb_day = int(nb_timeslot / 48)
        mmax = f['data'][()].max()
        mmin = f['data'][()].min()
        stat = '=' * 5 + 'stat' + '=' * 5 + '\n' + \
               'data shape: %s\n' % str(f['data'].shape) + \
               '# of days: %i, from %s to %s\n' % (nb_day, ts_str, te_str) + \
               '# of timeslots: %i\n' % int(nb_timeslot) + \
               '# of timeslots (available): %i\n' % f['date'].shape[0] + \
               'missing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
               'max: %.3f, min: %.3f\n' % (mmax, mmin) + \
               '=' * 5 + 'stat' + '=' * 5
        print(stat)


def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'][:]
    timestamps = f['date'][:]
    f.close()
    return data, timestamps

def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
#     vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]  # python2
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)

def remove_incomplete_days(data, timestamps, T=48):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i+T-1 < len(timestamps) and int(timestamps[i+T-1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps

def load_holiday(timeslots, fname):
    f = open(fname, 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    print(H.sum())
    # print(timeslots[H==1])
    return H[:, None]


def load_meteorol(timeslots, fname):
    '''
    timeslots: the predicted timeslots
    In real-world, we dont have the meteorol data in the predicted timeslot, instead, we use the meteoral at previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
    '''
    f = h5py.File(fname, 'r')
    Timeslot = f['date'][:]
    WindSpeed = f['WindSpeed'][:]
    Weather = f['Weather'][:]
    Temperature = f['Temperature'][:]
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # WindSpeed
    WR = []  # Weather
    TE = []  # Temperature
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id - 1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    print("shape: ", WS.shape, WR.shape, TE.shape)

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

    # print('meger shape:', merge_data.shape)
    return merge_data

class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X



def split_train_val_test_len(len_alldata, val_percent=0.1, test_percent=0.1):
    len_train = math.ceil(len_alldata * (1-val_percent-test_percent))
    len_val = int(len_alldata * val_percent)
    len_test = int(len_alldata * test_percent)
    return len_train, len_val, len_test


def mix_external_data(timestamps, holiday_fname='./data/processed/TaxiBJ/BJ_Holiday.txt',
                      meteoraol_fname='./data/processed/TaxiBJ/BJ_Meteorology.h5'):
    mix_data = []
    mix_data.append(timestamp2vec(timestamps))
    mix_data.append(load_holiday([str(int(x)) for x in timestamps], fname=holiday_fname))
    mix_data.append(load_meteorol(timestamps, meteoraol_fname))
    return mix_data


def save():
    '''
    :return:
    '''
    data_all = []
    timestamps_all = list()

    save_path = './data/processed/TaxiBJ/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    DATAPATH = "./data/processed"
    preprocess_name = os.path.join(save_path, 'st_preprocessing.pkl')

    # 连接13年到17年的数据
    for year in range(13, 17):
        fname = os.path.join(
            DATAPATH, 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        print("file name: ", fname)
        stat(fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days(data, timestamps, 48)
        data = data[:, :2]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")

    data_all_stack = np.vstack(copy(data_all))
    print('all data shape', data_all_stack.shape)

    len_train, len_val, len_test = split_train_val_test_len(data_all_stack.shape[0])
    print('len_train, len_val, len_test:', len_train, len_val, len_test)
    train_val_data = data_all_stack[:-(len_test)]
    print('train_val_data shape: ', train_val_data.shape)

    mmn = MinMaxNormalization()
    mmn.fit(train_val_data)
    data_all_mmn = [mmn.transform(d) for d in data_all]

    # 不知道这个有什么用
    fpkl = open(preprocess_name, 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    data_all_mmn = np.vstack(data_all_mmn)

    np.save(os.path.join(save_path, 'MMN_all_InOut'), data_all_mmn)
    npf = np.load(os.path.join(save_path, 'MMN_all_InOut.npy'))



    train_data = npf[:-len_test - len_val]
    val_data = npf[-len_test - len_val: -len_test]
    test_data = npf[-len_test:]

    np.save(os.path.join(save_path, 'train_data'), train_data)
    np.save(os.path.join(save_path, 'val_data'), val_data)
    np.save(os.path.join(save_path, 'test_data'), test_data)

    timestamps_all = np.concatenate(copy(timestamps_all))
    print('The shape of timestamps_all: ', timestamps_all.shape)

    np.save(os.path.join(save_path, 'timestamps_all'), timestamps_all)

    train_ts = timestamps_all[:-len_test - len_val]
    val_ts = timestamps_all[-len_test - len_val: -len_test]
    test_ts = timestamps_all[-len_test:]

    np.save(os.path.join(save_path, 'timestamp_train'), train_ts)
    np.save(os.path.join(save_path, 'val_ts'), val_ts)
    np.save(os.path.join(save_path, 'timestamp_test'), test_ts)

    # 生成测试数据 shape:(10, 48*100)
    test_timestamps_for_meta = np.asarray([train_ts[i: 100 * 48] for i in range(10)])
    a_test_timestamps_for_meta = test_timestamps_for_meta[0]
    external_data = mix_external_data(a_test_timestamps_for_meta)
    external_data = np.hstack(external_data)
    print('Shape of external data', external_data.shape)



class STData(data.Dataset):
    '''
    dataset_type can set as "train", "validate", "train_all", "test"

    return
    seq: [windows_size, 2, 32, 32]
    ext_data: [windows_size, 28]
    target: [2, 32, 32]
    '''
    def __init__(self, dataset_type="train", windows_size=7*48,  dir_path='./data/processed/TaxiBJ/',
                 transform=None, target_transform=None):

        self.val_percent = 1/9
        
        self.dir_path =dir_path
        self.dataset_type = dataset_type
        self.windows_size = windows_size
        self.holiday_fname = os.path.join(self.dir_path, 'BJ_Holiday.txt')
        self.meteoraol_fname = os.path.join(self.dir_path, 'BJ_Meteorology.h5')
        with open(self.holiday_fname, "r") as f:
            self.holiday_file = f.readlines()
        # self.holiday_file = open(self.holiday_fname, 'r')
        self.meteorol_file = h5py.File(self.meteoraol_fname, 'r')
        meteoroal = {}
        for key in self.meteorol_file.keys():
            meteoroal[key] = self.meteorol_file[key][()]
        self.meteorol_file = meteoroal

        self.transform = transform
        self.target_transform = target_transform

        from_path = os.path.join(self.dir_path, 'MMN_all_InOut.npy')
        self.data_all = torch.tensor(np.load(from_path).astype(np.float32))
        from_path = os.path.join(self.dir_path, 'timestamps_all.npy')
        self.data_all_ts = np.load(from_path)

        if self.dataset_type == "train":
            from_path = os.path.join(self.dir_path, 'timestamp_train.npy')
            self.train_ts = np.load(from_path)
            self.train_ts = self.train_ts[:int(-self.val_percent * self.train_ts.shape[0])]
        elif self.dataset_type == "validate":
            from_path = os.path.join(self.dir_path, 'timestamp_train.npy')
            self.validate_ts = np.load(from_path)
            self.validate_ts = self.validate_ts[int(-self.val_percent * self.validate_ts.shape[0]):]
        elif self.dataset_type == "test":
            from_path = os.path.join(self.dir_path, 'timestamp_test.npy')
            self.test_ts = np.load(from_path)
        elif self.dataset_type == "train_all":
            from_path = os.path.join(self.dir_path, 'timestamp_train.npy')
            self.train_all_ts = np.load(from_path)

            
    def mix_external_data(self, timestamps):
        mix_data = []
        mix_data.append(self.timestamp2vec(timestamps))
        mix_data.append(self.load_holiday([str(int(x)) for x in timestamps]))
        mix_data.append(self.load_meteorol(timestamps))
        return mix_data

    def timestamp2vec(self, timestamps):
        # tm_wday range [0, 6], Monday is 0
        vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
        #     vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]  # python2
        ret = []
        for i in vec:
            v = [0 for _ in range(7)]
            v[i] = 1
            if i >= 5:
                v.append(0)  # weekend
            else:
                v.append(1)  # weekday
            ret.append(v)
        return np.asarray(ret)

    def load_holiday(self, timeslots):
        holidays = self.holiday_file
        holidays = set([h.strip() for h in holidays])
        H = np.zeros(len(timeslots))
        for i, slot in enumerate(timeslots):
            if slot[:8] in holidays:
                H[i] = 1
        # print(H.sum())
        # print(timeslots[H==1])
        return H[:, None]

    def load_meteorol(self, timeslots):
        '''
        timeslots: the predicted timeslots
        In real-world, we dont have the meteorol data in the predicted timeslot, instead, we use the meteoral at previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
        '''
        f = self.meteorol_file
        Timeslot = f['date']
        WindSpeed = f['WindSpeed']
        Weather = f['Weather']
        Temperature = f['Temperature']

        M = dict()  # map timeslot to index
        for i, slot in enumerate(Timeslot):
            M[slot] = i

        WS = []  # WindSpeed
        WR = []  # Weather
        TE = []  # Temperature
        for slot in timeslots:
            predicted_id = M[slot]
            cur_id = predicted_id - 1
            WS.append(WindSpeed[cur_id])
            WR.append(Weather[cur_id])
            TE.append(Temperature[cur_id])

        WS = np.asarray(WS)
        WR = np.asarray(WR)
        TE = np.asarray(TE)

        # 0-1 scale
        if WS.size == 0:
            print()
        if WS.size!=0  and WS.max() != WS.min():
            WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
        if TE.size!=0 and TE.max() != TE.min():
            TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

        # print("shape: ", WS.shape, WR.shape, TE.shape)

        # concatenate all these attributes
        merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

        # print('meger shape:', merge_data.shape)
        return merge_data




    def __getitem__(self, index):
        if self.dataset_type == "train":
            y_index = np.argwhere(self.data_all_ts == self.train_ts[index])[0][0]
            # seq, target = self.data_all[y_index-self.windows_size: y_index], self.data_all[y_index: y_index+1]
            # timestamp_ = self.data_all_ts[y_index-self.windows_size: y_index]
        elif self.dataset_type == "validate":
            y_index = np.argwhere(self.data_all_ts == self.validate_ts[index])[0][0]
            # seq, target = self.data_all[y_index-self.windows_size: y_index], self.data_all[y_index: y_index+1]
            # timestamp_ = self.data_all_ts[y_index-self.windows_size: y_index]
        elif self.dataset_type == "test":
            y_index = np.argwhere(self.data_all_ts == self.test_ts[index])[0][0]
            # seq, target = self.data_all[y_index-self.windows_size: y_index], self.data_all[y_index: y_index+1]
            # timestamp_ = self.data_all_ts[y_index-self.windows_size: y_index]
        elif self.dataset_type == "train_all":
            y_index = np.argwhere(self.data_all_ts == self.train_all_ts[index])[0][0]
        seq, target = self.data_all[y_index: y_index+self.windows_size], self.data_all[y_index+self.windows_size: y_index+self.windows_size+1]
        # seq, target = self.data_all[y_index-self.windows_size: y_index], self.data_all[y_index: y_index+1]
        # timestamp_ = self.data_all_ts[y_index-self.windows_size: y_index]
        timestamp_ = self.data_all_ts[y_index: y_index+self.windows_size]
        # if timestamp_.size == 0:
        #     print()

        ext_data = torch.tensor(np.hstack(self.mix_external_data(timestamp_))).float()
        return seq.permute(1, 0, 2, 3), ext_data, target.permute(1, 0, 2, 3)


    def __len__(self):
        if self.dataset_type == "train":
             return len(self.train_ts)-self.windows_size
        elif self.dataset_type == "validate":
            return len(self.validate_ts)-self.windows_size
        elif self.dataset_type == "test":
            return len(self.test_ts)-self.windows_size
        elif self.dataset_type == "train_all":
            return len(self.train_all_ts)-self.windows_size

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        if self.dataset_type == "train":
            tmp = 'train'
        elif self.dataset_type == "validate":
            tmp = 'validate'
        elif self.dataset_type == "test":
            tmp = 'test'
        elif self.dataset_type == "train_all":
            tmp = "train_all"
        
        fmt_str += '    Train/Validate/Test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.dir_path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str




if __name__ == '__main__':
    save()
    windows_size = 7*48
    train_data = STData(dataset_type="train", windows_size=windows_size, dir_path='./data/processed/TaxiBJ/')
    print(train_data)
    train_data_loader = data.DataLoader(train_data, batch_size=32, num_workers=0)

    val_data = STData(dataset_type="validate", windows_size=windows_size, dir_path='./data/processed/TaxiBJ/')
    print(val_data)
    val_data_loader = data.DataLoader(val_data, batch_size=32, num_workers=0)

    test_data = STData(dataset_type="test", windows_size=windows_size, dir_path='./data/processed/TaxiBJ/')
    print(test_data)
    test_data_loader = data.DataLoader(test_data, batch_size=32, num_workers=0)

    train_all_data = STData(dataset_type="train_all", windows_size=windows_size, dir_path='./data/processed/TaxiBJ/')
    print(train_all_data)
    train_all_data_loader = data.DataLoader(train_all_data, batch_size=32, num_workers=0)

    # for idx, data in enumerate(train_data_loader):
    #     input_seq = data[0]
    #     ext_data = data[1]
    #     target = data[2]
        # print('current: {}'.format(idx))
        # print(input_seq.shape, ext_data.shape, target.shape)