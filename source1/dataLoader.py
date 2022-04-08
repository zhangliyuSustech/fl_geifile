import numpy as np
import pandas as pd
import os
import torch
import h3
from collections import OrderedDict


def dataset(user, start, end, step, offset=0):
    userdata = '../Geolife Trajectories 1.3/Data/' + user + '/Trajectory/'
    filelist = os.listdir(userdata)  # 返回指定路径下所有文件和文件夹的名字，并存放于一个列表中
    filelist.sort()

    names = ['lat', 'lng', 'zero', 'alt', 'days', 'date', 'time']
    df_list = [  # f为文件索引号，header为列数，names为列表列名，index_col为行索引的列编号或列名
        pd.read_csv(userdata + f, header=6, names=names, index_col=False)
        for f in filelist[start:end]]
    # 上面这一步的是每一天的数据都是一个列表，并且还有表头,下面concat之后才是很多行，7列
    df = pd.concat(df_list, ignore_index=True)  # 表格列字段不同的表合并

    df.drop(['zero', 'days'], axis=1, inplace=True)  # drop函数默认删除行，列需要加axis = 1
    df_min = df.iloc[offset::step, :]
    return df_min


def synthetic_data(df_min):
    a = df_min['lat'].tolist()
    b = df_min['lng'].tolist()
    a = torch.tensor(a, dtype=torch.float, requires_grad=True).reshape((-1, 1))
    b = torch.tensor(b, dtype=torch.float, requires_grad=True).reshape((-1, 1))
    features = torch.concat([a, b], 1)
    return features


# 返回（经度，纬度） shape：torch.Size([368, 2])


def geo_t_h3_norepeat(data):
    h3_list = OrderedDict()
    for i in data:
        a = h3.geo_to_h3(i[0], i[1], 10)
        # print(a)
        h3_list.setdefault(a)
    # 这这里去掉h3的重复
    return h3_list


def h3_t_geo(data):
    new_list = []
    for i in data:
        i = h3.h3_to_geo(i)
        new_list.append(i)
    return new_list


def generate_h3_list(data, label='repeat'):
    if label == 'no-repeat':
        # 不可重复的
        alist = geo_t_h3_norepeat(data)
        # print(type(alist))
        LIST = list(alist.keys())
        return np.array(LIST)
    elif label == 'repeat':
        LIST = []
        for temp in data:
            LIST.append(h3.geo_to_h3(temp[0], temp[1], 10))
        return np.array(LIST)
    else:
        return np.array([])


class DataLoader:
    @staticmethod
    def load_rnn_data(list_h3, step):
        data = []
        for i in range(len(list_h3) - step + 1):
            indata = list_h3[i:i + step]
            outdata =list_h3[i + step:i + step + 1]
            data.append((indata, outdata))
        return data

    @staticmethod
    def load_h3_list():
        # 这个提取出来有5个维度
        # 要在这里添加间隔
        train_dataset = dataset("006", 0, 20, 20, 0)
        test_dataset = dataset("006", 0, 20, 20, 10)
        # 这个提取出来有2个维度
        train_data = synthetic_data(train_dataset)
        test_data = synthetic_data(test_dataset)
        all_data = torch.concat([train_data, test_data], 0)
        Train_h3_list = generate_h3_list(train_data, label='repeat')
        Test_h3_list = generate_h3_list(test_data, label='repeat')
        # 这个词典
        vocab = generate_h3_list(all_data, label='no-repeat')  # vocab也是h3

        train_dataloader = DataLoader.load_rnn_data(Train_h3_list, 10)
        test_dataloader = DataLoader.load_rnn_data(Test_h3_list, 10)

        return train_dataloader,test_dataloader,vocab
