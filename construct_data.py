import csv

import numpy
import numpy as np
import torch


def map_range(x):
    if x < -10:
        return 0
    elif -10 <= x < -5:
        return 1
    elif -5 <= x < -3:
        return 2
    elif -3 <= x < -1:
        return 3
    elif -1 <= x < 0:
        return 4
    elif 0 <= x < 1:
        return 5
    elif 1 <= x < 3:
        return 6
    elif 3 <= x < 5:
        return 7
    elif 5 <= x < 10:
        return 8
    else:
        return 9

# def map_range1(x):
#     if x < -10:
#         return 0
#     elif -10 <= x < -5:
#         return 1
#     elif -5 <= x < -3:
#         return 2
#     elif -3 <= x < -1:
#         return 3
#     elif -1 <= x < 0:
#         return 4
#     elif 0 <= x < 1:
#         return 5
#     elif 1 <= x < 3:
#         return 6
#     elif 3 <= x < 5:
#         return 7
#     elif 5 <= x < 10:
#         return 8
#     else:
#         return 9
#
#
# def map_range2(x):
#     if x < -30:
#         return 0
#     elif -30 <= x < -20:
#         return 1
#     elif -20 <= x < -10:
#         return 2
#     elif -10 <= x < -5:
#         return 3
#     elif -5 <= x < 0:
#         return 4
#     elif 0 <= x < 5:
#         return 5
#     elif 5 <= x < 10:
#         return 6
#     elif 10 <= x < 20:
#         return 7
#     elif 20 <= x < 30:
#         return 8
#     else:
#         return 9
#
#
# def map_range3(x):
#     if x < -50:
#         return 0
#     elif -50 <= x < -30:
#         return 1
#     elif -30 <= x < -20:
#         return 2
#     elif -20 <= x < -10:
#         return 3
#     elif -10 <= x < -5:
#         return 4
#     elif -5 <= x < -3:
#         return 5
#     elif -3 <= x < 0:
#         return 6
#     elif 0 <= x < 3:
#         return 7
#     elif 3 <= x < 5:
#         return 8
#     elif 5 <= x < 10:
#         return 9
#     elif 10 <= x < 20:
#         return 10
#     elif 20 <= x < 30:
#         return 11
#     elif 30 <= x < 50:
#         return 12
#     else:
#         return 13
#
#
# def normalizing(x):
#     return (x - (maxx - minn)) / (maxx - minn) * 10


def get_numpy(mode="train"):
    # 打开csv文件并创建reader对象
    with open('dataset/dataset_'+mode+'.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # 去掉头标签
        lists = list(reader)  # 加载到二维列表中
        csvfile.close()

    str_array = np.array(lists)  # 转换为NumPy数组

    # 数据清洗
    i = 0
    while i < len(str_array):
        # print(lists[i][10])
        if not str_array[i][10]:
            str_array = np.delete(str_array, i, axis=0)
        else:
            i += 1

    rows, cols = str_array.shape
    float_array = np.zeros((rows, 4))
    float_array[:, 0] = str_array[:, 5].astype(float)
    float_array[:, 1] = str_array[:, 6].astype(float)
    float_array[:, 2] = str_array[:, 10].astype(float)
    float_array[:, 3] = str_array[:, 12].astype(float)

    print("进入get_numpy函数, 模式为:" + mode + ", 共输出{}行数据".format(rows))

    return float_array, rows


def construct(mode="train", days=3, type=True):
    data_array, rows = get_numpy(mode)
    test_size = int((rows - (days + 29)) / 10) + 1
    train_size = rows - (days + 29) - test_size
    if mode == "test":
        train_size = rows - (days + 29)
    train_data = np.empty((train_size, 61))
    test_data = np.empty((test_size, 61))
    testi = 0
    traini = 0
    for i in range(rows - (days + 29)):
        one_data = np.empty(61)
        for j in range(30):
            one_data[j] = data_array[i + j][2]
            one_data[j + 30] = data_array[i + j][3]
        if type:
            one_data[60] = map_range((data_array[i + days + 29][0] - data_array[i + 29][0]) /
                                     data_array[i + 29][0] * 100)
        else:
            one_data[60] = (data_array[i + days + 29][0] - data_array[i + 29][0]) / data_array[i + 29][0] * 100
        # print(one_data)
        if mode == "train":
            if i % 10 == 0:
                test_data[testi] = one_data
                testi += 1
            else:
                train_data[traini] = one_data
                traini += 1
        else:
            train_data[traini] = one_data
            traini += 1

    if mode == "train":
        print("进入construct函数, 模式为: train, 共读取到 {} 行训练数据和 {} 行测试数据".format(
            len(train_data), len(test_data)
        ))
        return train_data, test_data
    else:
        print("进入construct函数, 模式为: test, 共读取到 {} 行测试数据".format(len(train_data)))
        return train_data

# def construct():
#     test_size = int((rows - 30) / 10) + 1
#     train_size = rows - 30 - test_size
#     train_data = np.empty((train_size, 61))
#     test_data = np.empty((test_size, 61))
#
#     testi = 0
#     traini = 0
#     for i in range(rows - 30):
#         one_data = np.empty(61)
#         for j in range(30):
#             one_data[j] = normalizing(data_array[i + j][2])
#             one_data[j + 30] = data_array[i + j][3]
#         one_data[60] = map_range1(data_array[i + 30][3])
#         # print(one_data)
#         if i % 10 == 0:
#             test_data[testi] = one_data
#             testi += 1
#         else:
#             train_data[traini] = one_data
#             traini += 1
#
#     return train_data, test_data
#
#
# def map_range_two(x):
#     if x > 0:
#         return 1
#     else:
#         return 0
#
#
# def construct_two(mode):
#
#     test_size = int((rows - 32) / 10) + 1
#     train_size = rows - 32 - test_size
#     if not mode:
#         train_size = rows - 32
#     train_data = np.empty((train_size, 61))
#     test_data = np.empty((test_size, 61))
#     testi = 0
#     traini = 0
#     for i in range(rows - 32):
#         one_data = np.empty(61)
#         for j in range(30):
#             one_data[j] = data_array[i + j][2]
#             one_data[j + 30] = data_array[i + j][3]
#         one_data[60] = map_range2((data_array[i + 32][0] - data_array[i + 29][1]) /
#                                   data_array[i + 29][1])
#         # print(one_data)
#         if mode:
#             if i % 10 == 0:
#                 test_data[testi] = one_data
#                 testi += 1
#             else:
#                 train_data[traini] = one_data
#                 traini += 1
#         else:
#             train_data[traini] = one_data
#             traini += 1
#
#     if mode:
#         return train_data, test_data
#     else:
#         return train_data
#
#
#
# def construct2():
#     test_size = int((rows - 32) / 10) + 1
#     train_size = rows - 32 - test_size
#     train_data = np.empty((train_size, 61))
#     test_data = np.empty((test_size, 61))
#
#     testi = 0
#     traini = 0
#     for i in range(rows - 32):
#         one_data = np.empty(61)
#         for j in range(30):
#             one_data[j] = normalizing(data_array[i + j][2])
#             one_data[j + 30] = data_array[i + j][3]
#         one_data[60] = map_range2((data_array[i + 32][0] - data_array[i + 29][1]) /
#                                   data_array[i + 29][1] * 100)
#         # print(one_data)
#         if i % 10 == 0:
#             test_data[testi] = one_data
#             testi += 1
#         else:
#             train_data[traini] = one_data
#             traini += 1
#     return train_data, test_data
#
#
# def construct3():
#     test_size = int((rows - 34) / 10) + 1
#     train_size = rows - 34 - test_size
#     train_data = np.empty((train_size, 61))
#     test_data = np.empty((test_size, 61))
#
#     testi = 0
#     traini = 0
#     for i in range(rows - 34):
#         one_data = np.empty(61)
#         for j in range(30):
#             one_data[j] = normalizing(data_array[i + j][2])
#             one_data[j + 30] = data_array[i + j][3]
#         one_data[60] = map_range3((data_array[i + 34][0] - data_array[i + 29][1]) /
#                                   data_array[i + 29][1] * 100)
#         # print(one_data)
#         if i % 10 == 0:
#             test_data[testi] = one_data
#             testi += 1
#         else:
#             train_data[traini] = one_data
#             traini += 1
#     return train_data, test_data
#
#
# def construct_test():
#     test_size = rows - 30
#     test_data = np.empty((test_size, 61))
#
#     testi = 0
#     for i in range(rows - 30):
#         one_data = np.empty(61)
#         for j in range(30):
#             one_data[j] = normalizing(data_array[i + j][2])
#             one_data[j + 30] = data_array[i + j][3]
#         test_data[testi] = one_data
#         testi += 1
#
#     return test_data


construct("train", 3)
# maxx = max(data_array[:,2])
# minn = min(data_array[:,2])
# print(maxx, minn)

# train_data, test_data = construct2()
# print(train_data, test_data)
# train_data, test_data = construct()
# train_data, test_data = construct2()
# train_data, test_data = construct3()
# print(train_data, test_data)
# testing = construct_test()
# print(testing)

# train_data, test_data = construct_two()
# print(train_data, test_data)