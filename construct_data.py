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


def get_numpy():
    # 打开csv文件并创建reader对象
    with open('dataset/dataset_total.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # 去掉头标签
        lists = list(reader)  # 加载到二维列表中
        csvfile.close()

    str_array = np.array(lists)  # 转换为NumPy数组

    # 数据清洗
    i = 0
    while i < len(str_array):
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

    print("进入get_numpy函数, 共输出{}行数据".format(rows))

    return float_array, rows


"""
mode(train/test): 表示数据用途为训练或测试, 若为训练则返回一个"训练集"和一个"验证集", 否则返回一个"测试集"
days(1,3,5,...): 表示预测未来多少天
type(True/False): True表示分类模型, False表示回归模型
分类模型考虑将输出结果替换为向量
"""
def construct(mode="train", days=3, type=True):
    data_array, rows = get_numpy()
    test_size = int((rows - (days + 29)) / 7) + 1
    validation_size = test_size
    train_size = rows - (days + 29) - test_size - validation_size

    train_data = np.empty((train_size, 61))
    validation_data = np.empty((validation_size, 61))
    test_data = np.empty((test_size, 61))
    tri = 0
    vi = 0
    tei = 0
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
        if i % 7 == 1:
            test_data[tei] = one_data
            tei += 1
        elif i % 7 == 0:
            validation_data[vi] = one_data
            vi += 1
        else:
            train_data[tri] = one_data
            tri += 1

    if mode == "train":
        print("进入construct函数, 模式为: train, 共读取到 {} 行训练数据和 {} 行验证数据".format(
            len(train_data), len(validation_data)
        ))
        return train_data, validation_data
    else:
        print("进入construct函数, 模式为: test, 共读取到 {} 行测试数据".format(len(train_data)))
        return test_data


construct("train", 1, True)
