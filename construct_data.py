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


def map_range2(x):
    if x < 5:
        return 0
    else:
        return 1


def get_numpy(index):
    # 打开csv文件并创建reader对象
    totlist = list()
    with open('dataset/dataset_total{}.csv'.format(index), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # 去掉头标签
        lists = list(reader)  # 加载到二维列表中
        totlist.extend(lists)
        csvfile.close()

    # 数据清洗
    i = 0
    while i < len(totlist):
        if not totlist[i][10]:
            del totlist[i]
        else:
            i += 1

    str_array = np.array(totlist)  # 转换为NumPy数组

    rows, cols = str_array.shape
    float_array = np.zeros((rows, 4))
    float_array[:, 0] = str_array[:, 5].astype(float)
    float_array[:, 1] = str_array[:, 6].astype(float)
    float_array[:, 2] = str_array[:, 10].astype(float)
    float_array[:, 3] = str_array[:, 12].astype(float)

    print("进入get_numpy函数, dataset_total{}读取完毕, 共输出{}行数据".format(index, rows))

    return float_array, rows


"""
mode(train/test): 表示数据用途为训练或测试, 若为训练则返回一个"训练集"和一个"验证集", 否则返回一个"测试集"
days(1,3,5,...): 表示预测未来多少天
type(True/False): True表示分类模型, False表示回归模型
分类模型考虑将输出结果替换为向量
"""
def construct(mode="train", days=3, type=True, fw=30):
    final_train_data = np.empty((0, fw * 2 + 1))
    final_validation_data = np.empty((0, fw * 2 + 1))
    final_test_data = np.empty((0, fw * 2 + 1))

    for i in range(10):
        data_array, rows = get_numpy(i)
        validation_size = int((rows - (days + fw - 1)) / 10)
        test_size = validation_size
        if (rows - (days + fw - 1)) % 10 > 0:
            validation_size += 1
        if (rows - (days + fw - 1)) % 10 > 1:
            test_size += 1

        train_size = rows - (days + fw - 1) - test_size - validation_size

        train_data = np.empty((train_size, fw * 2 + 1))
        validation_data = np.empty((validation_size, fw * 2 + 1))
        test_data = np.empty((test_size, fw * 2 + 1))
        tri = 0
        vi = 0
        tei = 0
        for i in range(rows - (days + fw - 1)):
            one_data = np.empty(fw * 2 + 1)
            for j in range(fw):
                one_data[j] = data_array[i + j][2]
                one_data[j + fw] = data_array[i + j][3]
            if type:
                one_data[fw * 2] = map_range2((data_array[i + days + fw - 1][0] - data_array[i + fw - 1][0]) /
                                         data_array[i + fw - 1][0] * 100)
            else:
                one_data[fw * 2] = (data_array[i + days + fw - 1][0] - data_array[i + fw - 1][0]) / \
                                   data_array[i + fw - 1][0] * 100
            # print(one_data)
            if i % 10 == 1:
                test_data[tei] = one_data
                tei += 1
            elif i % 10 == 0:
                validation_data[vi] = one_data
                vi += 1
            else:
                train_data[tri] = one_data
                tri += 1

        final_train_data = np.vstack((final_train_data, train_data))
        final_test_data = np.vstack((final_test_data, test_data))
        final_validation_data = np.vstack((final_validation_data, validation_data))

    if mode == "train":
        print("进入construct函数, 模式为: train, 共读取到 {} 行训练数据和 {} 行验证数据".format(
            len(final_train_data), len(final_validation_data)
        ))
        return final_train_data, final_validation_data
    else:
        print("进入construct函数, 模式为: test, 共读取到 {} 行测试数据".format(len(final_test_data)))
        return final_test_data


# construct("train", 5, True, 15)
# get_numpy()
