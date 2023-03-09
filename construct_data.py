import csv
import numpy as np


def map_range1(x):
    if x < -10:
        return 0
    elif -10 <= x < -5:
        return 1
    elif -5 <= x < -3:
        return 2
    elif -3 <= x < -1:
        return 3
    elif -1 <= x < 1:
        return 4
    elif 1 <= x < 3:
        return 5
    elif 3 <= x < 5:
        return 6
    elif 5 <= x < 10:
        return 7
    else:
        return 8


def map_range2(x):
    if x < -30:
        return 0
    elif -30 <= x < -20:
        return 1
    elif -20 <= x < -10:
        return 2
    elif -10 <= x < -5:
        return 3
    elif -5 <= x < 0:
        return 4
    elif 0 <= x < 5:
        return 5
    elif 5 <= x < 10:
        return 6
    elif 10 <= x < 20:
        return 7
    elif 20 <= x < 30:
        return 8
    else:
        return 9


def map_range3(x):
    if x < -50:
        return 0
    elif -50 <= x < -30:
        return 1
    elif -30 <= x < -20:
        return 2
    elif -20 <= x < -10:
        return 3
    elif -10 <= x < -5:
        return 4
    elif -5 <= x < -3:
        return 5
    elif -3 <= x < 0:
        return 6
    elif 0 <= x < 3:
        return 7
    elif 3 <= x < 5:
        return 8
    elif 5 <= x < 10:
        return 9
    elif 10 <= x < 20:
        return 10
    elif 20 <= x < 30:
        return 11
    elif 30 <= x < 50:
        return 12
    else:
        return 13


def get_numpy():
    # 打开csv文件并创建reader对象
    with open('dataset_train.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # 去掉头标签
        lists = list(reader)  # 加载到二维列表中
        csvfile.close()

    data_array = np.array(lists)  # 转换为NumPy数组
    rows, cols = data_array.shape
    print(rows, cols)
    return data_array, rows, cols


def construct():
    train_data = np.empty((5565, 61))
    test_data = np.empty((55, 61))

    ti = 0
    for i in range(rows - 30):
        one_data = np.empty(61)
        for j in range(30):
            one_data[j] = data_array[i + j][7]
            one_data[j + 30] = data_array[i + j][9]
        one_data[60] = map_range1(float(data_array[i + 30][9]))
        train_data[i] = one_data
        # print(one_data)
        if i % 100 == 0 and ti < 55:
            test_data[ti] = one_data
            ti = ti + 1
    return train_data, test_data


def construct2():
    train_data = np.empty((5562, 61))
    test_data = np.empty((55, 61))

    ti = 0
    for i in range(rows - 33):
        one_data = np.empty(61)
        for j in range(30):
            one_data[j] = data_array[i + j][7]
            one_data[j + 30] = data_array[i + j][9]
        one_data[60] = map_range2(float(data_array[i + 30][9]) + float(data_array[i + 31][9]) +
                                  float(data_array[i + 32][9]))
        train_data[i] = one_data
        # print(one_data)
        if i % 100 == 0 and ti < 55:
            test_data[ti] = one_data
            ti = ti + 1
    return train_data, test_data


def construct3():
    train_data = np.empty((5560, 61))
    test_data = np.empty((55, 61))

    ti = 0
    for i in range(rows - 35):
        one_data = np.empty(61)
        for j in range(30):
            one_data[j] = data_array[i + j][7]
            one_data[j + 30] = data_array[i + j][9]
        one_data[60] = map_range3(float(data_array[i + 30][9]) + float(data_array[i + 31][9]) +
                                  float(data_array[i + 32][9]) + float(data_array[i + 33][9]) + float(data_array[i + 34][9]))
        train_data[i] = one_data
        # print(one_data)
        if i % 100 == 0 and ti < 55:
            test_data[ti] = one_data
            ti = ti + 1
    return train_data, test_data


data_array, rows, cols = get_numpy()
# train_data, test_data = construct2()
# print(train_data, test_data)