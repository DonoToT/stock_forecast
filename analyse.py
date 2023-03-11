import csv

import numpy as np
import torch

import construct_data


def get_numpy():
    # 打开csv文件并创建reader对象
    with open('dataset/dataset_test.csv', newline='') as csvfile:
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

    return float_array, rows


def eval1():
    test_data = construct_data.construct_test()
    test_data_size = len(test_data)
    model = torch.load("./models/stock_forecast1 (1).pth", map_location=torch.device('cuda'))
    model.eval()

    lists = np.empty(test_data_size, dtype=np.int64)

    for i in range(test_data_size):
        temp = torch.tensor(test_data[i,:-1]).to(torch.float32)
        output = model(temp.to('cuda'))
        output = output.cpu()
        output = output.detach().numpy()
        sum = output.sum()

        tot_cnt = 0

        flag = True
        for j in output:
            prob = j / sum * 100
            if prob >= 70:
                flag = False
                lists[i] = tot_cnt
            tot_cnt += 1
        if flag:
            lists[i] = -1

    return lists

def analyse():
    tot_array, rows = get_numpy()
    lists = eval1()

    stock_num = 0
    hand_money = 0

    nums = [-200, -100, -50, -30, -10, 10, 30, 50, 100, 200]
    loc = ["小于-10%", "-10%到-5%", "-5%到-3%", "-3%到-1%", "-1%到0%", "0%到1%", "1%到3%", "3%到5%", "5%到10%"]

    for i in range(len(lists)):
        if lists[i] == -1:
            continue
        num = nums[lists[i]]
        hand_money -= num * tot_array[i + 29][0]
        stock_num += num
        # if num > 0:
        #     hand_money -= num * tot_array[i + 29][0]
        #     stock_num += num
        # else:
        #     hand_money += num * tot_array[i + 29][0]
        #     stock_num += num
            # if stock_num <= abs(num):
            #     hand_money += stock_num * tot_array[i + 29][0]
            #     stock_num = 0
            # else:
            #     hand_money -= num * tot_array[i + 29][0]
            #     stock_num += num
        tot_money = hand_money + stock_num * tot_array[i + 30][0]
        print("第{}天, 手头金钱:{}, 股票数:{}, 总金额:{}".format(i, hand_money, stock_num, tot_money))
        print("预测位置为:{}, 实际涨跌幅为:{}%, 当前股价为:{}".format(loc[lists[i]], tot_array[i + 30][3], tot_array[i + 30][0]))


analyse()
