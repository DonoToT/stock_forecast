import csv

import numpy as np
import torch
from matplotlib import pyplot as plt

import construct_data

def eval1():
    train_data, test_data = construct_data.construct()
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
            if prob >= 50:
                flag = False
                lists[i] = tot_cnt
            tot_cnt += 1
        if flag:
            lists[i] = -1

    return lists

def analyse():
    tot_array, rows = construct_data.get_numpy()
    test_size = int(rows / 10)
    test_array = np.empty((test_size, 4))
    lists = eval1()

    testi = 0
    for i in range(30, rows):
        if i % 10 == 0:
            test_array[testi] = tot_array[i]
            testi += 1

    tot_stock = 0
    tot_money = 0
    day_win = 0
    day_los = 0
    day_buy = 0
    day_less = 0
    day_unbuy = 0
    x = []
    y = []
    z = []
    # nums = [-200, -100, -50, -30, -10, 10, 30, 50, 100, 200]
    loc = ["小于-10%", "-10%到-5%", "-5%到-3%", "-3%到-1%", "-1%到0%", "0%到1%", "1%到3%", "3%到5%", "5%到10%"]

    for i in range(len(lists)):
        print("第{}天".format(i))
        if lists[i] == -1:
            day_less += 1
            print("该天没有概率大于70%的区间")
            continue
        elif lists[i] <= 4:
            day_unbuy += 1
            print("预测位置为:{}, 实际涨跌幅为:{}, 不进行操作, 总体盈亏:{}".format(loc[lists[i]], test_array[i][3], tot_stock))
        else:
            daily_money = test_array[i][0] - test_array[i][1]   # 今日收盘价 - 昨日收盘价 = 今日盈亏额
            tot_stock += daily_money        # 把每日盈亏额累加
            print("预测位置为:{}, 实际涨跌幅为:{}, 当日盈亏:{}, 总体盈亏:{}".format(loc[lists[i]], test_array[i][3], daily_money, tot_stock))
            day_buy += 1
            if daily_money >= 0:
                day_win += 1
            else:
                day_los += 1
            day_stock = 100.0 / test_array[i][1]
            tot_money += day_stock * test_array[i][0] - day_stock * test_array[i][1]
            print("当日盈亏:{}, 总体盈亏:{}".format(day_stock * test_array[i][0] - day_stock * test_array[i][1], tot_money))
        x.append(i)
        # z.append(day_stock * test_array[i][0] - day_stock * test_array[i][1])
        y.append(tot_money)
    print(day_buy, day_win, day_los, day_less, day_unbuy)

    # 绘制曲线图
    plt.plot(x, y)

    # 添加标题和标签
    plt.title('概率60总计盈亏')
    plt.xlabel('Time')
    plt.ylabel('tot_money')

    # 显示图像
    plt.show()


    # plt.plot(x, z)
    # plt.title('概率60每日盈亏')
    # plt.xlabel('Time')
    # plt.ylabel('daily_money')
    #
    # # 显示图像
    # plt.show()
analyse()
