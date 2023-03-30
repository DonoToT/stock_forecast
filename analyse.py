import csv

import numpy as np
import torch
from matplotlib import pyplot as plt

import construct_data
import eval


def chg(now, last):
    return (now - last) / last * 10


def analyse(mode="test", days=1, probablity=70, fw=15):
    tot_array = np.zeros((0, 4))
    rows = 0
    for i in range(3):
        one_array, row = construct_data.get_numpy(i)
        tot_array = np.vstack((tot_array, one_array))
        rows += row
    lists = eval.eval(mode, days, probablity)

    day_less = 0
    day_unbuy = 0
    day_buy = 0
    day_win = 0
    day_los = 0
    tot_money = 0
    x = []
    y = []
    loc = ["低于-10%", "-10%~-5%", "-5%~-3%", "-3%~-1%", "-1%~0%", "0%~1%", "1%~3%", "3%~5%", "5%~10%", "高于10%"]
    for i in range(len(lists)):
        print("第{}天".format(i + 1), end=' ')
        now_chg = chg(tot_array[i + (fw - 1 + days)][0], tot_array[i + fw - 1][0])
        if lists[i] == -1:
            day_less += 1
            print("该天没有概率大于{}的区间".format(probablity))
            continue
        elif lists[i] <= 4:
            day_unbuy += 1
            print("预测位置为:{}, 实际涨跌幅为:{}, 不进行操作".format(loc[lists[i]], now_chg))
        else:
            print(
                "预测位置为:{}, 实际涨跌幅为:{}".format(loc[lists[i]], now_chg))
            day_buy += 1
            if now_chg >= 0:
                day_win += 1
            else:
                day_los += 1
            day_stock = 100.0 / tot_array[i + fw - 1][0]
            tot_money += day_stock * tot_array[i + (fw - 1 + days)][0] - day_stock * tot_array[i + fw - 1][0]
            print("当日盈亏:{}, 总体盈亏:{}".format(day_stock * tot_array[i + (fw - 1 + days)][0] - day_stock * tot_array[i + fw - 1][0],
                                            tot_money))
        x.append(i)
        # z.append(day_stock * test_array[i][0] - day_stock * test_array[i][1])
        y.append(tot_money)

    print(day_buy, day_win, day_los, day_less, day_unbuy)

    # 绘制曲线图
    plt.plot(x, y)

    # 添加标题和标签
    plt.title('{}percent_statistic'.format(probablity))
    plt.xlabel('Time')
    plt.ylabel('tot_money')

    # 显示图像
    plt.show()

analyse(mode="test", days=1, probablity=70, fw=15)
