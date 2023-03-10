import torch
from torch.utils.data import DataLoader

import construct_data
import train

# train_data, test_data = construct_data.construct()
# print(test_data[0])

# model = torch.load("stock_forecast1.pth", map_location=torch.device('cpu'))
# print(model)

# model.eval()
# output = model(torch.tensor(test_data[1, :60]).to(torch.float32))
# output = output.detach().numpy()
# print(output)

# sum = output.sum()

# print("未来1天涨跌率低于-10%的概率为: {:.4}%".format(output[0] / sum * 100))
# print("未来1天涨跌率在-10%到-5%的概率为: {:.4}%".format(output[1] / sum * 100))
# print("未来1天涨跌率在-5%到-3%的概率为: {:.4}%".format(output[2] / sum * 100))
# print("未来1天涨跌率在-3%到-1%的概率为: {:.4}%".format(output[3] / sum * 100))
# print("未来1天涨跌率在-1%到1%的概率为: {:.4}%".format(output[4] / sum * 100))
# print("未来1天涨跌率在1%到3%的概率为: {:.4}%".format(output[5] / sum * 100))
# print("未来1天涨跌率在3%到5%的概率为: {:.4}%".format(output[6] / sum * 100))
# print("未来1天涨跌率在5%到10%的概率为: {:.4}%".format(output[7] / sum * 100))
# print("未来1天涨跌率高于10%的概率为: {:.4}%".format(output[8] / sum * 100))


# 251 -> 2
# 227 -> 8
# 223 -> 1
# 204 -> 0
# 0 -> 5

def eval1():
    # train_data, test_data = construct_data.construct()
    test_data = construct_data.construct_test()
    test_data_size = len(test_data)
    model = torch.load("./models/stock_forecast1 (1).pth", map_location=torch.device('cuda'))
    model.eval()

    seventy_cnt = 0
    right = 0
    near1 = 0
    near2 = 0
    wrong = 0

    for i in range(test_data_size):
        temp = torch.tensor(test_data[i,:-1]).to(torch.float32)
        target = test_data[i, -1]
        output = model(temp.to('cuda'))
        output = output.cpu()
        output = output.detach().numpy()
        sum = output.sum()

        print("第{}行数据目标为{}, 预测概率为: ".format(i, target), end="")
        tot_cnt = 0

        for j in output:
            prob = j / sum * 100
            if prob >= 70:
                seventy_cnt += 1
                print("[{}]:{:.4}%".format(tot_cnt, j / sum * 100), end="\t")
                if int(target + 0.1) == tot_cnt:
                    right += 1
                elif abs(int(target + 0.1) - tot_cnt) == 1:
                    near1 += 1
                elif abs(int(target + 0.1) - tot_cnt) == 2:
                    near2 += 1
                else:
                    wrong += 1

            tot_cnt += 1
        print("")
    print("测试数据共有:{}个".format(test_data_size))
    print("预测概率最大值大于70%的共有:{}".format(seventy_cnt))
    print("其中预测区间正确的共有:{}".format(right))
    print("预测预测区间与实际区间距离为1的共有:{}".format(near1))
    print("预测预测区间与实际区间距离为2的共有:{}".format(near2))
    print("其他(错误)的共有:{}".format(wrong))
    print(seventy_cnt, right, near1, near2, wrong)


def eval2():
    train_data, test_data = construct_data.construct2()
    model = torch.load("stock_forecast2.pth", map_location=torch.device('cpu'))
    model.eval()
    output = model(torch.tensor(test_data[1, :60]).to(torch.float32))
    output = output.detach().numpy()
    sum = output.sum()

    print("***************************")
    print("未来3天涨跌率低于-30%的概率为: {:.4}%".format(output[0] / sum * 100))
    print("未来3天涨跌率在-30%到-20%的概率为: {:.4}%".format(output[1] / sum * 100))
    print("未来3天涨跌率在-20%到-10%的概率为: {:.4}%".format(output[2] / sum * 100))
    print("未来3天涨跌率在-10%到-5%的概率为: {:.4}%".format(output[3] / sum * 100))
    print("未来3天涨跌率在-5%到0%的概率为: {:.4}%".format(output[4] / sum * 100))
    print("未来3天涨跌率在0%到5%的概率为: {:.4}%".format(output[5] / sum * 100))
    print("未来3天涨跌率在5%到10%的概率为: {:.4}%".format(output[6] / sum * 100))
    print("未来3天涨跌率在10%到20%的概率为: {:.4}%".format(output[7] / sum * 100))
    print("未来3天涨跌率在20%到30%的概率为: {:.4}%".format(output[8] / sum * 100))
    print("未来3天涨跌率高于30%的概率为: {:.4}%".format(output[9] / sum * 100))


def eval3():
    train_data, test_data = construct_data.construct3()
    model = torch.load("stock_forecast3.pth", map_location=torch.device('cpu'))
    model.eval()
    output = model(torch.tensor(test_data[1, :60]).to(torch.float32))
    output = output.detach().numpy()
    sum = output.sum()

    print("***************************")
    print("未来5天涨跌率低于-50%的概率为: {:.4}%".format(output[0] / sum * 100))
    print("未来5天涨跌率在-50%到-30%的概率为: {:.4}%".format(output[1] / sum * 100))
    print("未来5天涨跌率在-30%到-20%的概率为: {:.4}%".format(output[2] / sum * 100))
    print("未来5天涨跌率在-20%到-10%的概率为: {:.4}%".format(output[3] / sum * 100))
    print("未来5天涨跌率在-10%到-5%的概率为: {:.4}%".format(output[4] / sum * 100))
    print("未来5天涨跌率在-5%到-3%的概率为: {:.4}%".format(output[5] / sum * 100))
    print("未来5天涨跌率在-3%到0%的概率为: {:.4}%".format(output[6] / sum * 100))
    print("未来5天涨跌率在0%到3%的概率为: {:.4}%".format(output[7] / sum * 100))
    print("未来5天涨跌率在3%到5%的概率为: {:.4}%".format(output[8] / sum * 100))
    print("未来5天涨跌率在5%到10%的概率为: {:.4}%".format(output[9] / sum * 100))
    print("未来5天涨跌率在10%到20%的概率为: {:.4}%".format(output[10] / sum * 100))
    print("未来5天涨跌率在20%到30%的概率为: {:.4}%".format(output[11] / sum * 100))
    print("未来5天涨跌率在30%到50%的概率为: {:.4}%".format(output[12] / sum * 100))
    print("未来5天涨跌率高于50%的概率为: {:.4}%".format(output[13] / sum * 100))


eval1()
# eval2()
# eval3()
