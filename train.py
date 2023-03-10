import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import *
import torch
from torch.utils.data import Dataset, DataLoader
import construct_data

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        features = self.data[index, :60].to(torch.float32)
        label = self.data[index, 60].to(torch.long)
        return features, label

    def __len__(self):
        return len(self.data)


def change_data(x):
    x = torch.from_numpy(x)
    x = MyDataset(x)
    return x


learning_rate = 1e-3


def tot_train(model, optimizer, writer, epoch, x, y, flag):
    train_dataloader = DataLoader(x, batch_size=64)
    test_dataloader = DataLoader(y, batch_size=64)
    train_data_size = len(x)
    test_data_size = len(y)
    total_train_step = 0
    total_test_step = 0

    # 定义损失函数的权重
    lenth = 0
    if flag == 3:
        lenth = 14
    else:
        lenth = 10
    my_weight = [0.0] * lenth
    for i in x:
        my_weight[i[-1].item()] += 1
    for i in range(lenth):
        if my_weight[i] > 0:
            my_weight[i] = train_data_size / my_weight[i]
        else:
            my_weight[i] = train_data_size
    weights = torch.tensor(my_weight, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)

    for i in range(epoch):
        print("------第{}轮训练开始------".format(i + 1))


        # 训练步骤
        model.train()
        total_train_loss = 0
        total_accuracy = 0
        for data in train_dataloader:
            feature, target = data
            feature = feature.to(device)
            target = target.to(device)
            outputs = model(feature)
            loss = loss_fn(outputs, target)
            total_train_loss = total_train_loss + loss.item()
            accuracy = (outputs.argmax(1) == target).sum()
            total_accuracy = total_accuracy + accuracy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)

        print("整体训练集上的Loss: {}".format(total_train_loss))
        print("整体训练集上的正确率: {}".format(total_accuracy / train_data_size))

        # 测试步骤
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_dataloader:
                feature, target = data
                feature = feature.to(device)
                target = target.to(device)
                outputs = model(feature)
                loss = loss_fn(outputs, target)
                total_test_loss = total_test_loss + loss.item()
                accuracy = (outputs.argmax(1) == target).sum()
                total_accuracy = total_accuracy + accuracy

        print("整体测试集上的Loss: {}".format(total_test_loss))
        print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))

        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)

        total_test_step = total_test_step + 1
        torch.save(model, "./models/stock_forecast{}.pth".format(flag))


def train1():
    writer = SummaryWriter("./logs_train")
    print("------对后1天的结果预测模型训练开始------")
    # 定义需要的参数, 包括数据, 模型, 优化器, 迭代次数等
    epoch = 100
    train_data, test_data = construct_data.construct()
    train_data = change_data(train_data)
    test_data = change_data(test_data)
    model = ClsModel().to(device)
    # model = torch.load("./models/stock_forecast1.pth", map_location=torch.device('cuda'))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tot_train(model, optimizer, writer, epoch, train_data, test_data, 1)

    writer.close()


def train2():
    writer = SummaryWriter("./logs_train2")
    print("------对后3天的结果预测模型训练开始------")
    # 定义需要的参数, 包括数据, 模型, 优化器, 迭代次数等
    epoch = 100
    train_data, test_data = construct_data.construct2()
    train_data = change_data(train_data)
    test_data = change_data(test_data)
    model = ClsModel2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tot_train(model, optimizer, writer, epoch, train_data, test_data, 2)

    writer.close()


def train3():
    writer = SummaryWriter("./logs_train3")
    print("------对后5天的结果预测模型训练开始------")
    # 定义需要的参数, 包括数据, 模型, 优化器, 迭代次数等
    epoch = 100
    train_data, test_data = construct_data.construct3()
    train_data = change_data(train_data)
    test_data = change_data(test_data)
    model = ClsModel2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tot_train(model, optimizer, writer, epoch, train_data, test_data, 3)

    writer.close()


# train1()
# train2()
# train3()