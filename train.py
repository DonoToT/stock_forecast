import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import *
import torch
from torch.utils.data import Dataset, DataLoader
import construct_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


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


def get_loss_weight(dataset, weight=0.1):
    lenth = 10
    my_weight = [0.0] * lenth
    for i in dataset:
        my_weight[i[-1].item()] += 1
    for i in range(lenth):
        if my_weight[i] > 0:
            my_weight[i] = len(dataset) / my_weight[i]
        else:
            my_weight[i] = len(dataset)
    return torch.tensor(my_weight, dtype=torch.float32) * weight


def tot_train(model, optimizer, writer, epoch, x, y, days):
    train_dataloader = DataLoader(x, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(y, batch_size=64, shuffle=True)
    train_data_size = len(x)
    test_data_size = len(y)
    total_train_step = 0
    total_test_step = 0

    # 定义损失函数的权重
    train_weights = get_loss_weight(x, 0.05)
    test_weights = get_loss_weight(y, 0.05)
    train_loss_fn = nn.CrossEntropyLoss(weight=train_weights).to(device)
    test_loss_fn = nn.CrossEntropyLoss(weight=test_weights).to(device)


    # loss_fn = nn.CrossEntropyLoss().to(device)

    for i in range(epoch):
        print("------第{}轮训练开始------".format(i + 1))

        # 训练步骤
        model.train()
        total_train_loss = 0
        total_train_loss2 = 0
        total_accuracy = 0
        for data in train_dataloader:
            feature, target = data
            feature = feature.to(device)
            target = target.to(device)
            outputs = model(feature)

            probs = torch.softmax(outputs, dim=1)
            centers = torch.sum(probs * torch.arange(10).to(device), dim=1)
            distances = torch.pow(centers - target, 2)
            dis_factor = 0.005

            loss = train_loss_fn(outputs, target) + torch.sum(distances) * dis_factor
            # loss = train_loss_fn(outputs, target)

            total_train_loss = total_train_loss + loss.item()
            total_train_loss2 = total_train_loss2 + torch.sum(distances) * dis_factor
            accuracy = (outputs.argmax(1) == target).sum()
            total_accuracy = total_accuracy + accuracy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)

        print("整体训练集上的Loss: {}, 误差值带来的Loss: {}".format(total_train_loss, total_train_loss2))
        print("整体训练集上的正确率: {}".format(total_accuracy / train_data_size))

        # 测试步骤
        model.eval()
        total_test_loss = 0
        total_test_loss2 = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_dataloader:
                feature, target = data
                feature = feature.to(device)
                target = target.to(device)
                outputs = model(feature)

                probs = torch.softmax(outputs, dim=1)
                centers = torch.sum(probs * torch.arange(10).to(device), dim=1)
                distances = torch.pow(centers - target, 2)
                dis_factor = 0.005

                loss = test_loss_fn(outputs, target) + torch.sum(distances) * dis_factor
                # loss = test_loss_fn(outputs, target)

                total_test_loss = total_test_loss + loss.item()
                total_test_loss2 = total_test_loss2 + torch.sum(distances) * dis_factor
                accuracy = (outputs.argmax(1) == target).sum()
                total_accuracy = total_accuracy + accuracy

        print("整体测试集上的Loss: {}, 误差值带来的Loss: {}".format(total_test_loss, total_test_loss2))
        print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))

        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)

        total_test_step = total_test_step + 1
        if i % 100 == 0:
            torch.save(model, "./models/stock_forecast_{}days.pth".format(days))


def tot_train2(model, optimizer, writer, epoch, x, y, days):
    train_dataloader = DataLoader(x, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(y, batch_size=64, shuffle=True)
    train_data_size = len(x)
    test_data_size = len(y)
    total_train_step = 0
    total_test_step = 0

    # 定义损失函数的权重
    train_loss_fn = nn.MSELoss().to(device)
    test_loss_fn = nn.MSELoss().to(device)

    # loss_fn = nn.CrossEntropyLoss().to(device)

    for i in range(epoch):
        print("------第{}轮训练开始------".format(i + 1))

        # 训练步骤
        model.train()
        total_train_loss = 0
        for data in train_dataloader:
            feature, target = data
            feature = feature.to(device)
            target = target.to(device)
            outputs = model(feature)

            loss = train_loss_fn(outputs, target)
            total_train_loss = total_train_loss + loss.item() * feature.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)

        rmse = torch.sqrt(total_train_loss / train_data_size)
        print("整体训练集上的Loss: {}".format(total_train_loss))
        print("整体训练集上的均方根误差: {}".format(rmse))

        # 测试步骤
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for data in test_dataloader:
                feature, target = data
                feature = feature.to(device)
                target = target.to(device)
                outputs = model(feature)

                loss = test_loss_fn(outputs, target)
                total_test_loss = total_test_loss + loss.item() * feature.size(0)

        rmse = torch.sqrt(total_test_loss / test_data_size)
        print("整体测试集上的Loss: {}".format(total_test_loss))
        print("整体测试集上的均方根误差: {}".format(rmse))

        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_rmse", rmse, total_test_step)

        total_test_step += 1
        if i % 100 == 0:
            torch.save(model, "./models/stock_forecast_{}days.pth".format(days))


def train(days=3, epoch=100, learning_rate=1e-2, choose_model=1, type=True):
    writer = SummaryWriter("./logs_train_{}".format(days))
    print("------对后{}天的结果预测模型训练开始------".format(days))

    train_data, test_data = construct_data.construct("train", days, type)
    train_data = change_data(train_data)
    test_data = change_data(test_data)
    global model
    if choose_model == 0:
        model = torch.load("./models/stock_forecast_{}days.pth".format(days), map_location=torch.device('cuda'))
    if choose_model == 1:
        model = ClsModel().to(device)
    elif choose_model == 2:
        model = ClsModel2().to(device)
    elif choose_model == 3:
        model = ClsModel3().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if type:
        tot_train(model, optimizer, writer, epoch, train_data, test_data, days)
    else:
        tot_train2(model, optimizer, writer, epoch, train_data, test_data, days)

    writer.close()


train(days=1, epoch=1000, learning_rate=1e-2, choose_model=2, type=True)
# train1()
# train2()
# train3()