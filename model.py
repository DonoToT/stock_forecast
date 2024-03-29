import torch
from torch import nn


class ClsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(60, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x


class ClsModel2(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.linear1 = nn.Linear(60, 128)
        self.dropout1 = nn.Dropout(p=p)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout1(x)
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class ClsModel3(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.linear1 = nn.Linear(60, 128)
        self.dropout1 = nn.Dropout(p=p)
        self.linear2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=p)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x


class ClsModel4(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.linear1 = nn.Linear(30, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 10)
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        # x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class ClsModel5(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(30, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, 10)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class ClsModel6(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(30, 30)
        self.linear2 = nn.Linear(30, 2)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

#
# input_x = torch.randn(1, 60)
# model1 = ClsModel()
# torch.save(model1, "./models/model1.pth")
# torch.onnx.export(model1, input_x, "./models/model1.onnx")
# print("--------模型已保存--------")


# pt_model = torch.load('classification_model.pt')
# pt_model.eval()
#
# input_names = ['input']
# output_names = ['output']
#
# input_x = torch.randn(1, 82)
#
# torch.onnx.export(model, input_x, 'classification_model.onnx', input_names=input_names, output_names=output_names, verbose='True')
# print("----- Finished -----")
