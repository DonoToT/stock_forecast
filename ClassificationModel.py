# Classification Model
import torch
from torch import nn


class ClsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(82, 100)
        self.linear2 = nn.Linear(100, 2)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))

        return x


print("----- Finished -----")

input_x = torch.randn(1, 82)
model = ClsModel()
traced_script_module = torch.jit.trace(model, input_x)
traced_script_module.save("classification_model.pt")

print("----- Finished -----")

pt_model = torch.load('classification_model.pt')
pt_model.eval()

input_names = ['input']
output_names = ['output']

input_x = torch.randn(1, 82)

torch.onnx.export(model, input_x, 'classification_model.onnx', input_names=input_names, output_names=output_names, verbose='True')
print("----- Finished -----")