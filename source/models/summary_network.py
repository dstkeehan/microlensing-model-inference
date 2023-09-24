import torch
import torch.nn as nn
import torch.nn.functional as F

# class ResnetGRU(nn.Module): 

#     def __init__(self): 
#         super().__init__()
#         # 1D convolutional layer
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=5, padding=2)
#         # Maxpool layer that reduces 32x32 image to 4x4
#         self.pool = nn.MaxPool1d(kernel_size = 4, stride = 4)
#         # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
#         self.fc = nn.Linear(in_features = 8 * 180, out_features=256) 

#     def forward(self, x):
#         x = x.view(-1, 1, 720)

#         # print(x.shape)

#         x = F.relu(self.conv1(x))

#         # print(x.shape)

        
#         x = self.pool(F.relu(self.conv2(x)))
#         # x = self.pool(x)

#         # print(x.shape)

#         x = x.view(-1, 8 * 180)

#         x = F.relu(self.fc(x))

#         # print(x.shape)

#         return x

# class MLP(nn.Module):
#     def __init__(self): 
#         super().__init__()

#         self.net = nn.Sequential(*[
#             # nn.Dropout(0.1),
#             nn.Linear(720, 500),
#             nn.ReLU(),
#             # nn.Dropout(0.2),
#             nn.Linear(500, 500),
#             nn.ReLU(),
#             # nn.Dropout(0.2),
#             nn.Linear(500, 500),
#             nn.ReLU(),
#             # nn.Dropout(0.3),
#             nn.Softmax()
#         ])

#     def forward(self, x):
#         return self.net(x)


# class ResBlock(nn.Module):
#     def __init__(self, in_ch, out_ch): 
#         super().__init__()

#         self.net = nn.Sequential(*[
#             nn.Conv1d(in_channels = in_ch, out_channels = out_ch, kernel_size = 8, padding = "same"),
#             nn.BatchNorm1d(out_ch),
#             nn.ReLU(),

#             nn.Conv1d(in_channels = out_ch, out_channels = out_ch, kernel_size = 5, padding = "same"),
#             nn.BatchNorm1d(out_ch),
#             nn.ReLU(),

#             nn.Conv1d(in_channels = out_ch, out_channels = out_ch, kernel_size = 3, padding = "same"),
#             nn.BatchNorm1d(out_ch),
#         ])

#         self.res = nn.Sequential(*[
#             nn.Conv1d(in_channels = in_ch, out_channels = out_ch, kernel_size = 1, padding = "same"),
#             nn.BatchNorm1d(out_ch),
#         ])

#     def forward(self, x):
#         y = self.net(x)

#         z = self.res(x)

#         return F.relu(y + z)


# class ResNet(nn.Module):
#     def __init__(self): 
#         super().__init__()

#         self.net = nn.Sequential(*[
#             ResBlock(1, 64),
#             ResBlock(64, 128),
#             ResBlock(128, 128),
#             nn.AvgPool1d(720, 1)
#         ])

#         self.after = nn.Sequential(*[
#             nn.Linear(128, 256),
#             # removed softmax
#             # nn.Softmax()
#         ])

#     def forward(self, x):
#         x = x.view(-1, 1, 720)

#         x = self.net(x)

#         x = x.view(-1, 1, 128)

#         x = self.after(x)

#         # print(x.shape)

#         return torch.squeeze(x, 1)

# class LinearNetwork(nn.Module): 

#     def __init__(self): 
#         super().__init__()
        
#         self.net = nn.Sequential(*[
#             nn.Linear(720, 400),
#             nn.ReLU(),
#             nn.Linear(400, 400),
#             nn.ReLU(),
#             nn.Linear(400, 256),
#         ])

#     def forward(self, x):
#         return self.net(x)

# model = ResnetGRU()


# from torchinfo import summary

# summary(model, input_size = (1, 720))