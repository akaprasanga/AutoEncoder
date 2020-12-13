import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Encoder, self).__init__()

        self.input_shape = input_shape
        self.encoder_out_shape = output_shape

        self.linear_one = nn.Linear(self.input_shape, 400)
        self.linear_two = nn.Linear(400, 200)
        self.linear_three = nn.Linear(200, 100)
        self.linear_four = nn.Linear(100, self.encoder_out_shape)

    def forward(self, x):
        x = F.relu(self.linear_one(x))
        x = F.relu(self.linear_two(x))
        x = F.relu(self.linear_three(x))
        x = F.relu(self.linear_four(x))

        return x

class Decoder(nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Decoder, self).__init__()

        self.input_shape = input_shape
        self.decoder_out_shape = output_shape

        self.linear_one = nn.Linear(self.input_shape, 100)
        self.linear_two = nn.Linear(100, 200)
        self.linear_three = nn.Linear(200, 400)
        self.linear_four = nn.Linear(400, self.decoder_out_shape)

    def forward(self, x):
        x = F.relu(self.linear_one(x))
        x = F.relu(self.linear_two(x))
        x = F.relu(self.linear_three(x))
        x = self.linear_four(x)

        return x

class AutoEncoder(nn.Module):

    def __init__(self):

        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(input_shape=487, output_shape=50)
        self.decoder = Decoder(input_shape=50, output_shape=487)

    def forward(self, x):
        x = self.encoder(x)
        x_hat = self.decoder(x)

        return x_hat

def train(input_data):
    # batch_size = 2
    epoch = 100
    # input_data = np.random.randint(1, 100, 1000)
    # x = torch.tensor(input_data, dtype=torch.float)
    # x = x.view(batch_size, 500)

    model = AutoEncoder()

    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for i in range(epoch):
        epoch_loss = 0
        for (batch_num, x) in enumerate(input_data):
            out = model(x)
            loss = criteria(out, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss

        print(f"Epoch = {i} Loss = {epoch_loss}")

    print("Trained ::")

    x_hat = model(x)


    # encoder = Encoder(input_shape=500, output_shape=50)
    # out = encoder(x)
    # print("Encoder Out::", out)

    # decoder = Decoder(input_shape=50, output_shape=500)
    # decoded = decoder(out)

    print("Decoder Out::", x_hat)

X = pd.read_excel(r"D:\GRAD\2020Fall\DeepLEarning7343\Project\AllData.xlsx").to_numpy()
X = torch.tensor(X, dtype=torch.float)
dataset = torch.utils.data.DataLoader(X, batch_size=2)

train(X)

