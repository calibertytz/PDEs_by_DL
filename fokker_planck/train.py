import torch
from model import NeuralNet
from data_prepare import PdeDataset, gen_train_data
from torch.utils.data import DataLoader
from utils import batch_get_difference, grad_net
from tqdm import tqdm
import argparse
import os

model_save_path = 'model_save'
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
else:
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-2)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_epoch', type=int, default=30000)
parser.add_argument('--hidden_size', type=int, default=20)
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--sigma', type=float, default=0.5)
parser.add_argument('--a', type=float, default=0.3)
parser.add_argument('--b', type=float, default=0.5)
parser.add_argument('--phase', type=str, default='grad')
config = parser.parse_args()

print(config)


f_func = lambda x: config.a * x - config.b * x**3
d_f_func = lambda x : config.a - 3 * config.b * x ** 2


train_data = gen_train_data(config.dt)
train_data_1 = PdeDataset(data=train_data[0])
train_data_2 = PdeDataset(data=train_data[1])
train_loader_1 = DataLoader(dataset=train_data_1, batch_size=config.batch_size, shuffle=True)
train_loader_2 = DataLoader(dataset=train_data_2, batch_size=config.batch_size, shuffle=True)

model = NeuralNet(1, config.hidden_size, 1)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)


def train(config):
    for epoch in tqdm(range(config.num_epoch)):
        for x, y in zip(train_loader_1, train_loader_2):
            var_x = x.cuda().requires_grad_()
            var_y = y.cuda().requires_grad_()
            out1 = model(var_x)
            out2 = model(var_y)
            f_value = f_func(var_x)

            if config.phase == 'difference':
                e1 = ((-batch_get_difference(f_value * out1, config.dt) +
                      ((config.sigma ** 2)/2) * batch_get_difference(out1, config.dt, order=2))**2).mean()
            elif config.phase == 'grad':
                f_deriviate_value = d_f_func(var_x)
                e1 = ((-(f_value * grad_net(out1, var_x) + f_deriviate_value * out1) +
                      ((config.sigma**2) / 2) * grad_net(out1, var_x, order=2))**2).mean()
            else:
                raise NotImplementedError

            e2 = (torch.abs(config.dt * out1.sum() - 1)) ** 2
            e3 = (out2 ** 2).mean()
            e1, e2, e3 = e1.cuda(), e2.cuda(), e3.cuda()
            loss = e1 + e2 + e3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch % 100) == 0:
                print(f'epoch: {epoch}, loss:{loss.item()}')

if __name__ == '__main__':
    train(config)
    torch.save(model, f'model_save/model_{config.phase}.pth')