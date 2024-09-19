import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import precision_recall_curve, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import time
import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd

Normalization = StandardScaler()
minmax = MinMaxScaler()

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--data", default='./data/A1.csv')
parser.add_argument("--n_epochs", type=int, default=300,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
# 你的噪声维度
# 间隔采样
parser.add_argument("--sample_interval", type=int,
                    default=20, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

data = pd.read_csv(opt.data)

# 数据处理
x_test = x_train = data.drop('label', axis=1).values
y_test = y_train = data['label'].values
x_train = Normalization.fit_transform(x_train)
#x_train = minmax.fit_transform(x_train)

x_test = x_train = torch.FloatTensor(x_train)
y_test = y_train = torch.LongTensor(y_train)

data_dim = (x_train.size(1))

cuda = True if torch.cuda.is_available() else False


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(data_dim)), 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        flat = x.view(x.size(0), -1)
        out = self.model(flat)

        return out


# Loss function
adversarial_loss = torch.nn.CrossEntropyLoss()
discriminator = Discriminator()


# 可视化数据分布
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_test)
plt.xticks([])
plt.yticks([])
plt.show()

# Optimizers
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=0.01, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
t = 0
t0 = time.time_ns()
for epoch in range(opt.n_epochs):

    # Adversarial ground truths
    # valid = Variable(Tensor(x_train.size(0)).fill_(1.0),
    # requires_grad=False)
    #fake = Variable(Tensor(x_train.size(0)).fill_(0.0), requires_grad=False)
    valid = torch.LongTensor(x_train.size(0)).fill_(1.0)
    fake = torch.LongTensor(x_train.size(0)).fill_(0.0)
    # Configure input

    batch_size = min(opt.batch_size, x_train.shape[0])
    num_batches = int(x_train.shape[0] / batch_size)
    # k= 数据维度
    k = data_dim

    for index in range(num_batches):

        a = round(min(np.std(x_train.numpy(), axis=0)), 4)
        x_max = int(round(x_train.numpy().max().max()))
        x_min = int(round(x_train.numpy().min().min()))

        # Sample noise as generator input.
        z = Variable(Tensor(np.random.normal(
            0, 5, (batch_size, data_dim))))

        #noise = Tensor(np.random.uniform(-10, 10, size=(int(batch_size), 2)))
        # plt.scatter(noise[:,0],noise[:,1])
        # plt.show()

        # Generate a batch of Potential outliers
        f = fake[index * batch_size: (index + 1) * batch_size]

        x = x_train[index * batch_size: (index + 1) * batch_size]
        y = valid[index * batch_size: (index + 1) * batch_size]

        noise = []
        # k 最大值设置为10
        k = min(k,10)
        c= [-1,1]
        for i in range(k-1):
            noise.append(Tensor(np.random.uniform(x_min, x_max,
                                                  (int(batch_size), data_dim))))
            n = x + Tensor(np.random.uniform(low=-a, high=a,
                           size=(int(batch_size), data_dim)))
            noise.append(n)

        noise = torch.cat(noise, dim=0)
        f = torch.LongTensor(noise.size(0)).fill_(0.0)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples   判别器的loss
        real_loss = adversarial_loss(discriminator(x), y)
        fake_loss = adversarial_loss(discriminator(noise.detach()), f)
        d_loss = (  real_loss + (k- 3/2) *  fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    if epoch % opt.sample_interval == 0:
        plt.scatter(noise.detach()[:, 0], noise.detach()[:, 1])
        #可视化生成的潜在离群点
        #plt.scatter(n.detach()[:, 0], n.detach()[:, 1])
        plt.show()
    test_acc_num = 0

    if epoch % opt.sample_interval == 0:
        Y_hat = discriminator(x_test)
        outputs = torch.softmax(Y_hat, dim=1)
        #outputs = torch.sigmoid(Y_hat)
        score = outputs[:, 1].detach().numpy()
        predict = torch.max(Y_hat, 1)[1]
        # 计算acc
        test_acc_num += torch.eq(predict, y_test).sum().item()
        a = y_test.shape[0]
        train_acc = test_acc_num / a
        # 计算auc
        
        AUC = metrics.roc_auc_score(y_test, score)
        # 这里设置把0作为正常值  把1作为离群点
        c_color = []
        for i in y_test:
            if i < 0.5:
                c_color.append(1)
            else:
                c_color.append(0)
        
        y_pred = []
        for i in predict:
            if i < 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
        c_color = torch.LongTensor(c_color)
        y_pred = torch.LongTensor(y_pred)

        f1_score = metrics.f1_score(c_color, y_pred)
        Recall = metrics.recall_score(c_color, y_pred)
        precision = metrics.precision_score(c_color, y_pred)
        print("step=", epoch, 'acc=', train_acc,
              'AUC=', AUC, 'f1 = ', f1_score,'recall =',Recall,'precision=',precision)
        Y_color = []
        for i in Y_hat.detach().numpy():
            Y_color.append(0 if i[0] > i[1] else 1)
        plt.scatter(x_test[:, 0], x_test[:, 1], c=predict)
        plt.xticks([])
        plt.yticks([])
        plt.show()

        t1 = time.time_ns()
        t += t1-t0
        print(
            "[Epoch %d/%d]  [D loss: %f]  [Time :%d]"
            % (epoch, opt.n_epochs,  d_loss.item(),t/1000000)
        )
