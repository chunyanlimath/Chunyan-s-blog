

'''
This code is used to implement VAE, where encoder and decoder is constructed seperately and saved seperately
The input is image and will be convert into latent variables z, then, if you sample z from prior, then,
you could generate new image by feed sampled z into decoder.

Encoder is a Gaussian with diagonal covariance matrix and decoder is a Gaussian with Identity covariance matrix
reference-- code link:
https://github.com/pytorch/examples/blob/master/vae/main.py

Encoder and decoder are implemented separately so that one can easily use only one of them for next step. 
If you find any mistakes, you are welcome to point out, please email me using chunyanlimath@gmail.com!

date: 2/28/22
Author: Li
'''

# ----------- import packages ------------
import random
import argparse  # ?
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import torch.optim as optim
import itertools
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  # plt 用于显示图片
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.image as mpimg   # mpimg 用于读取图片
from torchvision.datasets import MNIST

# ---------- hyper-parameters ------------------------
batch_size = 128
image_size = 784
h_dim = 400
z_dim = 20
learning_rate = 1e-3
dataset_path = './datasets'
sample_num = 1  # number of z sampled used to estimate expectation

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()   #
print(args)

# Sets the seed for generating random numbers. And returns a torch._C.Generator object.
torch.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# ---------------------- load dataset ----------------------
train_loader = DataLoader(MNIST(dataset_path, train=True, download=True, transform=transforms.ToTensor()),
                          batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = DataLoader(MNIST(dataset_path, train=False, download=True, transform=transforms.ToTensor()),
                         batch_size=args.batch_size, shuffle=False, **kwargs)


# ----------- VAE model ----------------------------
# ---------- Encoder N(mu, diag_cov) -------------------
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)   # mu
        self.fc22 = nn.Linear(h_dim, z_dim)   # log_var

        self.relu = nn.PReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        mu_encoder = self.fc21(h1)
        log_var_encoder = self.fc22(h1)
        return mu_encoder, log_var_encoder   # mean and covariance of encoder


# ------------- Decoder N(mu, I)-------------
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, image_size)   # mu

        self.relu = nn.PReLU()
       # self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h1 = self.relu(self.fc1(z))
        mu_decoder = torch.sigmoid(self.fc21(h1))
        return mu_decoder    # mean and covariance of decoder


encoder = Encoder()
decoder = Decoder()
optimizer = optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=learning_rate)


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)  # sample a point from normal distribution with same size of std.
    z = mu + eps * std
    return z


best_loss = 100000000000
train_loss_vector = []
test_loss_vector = []
for epoch in range(1, args.epochs + 1):
    encoder.train()
    decoder.train()
    train_loss = 0
    # --------- start training -----------------
    for batch_idx, (data, _) in enumerate(train_loader):  #
        data = data.view(-1, image_size)   # convert a 28*28 matrix into a long vector
        optimizer.zero_grad()
        mu_encoder, log_var_encoder = encoder(data)
        KLD = -0.5 * torch.sum(1 + log_var_encoder - mu_encoder.pow(2) - log_var_encoder.exp())
        # ---------- sample sample_num z to estimate expectation <the first term in ELBO> --------------------
        reconstruction_loss = 0
        for i in range(sample_num):
            z = reparametrize(mu_encoder, log_var_encoder)
            mu_decoder = decoder(z)
            reconstruction_loss += 0.5 * torch.sum((data-mu_decoder)*(data-mu_decoder))  # estimator of expectation * sample_num
        loss = reconstruction_loss/sample_num + KLD  # negative ELBO
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),   # len(data) = total sample/batch_size
                       100. * batch_idx / len(train_loader),    # len(train_loader.dataset)=total train samples
                       loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    train_loss_vector.append(train_loss/len(train_loader.dataset))
    # --------- start validation ---------------
    encoder.eval()
    decoder.eval()
    test_loss = 0
    with torch.no_grad():
        for (data1, _) in test_loader:
            data1 = data1.view(-1, image_size)   #
            mu_encoder, log_var_encoder = encoder(data1)
            KLD = -0.5 * torch.sum(1 + log_var_encoder - mu_encoder.pow(2) - log_var_encoder.exp())
            # --------- sample sample_num z to estimate expectation --------------
            reconstruction_loss = 0
            for i in range(sample_num):
                z = reparametrize(mu_encoder, log_var_encoder)
                mu_decoder = decoder(z)
                reconstruction_loss += 0.5 * torch.sum((data1 - mu_decoder)*(data1 - mu_decoder))
            test_loss += reconstruction_loss / sample_num + KLD
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        test_loss_vector.append(test_loss)

    # --------- save the loss vector --------------
    np.savetxt(f'training loss of VAE.txt', train_loss_vector, delimiter='\n')
    np.savetxt(f'test loss of VAE.txt', test_loss_vector, delimiter='\n')

    if test_loss < best_loss:
        best_loss = test_loss
        best_epoch = epoch
        save_path_en = f'./best_encoder.pth'
        save_path_de = f'./best_decoder.pth'
        torch.save(encoder.state_dict(), save_path_en)
        torch.save(decoder.state_dict(), save_path_de)

# --------------- load the saved model and show to final results -------------
encoder.load_state_dict(
        torch.load(f'./best_encoder.pth', map_location=lambda storage, loc: storage))  # 加载模型并将参数赋值给刚创建的模型
encoder = encoder.eval()
decoder.load_state_dict(
        torch.load(f'./best_decoder.pth', map_location=lambda storage, loc: storage))  # 加载模型并将参数赋值给刚创建的模型
decoder = decoder.eval()
latent_vector = torch.zeros(1, z_dim)
test_loss = 0
with torch.no_grad():
    for batch_idx1, (data1, _) in enumerate(test_loader):
        data1 = data1.view(-1, image_size)
        mu_encoder, log_var_encoder = encoder(data1)
        z = reparametrize(mu_encoder, log_var_encoder)
        mu_decoder = decoder(z)
        latent_vector = torch.vstack((latent_vector, mu_encoder))  # used for record latent representation
        KLD = -0.5 * torch.sum(1 + log_var_encoder - mu_encoder.pow(2) - log_var_encoder.exp())
        # --------- sample sample_num z to estimate expectation --------------
        reconstruction_loss = 0
        for i in range(sample_num):
            z = reparametrize(mu_encoder, log_var_encoder)
            mu_decoder = decoder(z)
            reconstruction_loss += 0.5 * torch.sum((data1 - mu_decoder) * (data1 - mu_decoder))
            comparison = torch.cat([data1.view(-1, 1, 28, 28),   # torch.cat
                                    mu_decoder.view(-1, 1, 28, 28)], dim=3)
            save_image(comparison.cpu(),
                 'result+recons' + str(batch_idx1) + '.png')
        test_loss += reconstruction_loss / sample_num + KLD
    test_loss /= len(test_loader.dataset)
    print(f'final test loss is: {test_loss}')
    np.savetxt(f'latent of test.txt', latent_vector, delimiter='\n')
# ------- generate new pictures -----------
z = torch.randn(20, z_dim)
decoder = decoder.eval()
with torch.no_grad():
    mu_new = decoder(z)
    comparison = torch.cat([mu_new.view(-1, 1, 28, 28),
                            mu_new.view(-1, 1, 28, 28)], dim=3)
    save_image(comparison.cpu(),
               'new' + '.png')


 
