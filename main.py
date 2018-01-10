from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import itertools
import argparse

import util
import model

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='CycleGAN')

# Directory
parser.add_argument('--dataset_A', type=str, default='A')
parser.add_argument('--dataset_B', type=str, default='B')

# Data
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--resume', '-r', action='store_true')

# Network
parser.add_argument('--G_channel', type=int, default=32)
parser.add_argument('--D_channel', type=int, default=32)
parser.add_argument('--G_downsample', type=int, default=2)
parser.add_argument('--D_downsample', type=int, default=5)

parser.add_argument('--G_input', type=int, default=3)
parser.add_argument('--G_output', type=int, default=3)
parser.add_argument('--D_input', type=int, default=3)
parser.add_argument('--D_output', type=int, default=1)
parser.add_argument('--D_layer', type=int, default=5)

parser.add_argument('--G_block', type=int, default=6)
parser.add_argument('--G_block_type', type=str, default='conv')
parser.add_argument('--G_enable_se', type=bool, default=True)

# Training
parser.add_argument('--learning_rate', type=int, default=2e-4)
parser.add_argument('--lr_decay_epoch', type=int, default=100)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--log_frequency', type=int, default=25)
parser.add_argument('--save_frequency', type=int, default=20)

config = parser.parse_args()

use_cuda = torch.cuda.is_available()

if config.resume:
    print('-- Resuming From Checkpoint')
    assert os.path.isdir('checkpoint'), '-- Error: No checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/cyclegan.nn')
    G_A = checkpoint['G_A']
    G_B = checkpoint['G_B']
    D_A = checkpoint['D_A']
    D_B = checkpoint['D_B']
    start = checkpoint['epoch'] + 1
else:
    G_A = model.Generator(config)
    G_B = model.Generator(config)  
    D_A = model.Discriminator(config)
    D_B = model.Discriminator(config)
    start = 1

if use_cuda:
    G_A = G_A.cuda()
    G_B = G_B.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    cudnn.benchmark = True

util.print_network(G_A)
util.print_network(D_A)
G_A.train()
G_B.train()
D_A.train()
D_B.train()

MSE_Loss = torch.nn.MSELoss()
L1_Loss = torch.nn.L1Loss()

G_A_Optimizer = Adam(G_A.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
G_B_Optimizer = Adam(G_B.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
D_A_Optimizer = Adam(D_A.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
D_B_Optimizer = Adam(D_B.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))

a_loader = util.get_loader(config, config.dataset_A + '/train')
b_loader = util.get_loader(config, config.dataset_B + '/train')
a_test_loader = util.get_loader(config, config.dataset_A + '/test')
b_test_loader = util.get_loader(config, config.dataset_B + '/test')

a_real_fixed = Variable(iter(a_test_loader).next()[0], volatile=True)
b_real_fixed = Variable(iter(b_test_loader).next()[0], volatile=True)
if use_cuda:
    a_real_fixed = a_real_fixed.cuda()
    b_real_fixed = b_real_fixed.cuda()

a_fake_pool = util.ItemPool()
b_fake_pool = util.ItemPool()

def adjust_learning_rate(optimizer, epoch):
    lr_now = config.learning_rate
    if epoch > config.lr_decay_epoch:
            lr_now = lr_now - lr_now*(epoch - config.lr_decay_epoch)/config.lr_decay_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_now

def train(start, epoch):
    last_time = time.time()
    epoch_time = time.time()
    print('-- Current Epoch: %d'%epoch)

    adjust_learning_rate(G_A_Optimizer, epoch)
    adjust_learning_rate(G_B_Optimizer, epoch)
    adjust_learning_rate(D_A_Optimizer, epoch)
    adjust_learning_rate(D_B_Optimizer, epoch)

    for i, (a_real, b_real) in enumerate(itertools.izip(a_loader, b_loader)):
        # Train Generators
        a_real = Variable(a_real[0])
        b_real = Variable(b_real[0])
        if use_cuda:
            a_real = a_real.cuda()
            b_real = b_real.cuda()

        a_fake = G_A(b_real)
        b_fake = G_B(a_real)
        a_rec = G_A(b_fake)
        b_rec = G_B(a_fake)
        a_fake_result = D_A(a_fake)
        b_fake_result = D_B(b_fake)

        real_labels = Variable(torch.ones(a_fake_result.size()))
        if use_cuda:
            real_labels = real_labels.cuda()

        G_A_loss = MSE_Loss(a_fake_result, real_labels)
        G_B_loss = MSE_Loss(b_fake_result, real_labels)
        a_rec_loss = L1_Loss(a_rec, a_real)
        b_rec_loss = L1_Loss(b_rec, b_real)
        G_loss = G_A_loss + G_B_loss + a_rec_loss*10 + b_rec_loss*10

        G_A.zero_grad()
        G_B.zero_grad()
        G_loss.backward()
        G_A_Optimizer.step()
        G_B_Optimizer.step()

        # Train Discriminators
        a_fake = Variable(torch.Tensor(a_fake_pool([a_fake.cpu().data.numpy()])[0]))
        b_fake = Variable(torch.Tensor(b_fake_pool([b_fake.cpu().data.numpy()])[0]))
        if use_cuda:
            a_fake = a_fake.cuda()
            b_fake = b_fake.cuda()

        a_real_result = D_A(a_real)
        a_fake_result = D_A(a_fake)
        b_real_result = D_B(b_real)
        b_fake_result = D_B(b_fake)

        real_labels = Variable(torch.ones(a_real_result.size()))
        fake_labels = Variable(torch.zeros(a_fake_result.size()))
        if use_cuda:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()

        D_A_real_loss = MSE_Loss(a_real_result, real_labels)
        D_A_fake_loss = MSE_Loss(a_fake_result, fake_labels)
        D_B_real_loss = MSE_Loss(b_real_result, real_labels)
        D_B_fake_loss = MSE_Loss(b_fake_result, fake_labels)

        D_A_loss = D_A_fake_loss + D_A_real_loss
        D_B_loss = D_B_fake_loss + D_B_real_loss

        D_A.zero_grad()
        D_B.zero_grad()
        D_A_loss.backward()
        D_B_loss.backward()
        D_A_Optimizer.step()
        D_B_Optimizer.step()

        # Log
        if i % config.log_frequency == 0:
            speed = time.time() - last_time
            last_time = time.time()
            format_str = ('Step: %d; Loss: G-A: %.3f, D-A: %.3f, G-B: %.3f, D-B: %.3f; Speed: %.2f sec/step')
            print(format_str % (i, G_A_loss, D_A_loss, G_B_loss, D_B_loss, speed/config.log_frequency))

    # Save Data
    print('-- Saving parameters and sample images.')
    state = {'G_A': G_A, 'G_B': G_B, 'D_A': D_A, 'D_B': D_B, 'epoch': epoch}
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/cyclegan.nn')

    if epoch >= 10 and epoch % config.save_frequency == 0:
        # Test Images
        for i, (a_real_test, b_real_test) in enumerate(itertools.izip(a_test_loader, b_test_loader)):
            a_real_test = Variable(a_real_test[0])
            b_real_test = Variable(b_real_test[0])
            if use_cuda:
                a_real_test = a_real_test.cuda()
                b_real_test = b_real_test.cuda()
            
            a_fake_test = G_A(b_real_test)
            b_fake_test = G_B(a_real_test)
            a_rec_test = G_A(b_fake_test)
            b_rec_test = G_B(a_fake_test)

            test = torch.cat([a_real_test, b_fake_test, a_rec_test, b_real_test, a_fake_test, b_rec_test], dim=0)
            test = util.denorm(test).data
            if not os.path.isdir('result'):
                os.mkdir('result')
            save_image(test, 'result/test%d-epoch-%d.jpg' % (i, epoch))
    else:
        # Sample Image
        a_fake_fixed = G_A(b_real_fixed)
        b_fake_fixed = G_B(a_real_fixed)
        a_rec_fixed = G_A(b_fake_fixed)
        b_rec_fixed = G_B(a_fake_fixed)
        sample = torch.cat([a_real_fixed, b_fake_fixed, a_rec_fixed, b_real_fixed, a_fake_fixed, b_rec_fixed], dim=0)
        sample = util.denorm(sample).data
        if not os.path.isdir('result'):
            os.mkdir('result')
        save_image(sample, 'result/sample-epoch-%d.jpg' % (epoch))
        
    epoch_time = (time.time() - epoch_time)/60
    time_remain = (epoch_time * (config.max_epoch - epoch))/60
    print('-- Epoch %d completed. Epoch Time: %.2f min, Time Est: %.2f hour.' %(epoch, epoch_time, time_remain))

# Training Loop
print('-- Start Training')
for epoch in range(start, config.max_epoch):
    train(start, epoch)