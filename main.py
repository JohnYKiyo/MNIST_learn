import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import cnn
import dataloader
import argparse

import matplotlib.pyplot as plt
import numpy as np

import os
import csv
from tqdm import tqdm


def train(epoch, model, trainloader):
    print(f'Train Epoch:{epoch}')
    model.train()
    
    batch_train_loss = []
    with tqdm(total=len(trainloader)) as pbar:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = CELoss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            batch_train_loss.append(loss.item())
            pbar.update(1)
            
    return np.mean(batch_train_loss), batch_train_loss


def test(epoch, model, testloader):
    global best_acc
    model.eval()
    batch_test_loss = []
    batch_correct = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = CELoss(outputs, targets)
            batch_test_loss.append(loss.item())
            
            _, predicted = torch.max(outputs, axis=1)
            batch_correct.append(predicted.eq(targets).cpu().numpy())
    
    batch_correct = np.array(batch_correct)
    correct = batch_correct.sum()
    total = batch_correct.size
    
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    
    return np.array(batch_test_loss).mean(), batch_test_loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-E', default=10)
    parser.add_argument('--lr', '-L', default=0.1)
    parser.add_argument('--gamma', '-G', default=0.95)
    args = parser.parse_args()
    print(args)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = cnn.model(num_blocks=[1,1,1], num_channels=1,num_classes=10)
    model = model.to(device)
    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    best_acc = 0

    train_loss_list = []
    test_loss_list = []
    acc_list = []
    N=int(args.epoch)
    with tqdm(total=N) as pbar:
        for epoch in range(N):
            _,train_loss = train(epoch, model, dataloader.trainloader)
            _,test_loss, acc = test(epoch, model, dataloader.testloader)
            scheduler.step()
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            acc_list.append(acc)
            pbar.set_postfix_str(s=f'acc:{acc}')
            pbar.update()


    with open('train_log.txt', 'w') as f:
        writer = csv.writer(f)
        for l in train_loss_list:
            writer.writerow(l)

    with open('test_log.txt', 'w') as f:
        writer = csv.writer(f)
        for l in test_loss_list:
            writer.writerow(l)

    with open('acc_log.txt', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(acc_list)

