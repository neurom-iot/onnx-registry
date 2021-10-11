import numpy as np
import time
import argparse
import torch
import torchvision
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import n3ml.model
import onnx
from n3ml.layer import _Wu
from tqdm import tqdm
np.random.seed(0)
torch.manual_seed(0)


def validate(val_loader, model, criterion):

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(val_loader):
        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)
        labels_ = torch.zeros(torch.numel(labels), 10).cuda()
        labels_ = labels_.scatter_(1, labels.view(-1, 1), 1)

        loss = criterion(preds, labels_)

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss += loss.cpu().detach().numpy() * images.size(0)
        total_images += images.size(0)

    val_acc = num_corrects.float() / total_images
    val_loss = total_loss / total_images

    return val_acc, val_loss


def train(train_loader, model, criterion, optimizer):

    total_images = 0
    num_corrects = 0
    total_loss = 0

    list_loss = []
    list_acc = []

    for step, (images, labels) in enumerate(train_loader):

        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)

        labels_ = torch.zeros(torch.numel(labels), 10).cuda()
        labels_ = labels_.scatter_(1, labels.view(-1, 1), 1)

        loss = criterion(preds, labels_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss   += loss.cpu().detach().numpy() * images.size(0)
        total_images += images.size(0)

        if total_images > 0:  #  and total_images % 30 == 0
            list_loss.append(total_loss / total_images)
            list_acc.append(float(num_corrects) / total_images)


    train_acc = num_corrects.float() / total_images
    train_loss = total_loss / total_images

    return train_acc, train_loss


def app(opt):
    print(opt)

    # Load MNIST / FashionMNIST dataset
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=True,
            download = True,
            transform=torchvision.transforms.Compose([
                transforms.ToTensor()])),  # , transforms.Lambda(lambda x: x * 32)
        drop_last=True,
        batch_size=opt.batch_size,
        shuffle=True)

    # Load MNIST/ FashionMNIST dataset
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                transforms.ToTensor(), transforms.Lambda(lambda x: x * 32)])),
        drop_last=True,
        batch_size=opt.batch_size,
        shuffle=True)


    model = n3ml.model.Wu2018(batch_size=opt.batch_size, time_interval=opt.time_interval).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

    for epoch in tqdm(range(opt.num_epochs)):
        start = time.time()
        train_acc, train_loss = train(train_loader, model, criterion, optimizer)
        end = time.time()
        print('total time: {:.2f}s - epoch: {} - accuracy: {} - loss: {}'.format(end-start, epoch, train_acc, train_loss))

        lr_scheduler.step()


    print("train finish")
    print("validation start")

    model.eval()
    val_acc, val_loss = validate(val_loader, model, criterion)
    print("val_acc :",val_acc)
    print("val_loss :", val_loss)
    print("validation finish")

    from types import MethodType

    def forward(self, input):
        x = self.conv1(input)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(-1, 7 * 7 * 32)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    model.forward = MethodType(forward, model)

    dummy_input = torch.randn(opt.batch_size, 1,28, 28, dtype=torch.float32, device="cuda")
    torch.onnx.export(model, dummy_input, "stbp.onnx")

    val_acc, val_loss = validate(val_loader, model, criterion)
    print("after change", val_acc)
    print("after change", val_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',          default='data')
    parser.add_argument('--num_classes',   default=10,    type=int)
    parser.add_argument('--num_epochs',    default=2,   type=int)
    parser.add_argument('--batch_size',    default=1024,    type=int)
    parser.add_argument('--num_workers',   default=-1,     type=int)
    parser.add_argument('--time_interval', default=1,     type=int)
    parser.add_argument('--lr',            default=1e-03, type=float)

    app(parser.parse_args())
