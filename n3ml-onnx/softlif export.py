import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from n3ml.model import Hunsberger2015


class Plot:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax2 = self.ax.twinx()
        plt.title('Soft LIF')

    def update(self, y1, y2):
        x = torch.arange(y1.shape[0]) * 64 * 100

        ax1 = self.ax
        ax2 = self.ax2

        ax1.plot(x, y1, 'g')
        ax2.plot(x, y2, 'b')

        ax1.set_xlabel('number of images')
        ax1.set_ylabel('accuracy', color='g')
        ax2.set_ylabel('loss', color='b')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def validate(val_loader, model, criterion):
    model.eval()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(val_loader):
        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)

        loss = criterion(preds, labels)

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss += loss.cpu().detach().numpy() * images.size(0)
        total_images += images.size(0)

    val_acc = num_corrects.float() / total_images
    val_loss = total_loss / total_images

    return val_acc, val_loss


def train(train_loader, model, criterion, optimizer, list_acc, list_loss, plotter):
    model.train()

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss += loss.cpu().detach().numpy() * images.size(0)
        total_images += images.size(0)

        if step > 0 and step % 100 == 0:
            list_loss.append(total_loss / total_images)
            list_acc.append(num_corrects.float() / total_images)

            plotter.update(y1=np.array(list_acc), y2=np.array(list_loss))

    train_acc = num_corrects.float() / total_images
    train_loss = total_loss / total_images

    return train_acc, train_loss


def app(opt):
    print(opt)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(24),
                torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(24),
                torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size,
        num_workers=opt.num_workers)

    model = Hunsberger2015(num_classes=opt.num_classes, amplitude=opt.amplitude, tau_ref=opt.tau_ref,
                           tau_rc=opt.tau_rc, gain=opt.gain, sigma=opt.sigma).cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

    # for plot
    plotter = Plot()

    list_loss = []
    list_acc = []

    best_acc = 0

    for epoch in range(opt.num_epochs):
        start = time.time()
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, list_acc, list_loss, plotter)
        end = time.time()
        print('total time: {:.2f}s - epoch: {} - accuracy: {} - loss: {}'.format(end-start, epoch, train_acc, train_loss))

        val_acc, val_loss = validate(val_loader, model, criterion)

        if val_acc > best_acc:
            best_acc = val_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()}
            torch.save(state, opt.pretrained)
            print('in test, epoch: {} - best accuracy: {} - loss: {}'.format(epoch, best_acc, val_loss))

        lr_scheduler.step()

    # training finish

    # method change

    temp_weights = []
    for i, param in enumerate(model.parameters()):
        print(param.data.shape)
        temp_weights.append(param.data)

    # change softlif -> relu
    model.extractor = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2)
    )
    model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256, 1024, bias=False),
        nn.ReLU(),
        nn.Linear(1024, opt.num_classes, bias=False)
    )

    # weight initialize
    model.extractor[0].weight.data = temp_weights[0]
    model.extractor[3].weight.data = temp_weights[1]
    model.extractor[6].weight.data = temp_weights[2]
    model.classifier[1].weight.data = temp_weights[3]
    model.classifier[3].weight.data = temp_weights[4]

    model.cuda()

    val_acc, val_loss = validate(val_loader, model, criterion)
    print("val_acc :", val_acc)
    print(val_loss)

    # crop by 24x24
    dummy_input = torch.randn(opt.batch_size, 3,24, 24, dtype=torch.float32).cuda()
    torch.onnx.export(model, dummy_input, "softlif.onnx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=4e-2, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--pretrained', default='pretrained/softlif.pt')

    parser.add_argument('--amplitude', default=0.063, type=float)
    parser.add_argument('--tau_ref', default=0.001, type=float)
    parser.add_argument('--tau_rc', default=0.05, type=float)
    parser.add_argument('--gain', default=0.825, type=float)
    parser.add_argument('--sigma', default=0.02, type=float)

    app(parser.parse_args())
