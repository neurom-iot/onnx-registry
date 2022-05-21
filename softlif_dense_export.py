# n3ml2

import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from n3ml.network import Network
from n3ml.layer import SoftLIF
import torchvision.transforms as transforms
import onnx
import onnx.numpy_helper as numpy_helper


class Hunsberger2015(Network):
    def __int__(self):
        super(Hunsberger2015, self).__init__()

    def forward(self, x):
        for m in self.named_children():
            x = m[1](x)
        return x


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


def train(train_loader, model, criterion, optimizer):
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

    train_acc = num_corrects.float() / total_images
    train_loss = total_loss / total_images

    return train_acc, train_loss


def softlif_onnx_export(model, model_name, opt):
    # onnx export start
    model_layer_info = []
    model_weight = []
    # 모델 레이어 이름, 오브젝트 저장
    for i in model.named_children():
        model_layer_info.append(i)
    # 모델 가중치 저장
    for i in model.named_parameters():
        model_weight.append(i)

    # 모델 새로 생성
    model = Hunsberger2015()

    # 레이어를 다시 쌓으며 softlif를 제거하고 relu로 대체
    for i in range(len(model_layer_info)):
        if "LIF" in str(model_layer_info[i][1]):  # soft lif layer
            model.add_module(model_layer_info[i][0], nn.ReLU())
        else:
            model.add_module(model_layer_info[i][0], model_layer_info[i][1])

    # 가중치를 원래 모델
    for i in model_weight:
        model.state_dict()[i[0]].data.copy_(i[1])

    dummy_input = torch.randn(opt.batch_size, 1,28, 28, dtype=torch.float32).cuda()
    torch.onnx.export(model, dummy_input, model_name)

    # onnx export end
    print("onnx export")

    onnx_model = onnx.load(model_name)

    for i in range(len(onnx_model.graph.node)):
        if "relu" in onnx_model.graph.node[i].name.lower():
            onnx_model.graph.node[i].name += "_softlif"

    onnx.save(onnx_model, model_name)

def app(opt):
    print(opt)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([transforms.ToTensor()])),
        batch_size=opt.batch_size)

    model = Hunsberger2015()

    model.add_module('flatten', nn.Flatten())
    model.add_module('fc1', nn.Linear(784, 128, bias=False))
    model.add_module('slif1', SoftLIF(amplitude=opt.amplitude, tau_ref=opt.tau_ref, tau_rc=opt.tau_rc, gain=opt.gain, sigma=opt.sigma))
    model.add_module('fc2', nn.Linear(128, 64, bias=False))
    model.add_module('slif2', SoftLIF(amplitude=opt.amplitude, tau_ref=opt.tau_ref, tau_rc=opt.tau_rc, gain=opt.gain, sigma=opt.sigma))
    model.add_module('fc6', nn.Linear(64, opt.num_classes, bias=False))

    model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

    best_acc = 0

    for epoch in range(opt.num_epochs):
        start = time.time()
        train_acc, train_loss = train(train_loader, model, criterion, optimizer)
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

    softlif_onnx_export(model, opt.name, opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=4e-2, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--pretrained', default='pretrained/softlif_dynamic.pt')

    parser.add_argument('--amplitude', default=0.063, type=float)
    parser.add_argument('--tau_ref', default=0.001, type=float)
    parser.add_argument('--tau_rc', default=0.05, type=float)
    parser.add_argument('--gain', default=0.825, type=float)
    parser.add_argument('--sigma', default=0.02, type=float)
    parser.add_argument('--name', default='softlif_dense.onnx')

    app(parser.parse_args())