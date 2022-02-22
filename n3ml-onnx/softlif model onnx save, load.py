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


def load_onnx_to_softlif(model_path, opt=None):
    onnx_model = onnx.load_model(model_path)
    graph = onnx_model.graph

    nodes = [i for i in graph.node]
    weights = []
    for init in graph.initializer:
        weight = numpy_helper.to_array(init)

        if len(weight) == 2:
            weight = weight.T
        else:
            weight = numpy_helper.to_array(init)

        weights.append(weight)

    weights = sorted(weights, key=lambda x: len(x.shape), reverse=True)

    model = Hunsberger2015()

    for node in nodes:
        name = node.name
        op = node.op_type.lower()

        if op == "matmul":  # dense without bias
            input_unit, output_unit = weights[0].shape[0], weights[0].shape[1]
            model.add_module(name, nn.Linear(input_unit, output_unit, bias=False))
            del weights[0]

        elif op == "relu":
            if opt is not None:
                model.add_module(name, SoftLIF(amplitude=opt.amplitude, tau_ref=opt.tau_ref, tau_rc=opt.tau_rc, gain=opt.gain, sigma=opt.sigma))
            else:
                model.add_module(name, SoftLIF())

        elif op == "flatten":
            model.add_module(name, nn.Flatten())

        elif op == "conv":
            input_channels = weights[0].shape[1]
            output_channels = weights[0].shape[0]

            kernel_size = 3
            stride = 1
            padding = 0

            for att in node.attribute:
                if att.name == "kernel_shape":
                    kernel_size = att.ints[0]
                elif att.name == "strides":
                    stride = att.ints[0]
                elif att.name == "pads":
                    padding = att.ints[0]

            model.add_module(name, nn.Conv2d(input_channels,
                                             output_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             bias=False))

            del weights[0]

        elif op == "averagepool":
            kernel_size = 2

            for att in node.attribute:
                if att.name == "kernel_shape":
                    kernel_size = att.ints[0]

            model.add_module(name, nn.AvgPool2d(kernel_size=kernel_size))

        elif op == "pad":
            pass

        else:
            pass

    # generate model end

    # load model weight insert
    initializers = []
    for init in graph.initializer:
        initializers.append(numpy_helper.to_array(init))

    model_layer_info = [i for i in model.named_parameters()]

    initializers = sorted(initializers, key=lambda x: len(x.shape), reverse=True)

    for i in range(len(model_layer_info)):
        layer_name = model_layer_info[i][0]

        if "matmul" in layer_name.lower():
            weight = torch.from_numpy(initializers[i]).T
        else:
            weight = torch.from_numpy(initializers[i])

        model.state_dict()[layer_name].data.copy_(weight)
    # load model weight end

    print("load {}".format(model_path))
    return model


def save_softlif_to_onnx(model, model_name):
    model_layer_info = [i for i in model.named_children()]
    model_weight = [i for i in model.named_parameters()]

    model = Hunsberger2015()

    # softLIF -> RELU
    for i in range(len(model_layer_info)):
        if "LIF" in str(model_layer_info[i][1]):
            model.add_module(model_layer_info[i][0], nn.ReLU())
        else:
            model.add_module(model_layer_info[i][0], model_layer_info[i][1])

    # weight recovery
    for i in model_weight:
        model.state_dict()[i[0]].data.copy_(i[1])

    dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32).cuda()
    torch.onnx.export(model, dummy_input, model_name)
    print("saved {}".format(model_name))

    # layer name change
    onnx_model = onnx.load(model_name)

    for i in range(len(onnx_model.graph.node)):
        if "relu" in onnx_model.graph.node[i].name.lower():
            onnx_model.graph.node[i].name += "_softlif"

    onnx.save(onnx_model, model_name)
    # layer name change end


def app(opt):
    print(opt)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([transforms.ToTensor()])),
        batch_size=opt.batch_size)

    model = load_onnx_to_softlif("result/softlif_conv.onnx", opt)
    model.cuda()

    criterion = nn.CrossEntropyLoss()

    val_acc, val_loss = validate(val_loader, model, criterion)
    print('in test, val accuracy: {} - loss: {}'.format(val_acc, val_loss))

    save_softlif_to_onnx(model, "result/softlif_conv.onnx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=4e-2, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--pretrained', default='pretrained/softlif_dynamic.pt')

    parser.add_argument('--amplitude', default=0.063, type=float)
    parser.add_argument('--tau_ref', default=0.001, type=float)
    parser.add_argument('--tau_rc', default=0.05, type=float)
    parser.add_argument('--gain', default=0.825, type=float)
    parser.add_argument('--sigma', default=0.02, type=float)

    app(parser.parse_args())