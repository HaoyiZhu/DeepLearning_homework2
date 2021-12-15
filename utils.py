#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------
# Copyright (c) Haoyi Zhu. All rights reserved.
# Authored by Haoyi Zhu (hyizhu1108@gmail.com)
# -----------------------------------------------------

import paddle

def build_model(model_type, num_classes=10):
    if model_type == 'resnet18':
        from models.resnet import ResNet18
        return ResNet18(num_classes=num_classes)
    elif model_type == 'resnet34':
        from models.resnet import ResNet34
        return ResNet34(num_classes=num_classes)
    elif model_type == 'resnet50':
        from models.resnet import ResNet50
        return ResNet50(num_classes=num_classes)
    elif model_type == 'resnet101':
        from models.resnet import ResNet101
        return ResNet101(num_classes=num_classes)
    elif model_type == 'resnet152':
        from models.resnet import ResNet152
        return ResNet152(num_classes=num_classes)
    elif model_type == 'lenet':
        from models.lenet import LeNet
        return LeNet(num_classes=num_classes)
    elif model_type == 'vgg11':
        from models.vgg import VGG
        return VGG('VGG11', num_classes=num_classes)
    elif model_type == 'vgg13':
        from models.vgg import VGG
        return VGG('VGG13', num_classes=num_classes)
    elif model_type == 'vgg16':
        from models.vgg import VGG
        return VGG('VGG16', num_classes=num_classes)
    elif model_type == 'vgg19':
        from models.vgg import VGG
        return VGG('VGG19', num_classes=num_classes)
    else:
        raise NotImplementedError

def build_loss(cfg):
    if cfg.TYPE == 'CrossEntropy':
        # criterion = paddle.nn.CrossEntropyLoss(soft_label=True)
        from models.criterion import CELoss
        epsilon = cfg.get('EPSILON', None)
        criterion = CELoss(epsilon=epsilon, weighted=cfg.WEIGHTED)
    else:
        raise NotImplementedError

    return criterion

class DataLogger(object):
    """Average data logger."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt

def calc_accuracy(preds, labels):
    assert len(preds.shape) == 2, preds.shape
    
    preds = preds.cpu().numpy().argmax(-1)
    labels = labels.cpu().numpy().argmax(-1)

    acc = (preds == labels).mean()

    return acc