#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------
# Copyright (c) Haoyi Zhu. All rights reserved.
# Authored by Haoyi Zhu (hyizhu1108@gmail.com)
# -----------------------------------------------------

"""
Deep learning homework 2
"""

import numpy as np
import paddle
from tqdm import tqdm

from opt import cfg, logger, opt
from dataset.mnist import MNISTDataset
from utils import *

def train(train_loader, m, criterion, optimizer):
    loss_logger = DataLogger()
    acc_logger = DataLogger()

    m.train()
    train_loader = tqdm(train_loader, dynamic_ncols=True)

    for i, (inps, labels) in enumerate(train_loader):
        batch_size = inps.shape[0]

        output = m(inps)
        loss = criterion(output, labels)
        acc = calc_accuracy(output, labels)

        loss_logger.update(loss.item(), batch_size)
        acc_logger.update(acc, batch_size)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        train_loader.set_description(
            'loss: {loss:.8f} | acc: {acc:.4f}'.format(
                loss=loss_logger.avg,
                acc=acc_logger.avg)
        )

    train_loader.close()

    return loss_logger.avg, acc_logger.avg

def test(m, opt, batch_size=200, ssp=False):
    test_dataset = MNISTDataset(train=False, ssp=ssp)
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                        num_workers=opt.nThreads, drop_last=False)

    true_num = 0
    total_num = 0

    m.eval()
    for imgs, labels in tqdm(test_loader, dynamic_ncols=True):
        output = m(imgs).cpu().numpy().argmax(-1)
        labels = labels.cpu().numpy().argmax(-1)
        true_num += (output == labels).sum()
        total_num += output.shape[0]

    acc = true_num / total_num

    return acc

def ssp(cfg, main_model):
    logger.info('=================================================================')
    logger.info('------------------ Self-supervised pretraining ------------------')
    logger.info('=================================================================')
    
    m = preset_model(cfg, num_classes=4)
    m = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(m)

    criterion = paddle.nn.CrossEntropyLoss(soft_label=True)

    if cfg.SSP.LR_SCHEDULER.TYPE == 'multistep':
        lr_scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=cfg.SSP.LR, milestones=cfg.SSP.LR_SCHEDULER.LR_STEP,
                                                   gamma=cfg.SSP.LR_SCHEDULER.LR_FACTOR)
    elif cfg.SSP.LR_SCHEDULER.TYPE == 'cosineannealing':
        lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=cfg.SSP.LR, T_max=cfg.SSP.LR_SCHEDULER.T_MAX)
    else:
        raise NotImplementedError    

    if cfg.SSP.OPTIMIZER == 'adam':
        optimizer = paddle.optimizer.Adam(learning_rate=lr_scheduler, parameters=m.parameters())
    else:
        raise NotImplementedError

    train_dataset = MNISTDataset(train=True, reduce_dataset=True, ssp=True)
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=cfg.SSP.BATCH_SIZE, shuffle=True,
                                        num_workers=opt.nThreads)

    for i in range(cfg.SSP.BEGIN_EPOCH, cfg.SSP.END_EPOCH):
        current_lr = optimizer.get_lr()

        logger.info(f'############# (SSP) Starting Epoch {i} | LR: {current_lr} #############')

        # Training
        loss, acc = train(train_loader, m, criterion, optimizer)
        logger.epochInfo('Train', i, loss, acc)

        lr_scheduler.step()

        if (i + 1) % opt.snapshot == 0:
            # Save checkpoint
            M = paddle.Model(m)
            M.save('./exp/{}-{}/ssp_model_{}'.format(opt.exp_id, cfg.FILE_NAME, i))
            with paddle.no_grad():
                test_acc = test(m, opt, ssp=True)
                logger.info(f'##### (SSP) Epoch {i} | test_acc: {test_acc} #####')

    M = paddle.Model(m)
    M.save('./exp/{}-{}/ssp_final'.format(opt.exp_id, cfg.FILE_NAME))

    M = paddle.Model(main_model)
    M.load('./exp/{}-{}/ssp_final'.format(opt.exp_id, cfg.FILE_NAME), skip_mismatch=True, reset_optimizer=True)

    logger.info('=================================================================')
    logger.info('--------------- SSP finished. Start main training. --------------')
    logger.info('=================================================================')

    return M.network


def main():
    logger.info('******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')

    # Model Initialize
    paddle.set_device(opt.device)
    m = preset_model(cfg)

    if opt.sync:
        m = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(m)

    if opt.ssp:
        m = ssp(cfg, m)

    criterion = build_loss(cfg.LOSS)

    if cfg.TRAIN.LR_SCHEDULER.TYPE == 'multistep':
        lr_scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=cfg.TRAIN.LR, milestones=cfg.TRAIN.LR_SCHEDULER.LR_STEP,
                                                   gamma=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR)
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'cosineannealing':
        lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=cfg.TRAIN.LR, T_max=cfg.TRAIN.LR_SCHEDULER.T_MAX)
    else:
        raise NotImplementedError    

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = paddle.optimizer.Adam(learning_rate=lr_scheduler, parameters=m.parameters())
    else:
        raise NotImplementedError

    train_dataset = MNISTDataset(train=True, reduce_dataset=opt.reduce_dataset)
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                        num_workers=opt.nThreads)

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i
        current_lr = optimizer.get_lr()

        logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')

        # Training
        loss, acc = train(train_loader, m, criterion, optimizer)
        logger.epochInfo('Train', opt.epoch, loss, acc)

        lr_scheduler.step()

        if (i + 1) % opt.snapshot == 0:
            # Save checkpoint
            M = paddle.Model(m)
            M.save('./exp/{}-{}/model_{}'.format(opt.exp_id, cfg.FILE_NAME, opt.epoch))
            with paddle.no_grad():
                test_acc = test(m, opt)
                logger.info(f'##### Epoch {opt.epoch} | test_acc: {test_acc} #####')

    M = paddle.Model(m)
    M.save('./exp/{}-{}/final'.format(opt.exp_id, cfg.FILE_NAME))

def preset_model(cfg, num_classes=10):
    model = build_model(cfg.MODEL.TYPE, num_classes=num_classes)

    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        M = paddle.Model(model)
        M.load(cfg.MODEL.PRETRAINED, reset_optimizer=True)
        model = M.network
    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        M = paddle.Model(model)
        M.load(cfg.MODEL.TRY_LOAD, skip_mismatch=True, reset_optimizer=True)
        model = M.network

    return model


if __name__ == '__main__':
    main()
