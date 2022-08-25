import os
import logging
import torch
import ipdb
import time
from decimal import Decimal
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torchvision
from contextlib import suppress
from collections import OrderedDict
from torch.autograd import Variable
from . import Metric, classification_accuracy
from timm.models import model_parameters
from timm.utils import *

from timm.data import Mixup

class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model, train_loader, val_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader   = val_loader

        if args.dataset.startswith('Aff'):
            # cw = 1 / class_counts
            # cw /= cw.min()
            # class_weights = cw
            class_counts = np.array([74874, 134415, 25459, 14090, 6378, 3803, 24882, 3750])
            class_counts = class_counts[:args.num_classes]
            class_weights = (np.sum(class_counts) - class_counts) / class_counts
            train_weights = torch.FloatTensor(class_weights).cuda()
            self.criterion = nn.CrossEntropyLoss(weight=train_weights).cuda()
        # elif args.dataset.startswith('RAF'):
        #     class_counts = np.array([2524, 4772, 1982, 1290, 281, 717, 705])
        #     class_weights = (np.sum(class_counts) - class_counts) / class_counts
        #     train_weights = torch.FloatTensor(class_weights).cuda()
        #     self.criterion = nn.CrossEntropyLoss(weight=train_weights).cuda()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.val_criterion = nn.CrossEntropyLoss()


    def train(self, optimizer, epoch_idx, scheduler):
        # Set model to training mode
        self.model.train()

        train_loss     = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')

        num_updates = epoch_idx * len(self.train_loader)
        with tqdm(total=len(self.train_loader),
                  desc='Train Ep. #{}: '.format(epoch_idx + 1),
                  disable=False,
                  ncols=110,
                  ascii=True) as t:
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()
                # Do forward-backward.
                num =  data.size(0)
                num_updates += 1
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss.update(loss, num)
                train_accuracy.update(classification_accuracy(output, target), num)

                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                scheduler.step_update(num_updates=num_updates)

                t.set_postfix({'loss': train_loss.avg.item(),
                               'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                               'lr': '{:.6f}'.format(lr)})
                t.update(1)


        summary = {'loss': '{:.3f}'.format(train_loss.avg.item()),
                   'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                   'lr': '{:.4f}'.format(lr)
                   }
        # logging.info(('In train()-> Train Ep. #{} '.format(epoch_idx + 1)
        #               + ', '.join(['{}: {}'.format(k, v) for k, v in summary.items()])))
        return train_accuracy.avg.item()
    #{{{ Evaluate classification
    def validate(self, epoch_idx, biases=None):
        """Performs evaluation."""

        self.model.eval()
        val_loss = Metric('val_loss')
        val_accuracy = Metric('val_accuracy')

        with tqdm(total=len(self.val_loader),
                  desc='Val Ep. #{}: '.format(epoch_idx + 1),
                  ncols=100,
                  ascii=True) as t:
            with torch.no_grad():
                for data, target in self.val_loader:
                    if self.args.cuda:
                        data, target = data.cuda(), target.cuda()

                    num = torch.tensor(data.size(0)).cuda()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    val_loss.update(loss, num)
                    val_accuracy.update(classification_accuracy(output, target), num)

                    t.set_postfix({'loss': val_loss.avg.item(),
                                   'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                                   })
                    t.update(1)

        summary = {'loss': '{:.3f}'.format(val_loss.avg.item()),
                   'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item())
                   }

        # logging.info(('In validate()-> Val Ep. #{} '.format(epoch_idx + 1)
        #               + ', '.join(['{}: {}'.format(k, v) for k, v in summary.items()])))
        return val_accuracy.avg.item()


    def save_checkpoint(self, save_folder, arch, dataset, epoch_idx, avg_val_acc):
        """Saves model to file."""
        filepath = self.args.checkpoint_format.format(save_folder=save_folder, arch=arch, dataset=dataset, epoch=epoch_idx)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }
        torch.save(checkpoint, filepath)
        accpath = os.path.dirname(filepath) + '/epoch-{}_acc={}.txt'.format(epoch_idx, Decimal(avg_val_acc).quantize(
            Decimal("0.0000")))
        os.mknod(accpath)
        return

    def train_one_epoch(self,
            epoch, model, loader, optimizer, args,
            lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
            loss_scaler=None, model_ema=None, mixup_fn=None):

        if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
            if args.prefetcher and loader.mixup_enabled:
                loader.mixup_enabled = False
            elif mixup_fn is not None:
                mixup_fn.mixup_enabled = False

        second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()

        model.train()

        end = time.time()
        last_idx = len(loader) - 1
        num_updates = epoch * len(loader)


        for batch_idx, (input, target) in enumerate(loader):
                last_batch = batch_idx == last_idx
                data_time_m.update(time.time() - end)
                if not args.prefetcher:
                    input, target = input.cuda(), target.cuda()
                    if mixup_fn is not None:
                        input, target = mixup_fn(input, target)
                if args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                with amp_autocast():
                    output = model(input)
                    loss = self.criterion(output, target)

                if isinstance(output, (tuple, list)):
                    output = output[0]
                acc1, acc3 = accuracy(output, target, topk=(1, 3))

                if not args.distributed:
                    losses_m.update(loss.item(), input.size(0))

                optimizer.zero_grad()
                if loss_scaler is not None:
                    loss_scaler(
                        loss, optimizer,
                        clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                        parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                        create_graph=second_order)
                else:
                    loss.backward(create_graph=second_order)
                    if args.clip_grad is not None:
                        dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad, mode=args.clip_mode)
                    optimizer.step()

                if model_ema is not None:
                    model_ema.update(model)

                top1_m.update(acc1.item(), output.size(0))
                torch.cuda.synchronize()
                num_updates += 1
                batch_time_m.update(time.time() - end)
                if last_batch or batch_idx % args.log_interval == 0:
                    lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)

                    if args.distributed:
                        reduced_loss = reduce_tensor(loss.data, args.world_size)
                        losses_m.update(reduced_loss.item(), input.size(0))

                    if args.local_rank == 0:
                        logging.info(
                            'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                            'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                            '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                            'LR: {lr:.3e}  '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                                epoch,
                                batch_idx, len(loader),
                                100. * batch_idx / last_idx,
                                loss=losses_m,
                                batch_time=batch_time_m,
                                rate=input.size(0) * args.world_size / batch_time_m.val,
                                rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                                lr=lr,
                                data_time=data_time_m))

                        if args.save_images and output_dir:
                            torchvision.utils.save_image(
                                input,
                                os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                                padding=0,
                                normalize=True)


                if saver is not None and args.recovery_interval and (
                        last_batch or (batch_idx + 1) % args.recovery_interval == 0):
                    saver.save_recovery(epoch, batch_idx=batch_idx)

                if lr_scheduler is not None:
                    lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

                end = time.time()
                # end for

        if hasattr(optimizer, 'sync_lookahead'):
            optimizer.sync_lookahead()

        return OrderedDict([('loss', losses_m.avg)])

    def validate_one_epoch(self, epoch, model, loader, args, amp_autocast=suppress, log_suffix=''):
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        top3_m = AverageMeter()

        model.eval()

        end = time.time()
        last_idx = len(loader) - 1

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                last_batch = batch_idx == last_idx
                if not args.prefetcher:
                    input = input.cuda()
                    target = target.cuda()
                if args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                with amp_autocast():
                    output = model(input)

                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]
                loss = self.val_criterion(output, target)
                acc1, acc3 = accuracy(output, target, topk=(1, 3))

                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data, args.world_size)
                    acc1 = reduce_tensor(acc1, args.world_size)
                    acc3 = reduce_tensor(acc3, args.world_size)
                else:
                    reduced_loss = loss.data

                torch.cuda.synchronize()

                losses_m.update(reduced_loss.item(), input.size(0))
                top1_m.update(acc1.item(), output.size(0))
                top3_m.update(acc3.item(), output.size(0))

                batch_time_m.update(time.time() - end)
                end = time.time()
                if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                    log_name = 'Test' + log_suffix
                    logging.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@3: {top3.val:>7.4f} ({top3.avg:>7.4f})'.format(
                            log_name, batch_idx, last_idx, batch_time=batch_time_m,
                            loss=losses_m, top1=top1_m, top3=top3_m))


        metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top3', top3_m.avg)])

        return metrics