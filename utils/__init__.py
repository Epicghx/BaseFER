import sys, os
import torch
import ipdb
import torch.optim as optim
import logging

def make_optimizer(model, opt, lr, weight_decay, momentum):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    parameters = [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

    weight_decay = 0.

    opt_args = dict(weight_decay=weight_decay, lr=lr)

    if opt.lower() == 'sgd':
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt.lower() == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt.lower() == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)

    return optimizer

class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

# class Optimizers(object):
#     def __init__(self):
#         self.optimizers = []
#         self.lrs = []
#
#     def add(self, optimizer, lr):
#         self.optimizers.append(optimizer)
#         self.lrs.append(lr)
#
#     def step(self):
#         for optimizer in self.optimizers:
#             optimizer.step()
#
#     def zero_grad(self):
#         for optimizer in self.optimizers:
#             optimizer.zero_grad()
#
#     def __getitem__(self, index):
#         return self.optimizers[index]
#
#     def __setitem__(self, index, value):
#         self.optimizers[index] = value


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
        return

    def update(self, val, num):
        self.sum = self.sum + val * num
        self.n = self.n + num

    @property
    def avg(self):
        return self.sum / self.n


def classification_accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def set_dataset_paths(args):
    """Set default train and test path if not provided as input."""

    if not args.train_path:
        args.train_path = 'data/%s/train' % (args.dataset)

    if not args.val_path:
        if (args.dataset in ['imagenet', 'face_verification', 'emotion', 'emotion3', 'gender'] or
            args.dataset[:3] == 'age'):
            args.val_path = 'data/%s/val' % (args.dataset)
        else:
            args.val_path = 'data/%s/test' % (args.dataset)

def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir

def set_logger(filepath):
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    _format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return
