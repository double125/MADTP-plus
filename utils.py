import math

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    

def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):        
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
        
import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)        
        

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table
from torch import nn


def print_params_and_flops(print_type, model, device, config=None):

    model.eval()

    if print_type == 'nlvr':
        class Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, inputs):
                images, text, targets = inputs
                return self.model(images, text, targets=targets, train=False)
        with torch.no_grad():
            wrapper_model = Wrapper(model); 
            inputs = [torch.randn(2, 3, 384, 384).to(device) , 
                    ['Params and FLOPs test, test params and FLOPs, params and FLOPs test, test params and FLOPs, params and FLOPs test'] * 1, 
                    torch.randint(0, 2, (1,)).to(device) 
                    ]
            flop = FlopCountAnalysis(wrapper_model, inputs)
            print(flop_count_table(flop, max_depth=3, show_param_shapes=True))
            print("Total", flop.total() / 1e9)

    elif print_type == 'caption':
        class Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, inputs):
                images, caption = inputs
                return self.model(images, caption)
        with torch.no_grad():
            wrapper_model = Wrapper(model); 
            inputs = [torch.randn(1, 3, 384, 384).to(device) , 
                    ['a picture of car driving down a road behind a lot of sheep'] * 1
                    ]
            flop = FlopCountAnalysis(wrapper_model, inputs)
            print(flop_count_table(flop, max_depth=3, show_param_shapes=True))
            print("Total", flop.total() / 1e9)

    elif print_type == 'vqa':
        class Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, inputs):
                image, weights, question, answer, n = inputs
                return self.model(image, question, answer, train=True, n=n, weights=weights)
        with torch.no_grad():
            wrapper_model = Wrapper(model); 
            inputs = [torch.randn(1, 3, 480, 480).to(device) , 
                    torch.randn(47).to(device), 
                    ['where is the yellow pedestrian crossing?', 'how many people are in the photo?', 'where are boats?', 'what is in the window?', 
                    'what color is the road?', 'does the banana need the pillow?', 'do you need a chopstick?', 'what is the man sitting on?', 'who is wearing a white top?', 
                    'what is the man studying?', 'what color is the roof?', 'how many people are in the photo?', 'what is beyond the beach?', 'what is the woman doing to the pizza?', 
                    "what color is the man's shirt?", 'where is the man surfing?', 'what is the person with the knife doing?', 'when was the picture taken of the clock tower?',
                    'what is blue color?', 'how many people are there?', 'how many people in the picture?', 'who is to the right?', 'what country is shown on the placemat?', 
                    'how many people are in this picture?', 'where is the cat?', 'how many spoons are in the picture?', 'are the names on the scoreboard the names of people?', 
                    'what is the sign?', 'what are the people playing?', 'what is the shoreline?', 'what is on the police officers head?', 'who are eating on the table?'],
                    ['on sign', '1', 'on water', 'cat in window', 'gray', 'no', 'no', 'surfboard', 'tennis player', 'science project', 'attendance list', 'science', 
                        'school projects', 'paper', 'speech', 'no idea', 'gray and green', 'four', 'hills', 'taking picture', 'taking photo', 'photographing it', 'smiling', 
                        's', 'gray', 'in ocean', 'cutting cake', 'early morning', 'umbrella', 'one', 'three', 'man', 'italy', 'sicily', 'italia', 'australia', 'zero', 
                        'on dashboard', '1', 'yes', 'newport beach sacramento', 'no', 'road closed', 'nintendo wii', 'rocks', 'hats', 'no one'],
                    [1]
                    ]
            flop = FlopCountAnalysis(wrapper_model, inputs)
            print(flop_count_table(flop, max_depth=3, show_param_shapes=True))
            print("Total", flop.total() / 1e9)
            
    elif print_type == 'retrieval':
        class Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, inputs):
                image, caption, alpha, idx = inputs
                return self.model(image, caption, alpha=alpha, idx=idx)
        with torch.no_grad():
            wrapper_model = Wrapper(model); 
            inputs = [torch.randn(1, 3, 384, 384).to(device) , 
                    ['car driving down a road behind a lot of sheep'] * 1, 
                    0.0,
                    torch.randint(1000, 10000, (1,)).to(device) 
                    ]
            flop = FlopCountAnalysis(wrapper_model, inputs)
            print(flop_count_table(flop, max_depth=3, show_param_shapes=True))
            print("Total", flop.total() / 1e9)
    
    elif print_type == 'retrieval_clip':
        class Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, inputs):
                image, text, alpha, idx = inputs
                return self.model(image, text, alpha, idx)
        with torch.no_grad():
            wrapper_model = Wrapper(model); 
            inputs = [torch.randn(1, 3, config['image_size'], config['image_size']).to(device) , 
                    ["car driving down a road behind a lot of sheep"], 
                    0.0,
                    torch.full((1, ),-100).to(device) 
                    ]
            flop = FlopCountAnalysis(wrapper_model, inputs)
            print(flop_count_table(flop, max_depth=3, show_param_shapes=True))
            print("Total", flop.total() / 1e9)
        model.reset_queue()

    model.train()

