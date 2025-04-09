import os
import torch
import torch.optim as optim

# dataloader
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
# utils
from utils.logger import *
from utils.misc import *
from utils.config import *

from datasets.PartUSSPADataset import PartUSSPA

def dataset_builder(args, config, split):
    
    if args.evaluation:
        config.dataset.split = split
        dataset = PartUSSPA(config.dataset)
            
        sampler = None
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle = False, 
            drop_last = False,
            num_workers = int(args.num_workers),
            worker_init_fn=worker_init_fn)
    
    else:
        config.dataset.split = split
        dataset = build_dataset_from_cfg(config.dataset)
        shuffle = split == 'train'
      
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size = args.bs if shuffle else 1,
                num_workers = int(args.num_workers),
                drop_last = split == 'train',
                worker_init_fn = worker_init_fn,
                sampler = sampler)
        else:
            sampler = None
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.bs if shuffle else 1,
                shuffle = shuffle, 
                drop_last = split == 'train',
                num_workers = int(args.num_workers),
                worker_init_fn=worker_init_fn)
    return sampler, dataloader

def model_builder(config):
    model = build_model_from_cfg(config)
    return model

def build_optimizer(base_model, config):
    
    generator, refine = [], []
    for name, params in base_model.module.generator.named_parameters():
        if 'lin' in name:
            refine.append(params)
        else:
            generator.append(params)
    
    G_optimizer = optim.AdamW(
        generator,
        lr=config.optimizer.kwargs.G_lr,
        weight_decay=config.optimizer.kwargs.G_weight_decay
        )
    
    if config.model.use_disc:
        D_optimizer = optim.AdamW(
            base_model.module.discriminator.parameters(),
            lr=config.optimizer.kwargs.D_lr,
            weight_decay=config.optimizer.kwargs.D_weight_decay
            )
    else:
        D_optimizer = None
        
    if config.model.use_ref:
        R_optimizer = optim.AdamW(
            refine,
            lr=config.optimizer.kwargs.D_lr,
            weight_decay=config.optimizer.kwargs.D_weight_decay
            )
    else:
        R_optimizer = None
    
    return (G_optimizer, D_optimizer, R_optimizer)
    
def build_scheduler(base_model, optimizer, config, last_epoch=-1):
    
    G_optimizer, D_optimizer, R_optimizer = optimizer
    
    sche_config = config.scheduler
    G_scheduler = build_lambda_sche(G_optimizer, sche_config.kwargs, last_epoch=last_epoch)
    if D_optimizer is not None:
        D_scheduler = build_lambda_sche(D_optimizer, sche_config.kwargs, last_epoch=last_epoch)
    else:
        D_scheduler = None
    if R_optimizer is not None:
        R_scheduler = build_lambda_sche(R_optimizer, sche_config.kwargs, last_epoch=last_epoch)
    else:
        R_scheduler = None
    return G_scheduler, D_scheduler, R_scheduler

def resume_model(base_model, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger = logger)
    return start_epoch, best_metrics

def resume_optimizer(optimizer, args, logger = None, **kwargs):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger = logger )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    if 'G' in kwargs:
        optimizer.load_state_dict(state_dict['G_optimizer'])
    elif 'D' in kwargs:
        optimizer.load_state_dict(state_dict['D_optimizer'])
    elif 'R' in kwargs:
        optimizer.load_state_dict(state_dict['R_optimizer'])
    else:
        optimizer.load_state_dict(state_dict['optimizer'])

def save_checkpoint(base_model, G_optimizer, epoch, metrics, best_metrics, prefix, args, D_optimizer=None, R_optimizer=None, logger = None):
    
    torch.save({
                'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                'G_optimizer' : G_optimizer.state_dict(),
                'D_optimizer' : D_optimizer.state_dict() if D_optimizer is not None else dict(),
                'R_optimizer' : R_optimizer.state_dict() if R_optimizer is not None else dict(),
                'epoch' : epoch,
                'metrics' : metrics.state_dict() if metrics is not None else dict(),
                'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                }, os.path.join(args.experiment_path, prefix + '.pth'))
    print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)
    
def load_model(base_model, ckpt_path, logger = None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger )

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger)
    return 