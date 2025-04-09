import torch
import torch.nn as nn

from tools import builder
from utils import misc
from utils.logger import *
from utils.metrics import Metrics
from utils.AverageMeter import AverageMeter
from extensions.chamfer_dist import *

import time
import numpy as np
from tqdm import tqdm

crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def run_net(args, config, writer=None):
    logger = get_logger(args.log_name)
    
    # build dataset and model
    config.model.category = config.dataset.category = args.category
    config.model.seg_num_all = config.seg_dict[config.dataset.category]
    (train_sampler, train_dataloader) = builder.dataset_builder(args, config, 'train')
    (_, val_dataloader) = builder.dataset_builder(args, config, 'val')
    
    base_model = builder.model_builder(config.model)
    
    if args.use_gpu:
        base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    tgt_best_metrics = None
    tgt_metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    
    # optimizer & scheduler
    G_optimizer, D_optimizer, R_optimizer = builder.build_optimizer(base_model, config)
    G_scheduler, D_scheduler, R_scheduler = builder.build_scheduler(
        base_model, (G_optimizer, D_optimizer, R_optimizer), config, last_epoch=start_epoch-1)
    
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        num_iter = 0
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        losses = AverageMeter(base_model.module.loss.loss_name)

        mean_loss = np.zeros([base_model.module.loss.loss_num])
        n_batches = len(train_dataloader)
        for idx, (_, data) in enumerate(train_dataloader):
            gt, tgt_partial = data
            gt, tgt_partial = gt.cuda(), tgt_partial.cuda()
            npoints = gt.shape[1]
            seg_num_all = config.seg_dict[config.dataset.category]
            mask = torch.cat([(gt[...,3] == sidx).sum(1).unsqueeze(-1) for sidx in range(seg_num_all)], -1) != 0
            src_partial, _ = misc.separate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
            
            outputs = base_model(((src_partial, mask), tgt_partial))
            loss = base_model.module.loss(outputs, data)
            
            loss[0] = config.model.loss.weight[0] * loss[0]
            loss[1] = config.model.loss.weight[1] * loss[1]
            if config.refine_start_epoch < epoch:
                loss[2] = config.model.loss.weight[2] * loss[2]
            
            mean_loss += np.array([l.item() for l in loss])
            
            G_optimizer.zero_grad()
            loss[0].backward(retain_graph=True)
            D_optimizer.zero_grad()
            loss[1].backward()
            if config.refine_start_epoch < epoch:
                R_optimizer.zero_grad()
                loss[2].backward()

            G_optimizer.step()
            D_optimizer.step()
            if config.refine_start_epoch < epoch:
                R_optimizer.step()
            
            base_model.zero_grad()
            losses.update([l.item() * 1000 for l in loss if l is not None])
            
            num_iter += 1
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % args.print_freq == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) || G_loss = %.2f, D_loss = %.2f, R_loss = %.2f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                             mean_loss[0]/(idx + 1), mean_loss[1]/(idx + 1), mean_loss[2]/(idx + 1)))
                mean_loss = np.zeros([base_model.module.loss.loss_num])

        for i, name in enumerate(base_model.module.loss.loss_name):
            writer.add_scalar(f'train/{name}', mean_loss[i], epoch)
        
        if isinstance(G_scheduler, list):
            for item in G_scheduler:
                item.step()
        if D_scheduler is not None:
            if isinstance(D_scheduler, list):
                for item in D_scheduler:
                    item.step()
        if R_scheduler is not None:
            if config.refine_start_epoch < epoch:
                if isinstance(R_scheduler, list):
                    for item in R_scheduler:
                        item.step()
        
        epoch_end_time = time.time()
        print_log('[Training][Source] EPOCH: %d EpochTime = %.3f || G_loss = %.2f, D_loss = %.2f, R_loss = %.2f' %
                  (epoch, epoch_end_time - epoch_start_time,
                   mean_loss[0]/n_batches, mean_loss[1]/n_batches, mean_loss[2]/n_batches))
                
        if epoch % args.val_freq == 0:
            _, tgt_metrics = validate(base_model, val_dataloader, epoch, writer, config, logger=logger)
            if epoch > 100:
                builder.save_checkpoint(base_model, G_optimizer, epoch, tgt_metrics, tgt_best_metrics, f'ckpt-{epoch}', args, D_optimizer, R_optimizer)
            builder.save_checkpoint(base_model, G_optimizer, epoch, tgt_metrics, tgt_best_metrics, 'ckpt-last', args, D_optimizer, R_optimizer)      

    if writer is not None:
        writer.close()

def validate(base_model, val_dataloader, epoch, writer, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}")
    
    # synthetic
    src_test_metrics = AverageMeter(Metrics.names())
    src_category_metrics = dict()
    
    # real
    tgt_test_metrics = AverageMeter(Metrics.names())
    tgt_category_metrics = dict()

    base_model.eval()
    with torch.no_grad():
        for _, (category, data) in enumerate(val_dataloader):
            
            gt, tgt = data
            tgt_partial, tgt, _ = tgt
            gt, tgt_partial, tgt = gt.cuda(), tgt_partial.cuda(), tgt.cuda()
            npoints = gt.shape[1]
            src_partial, _ = misc.separate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
            
            outputs = base_model((src_partial, tgt_partial), True)
            ref_src_pts, ref_tgt_pts = outputs
            
            src0 = ref_src_pts[1]
            tgt0 = ref_tgt_pts[1]
            
            gt = misc.fps(gt[:,:,:3], 2048)
            src_metrics = Metrics.get(src0, gt[:,:,:3].contiguous())
            tgt_metrics = Metrics.get(tgt0, tgt[:,:,:3].contiguous())
            src_metrics = [_metric.item() for _metric in src_metrics]
            tgt_metrics = [_metric.item() for _metric in tgt_metrics]

            # target category
            category = config.dataset.category
            src_category = config.dataset.category
            tgt_category = config.dataset.category
                
            if src_category not in src_category_metrics:
                src_category_metrics[src_category] = AverageMeter(Metrics.names())
            if tgt_category not in tgt_category_metrics:
                tgt_category_metrics[tgt_category] = AverageMeter(Metrics.names())
            src_category_metrics[src_category].update(src_metrics)
            tgt_category_metrics[tgt_category].update(tgt_metrics)
            
        for _, v in src_category_metrics.items():
            src_test_metrics.update(v.avg())
        
        for _, v in tgt_category_metrics.items():
            tgt_test_metrics.update(v.avg())
    
    # Print testing results
    print_log('------------------------------------- EPOCH %3s -------------------------------------' %(epoch),logger=logger)
    
    # synthetic
    msg = ''
    msg += '%-12s' %(category)
    for metric in src_test_metrics.items:
        msg += '%12s' %metric
    print_log(msg, logger=logger)

    for category in src_category_metrics:
        msg = ''
        msg += '%-12s' %(f'SYN_{category}')
        for value in src_category_metrics[category].avg():
            msg += '%12.3f' % value
        print_log(msg, logger=logger)

    for category in tgt_category_metrics:
        msg = ''
        msg += '%-12s' %(f'REAL_{category}')
        for value in tgt_category_metrics[category].avg():
            msg += '%12.3f' % value
        print_log(msg, logger=logger)

    print_log('-------------------------------------------------------------------------------------',logger=logger)

    if writer is not None:
        for i, metric in enumerate(src_test_metrics.items):
            writer.add_scalar('val_synthetic/%s' % metric, src_test_metrics.avg(i), epoch)
    
    if writer is not None:
        for i, metric in enumerate(tgt_test_metrics.items):
            writer.add_scalar('val_real/%s' % metric, tgt_test_metrics.avg(i), epoch)
    
    return Metrics(config.consider_metric, src_test_metrics.avg()), Metrics(config.consider_metric, tgt_test_metrics.avg())

def test_net(args, config):

    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    
    # build dataset and model
    config.model.category = config.dataset.category = args.category
    config.model.seg_num_all = config.seg_dict[args.category]
    _, test_dataloader = builder.dataset_builder(args, config, 'test')
    base_model = builder.model_builder(config.model)

    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    evaluation(base_model, test_dataloader, logger=logger)

def evaluation(base_model, test_dataloader, logger = None):

    base_model.eval()
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    with torch.no_grad():
        for _, (category, data) in enumerate(tqdm(test_dataloader)):
            
            tgt_partial, tgt, name = data
            name = name[0]
            tgt_partial, tgt = tgt_partial.cuda(), tgt.cuda()
            outputs = base_model((None, tgt_partial), False)
            ref_tgt_pts = outputs[1]

            _metrics = Metrics.get(ref_tgt_pts, tgt[:,:,:3].contiguous())
            _metrics = [_metric.item() for _metric in _metrics] 

            # target category
            if category not in category_metrics:
                category_metrics[category] = AverageMeter(Metrics.names())
            category_metrics[category].update(_metrics)

        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]))

    # Print testing results
    print_log('-------------------------------------- RESULT --------------------------------------',logger=logger)
    
    # synthetic
    msg = ''
    msg += '%-12s' %(category)
    for metric in test_metrics.items:
        msg += '%12s' %metric
    print_log(msg, logger=logger)

    for category in category_metrics:
        msg = ''
        msg += '%-12s' %('prediction')
        for value in category_metrics[category].avg():
            msg += '%12.3f' % value
        print_log(msg, logger=logger)
    return
