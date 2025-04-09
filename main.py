from tools import run_net
from tools import test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *

import os
import time
import torch
from tensorboardX import SummaryWriter

def main():
    # args
    args = parser.get_args()
    args.use_gpu = torch.cuda.is_available()

    if args.use_gpu:
        torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    
    # logger
    timestamp = time.strftime('%Y-%m-%d | %H:%M', time.localtime())
    if args.evaluation:
        log_file = os.path.join(args.evaluation_path, f'{timestamp}.log')
    else:
        log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    
    # define the tensorboard writer
    if not args.evaluation:
        writer = SummaryWriter(args.tfboard_path)
    
    # config
    config = get_config(args, logger = logger)

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic)
    
    # batch size
    args.bs = config.total_bs    
    
    # run
    if args.evaluation:
        test_net(args, config)
    else:
        run_net(args, config, writer)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()

