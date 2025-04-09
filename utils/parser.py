import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    
    # general
    parser.add_argument('--config', type=str, default='./config/PPCC.yaml', help = 'yaml config file')
    parser.add_argument('--local_rank', type=int, default=0, help='current GPU')
    parser.add_argument('--num_workers', type=int, default=0, help='data processing CPU')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')   
    
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')      
    
    # bn
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    
    # some args
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--resume', action='store_true', default=False, help = 'autoresume training (interrupted by accident)')
    parser.add_argument('--mode', choices=['easy', 'median', 'hard', None], default=None, help = 'difficulty mode for shapenet')
    
    # evaluation
    parser.add_argument('--evaluation', action='store_true', default=False, help = 'activate test code')
    parser.add_argument('--eval_npoints', type=int, default=2048)
    parser.add_argument('--eval_dataset', type=str, default='PartUSSPA')
    parser.add_argument('--eval_only_exist', action='store_true')
    
    # ppcc
    parser.add_argument('--exp_name', type = str, default=None, help = 'experiment name')
    parser.add_argument('--category', type = str, default='chair', help = 'category')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--val_freq', type = int, default=1, help = 'validation freq')
    parser.add_argument('--print_freq', type = int, default=100, help = 'print freq')
    
    args = parser.parse_args()

    if args.resume and args.start_ckpts is not None:
        raise ValueError('--resume and --start_ckpts cannot be both activate')
    
    if args.evaluation:
        if args.resume: raise ValueError('--evaluation and --resume cannot be both activate')
        if args.ckpts is None: raise ValueError('ckpts shouldnt be None while test mode')
        args.evaluation_path = os.path.join('./evaluations', args.ckpts.split('/')[-2])
        args.log_name = Path(args.config).stem
        create_experiment_dir(args)
        return args
    else:
        args.experiment_path = os.path.join('./experiments', args.exp_name)
        args.tfboard_path = os.path.join('./tb_logger', args.exp_name)

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.mode is not None:
        args.exp_name = args.exp_name + f'_{args.mode}'
    
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if args.evaluation:
        if not os.path.exists(args.evaluation_path):
            os.makedirs(args.evaluation_path, exist_ok=True)
            print('Create experiment path successfully at %s' % args.evaluation_path)
    else:
        if not os.path.exists(args.experiment_path):
            os.makedirs(args.experiment_path, exist_ok=True)
            print('Create experiment path successfully at %s' % args.experiment_path)
        if not os.path.exists(args.tfboard_path):
            os.makedirs(args.tfboard_path, exist_ok=True)
            print('Create TFBoard path successfully at %s' % args.tfboard_path)