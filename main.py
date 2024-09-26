import os
import os.path as osp
import sys
from hand_ischemia.config import get_cfg_defaults, default_argument_parser, setup_logger
from hand_ischemia.engine import Hand_Ischemia_Trainer, Hand_Ischemia_Tester
from hand_ischemia.models import build_model
import mlflow
import time

import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

#mlflow.set_tracking_uri("http://127.0.0.1:5000")

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)



def main(rank, args, world_size, experiment_id):

    cfg = get_cfg_defaults()  # This is take from torch_SparsePPG/config/config.py
    # overwrite default configs args with those from file
    cfg.merge_from_file(args.config_file)
    # overwrite config args with those from command line
    cfg.merge_from_list(args.opts)
    #cfg.freeze()

    logger = setup_logger(cfg.OUTPUT.OUTPUT_DIR, distributed_rank=rank)

    if args.test_only:
        
        experiment_name = 'TEST-{}-{}'.format(time.strftime("%m-%d-%H:%M:%S"), args.experiment_id)
        
        #Create a new "experiment" to record data
        exp_id = mlflow.create_experiment(experiment_name)
        mlflow.start_run(experiment_id=exp_id)
        tester = Hand_Ischemia_Tester(cfg, args)
        
        logger.info('Inside Tester')
        #Use old experiment id to retriev runs
        tester.test(args.experiment_id, exp_id)
        
        mlflow.end_run()
        
        
        return
    
    #ddp_setup(rank, world_size)
    trainer = Hand_Ischemia_Trainer(cfg, rank)
    # Dump the configuration to file, as well as write the output directory
    logger.info('Dumping configuration')
    os.makedirs(cfg.OUTPUT.OUTPUT_DIR, exist_ok=True)
    
    if rank == 0:
        mlflow.start_run(experiment_id=experiment_id)
    
    trainer.train(experiment_id=experiment_id)
    #trainer.train_no_val(experiment_id=experiment_id)
    
    if rank == 0:
        mlflow.end_run()
    #destroy_process_group()

if __name__ == '__main__':

    args = default_argument_parser().parse_args()
    print("Command Line Args", args)

    
    world_size = torch.cuda.device_count()
    
    
    if args.test_only:
        #experiment_id = mlflow.create_experiment(time.strftime("%m-%d-%H:%M:%S"))
        main(0, args, world_size, None)
        
    else:
        experiment_id = mlflow.create_experiment(time.strftime("%m-%d-%H:%M:%S"))
        #mp.spawn(main, args=(args, world_size, experiment_id), nprocs=world_size)
        main(0, args, world_size, experiment_id)
