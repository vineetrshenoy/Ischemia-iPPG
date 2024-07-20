import os
import os.path as osp
import sys
from hand_ischemia.config import get_cfg_defaults, default_argument_parser, setup_logger
from hand_ischemia.engine.hand_ischemia_trainer import Hand_Ischemia_Trainer
import mlflow
import time


def main(args):

    cfg = get_cfg_defaults()  # This is take from torch_SparsePPG/config/config.py
    # overwrite default configs args with those from file
    cfg.merge_from_file(args.config_file)
    # overwrite config args with those from command line
    cfg.merge_from_list(args.opts)
    #cfg.freeze()

    logger = setup_logger(cfg.OUTPUT.OUTPUT_DIR, distributed_rank=0)

    trainer = Hand_Ischemia_Trainer(cfg)  # Build the algorithmic engine

    # Dump the configuration to file, as well as write the output directory
    logger.info('Dumping configuration')
    os.makedirs(cfg.OUTPUT.OUTPUT_DIR, exist_ok=True)
    output_config_path = osp.join(cfg.OUTPUT.OUTPUT_DIR, 'config.yaml')
    with open(output_config_path, 'w') as f:
        f.write(cfg.dump())

    experiment_id = mlflow.create_experiment(time.strftime("%m-%d-%H:%M:%S"))
    
    mlflow.start_run(experiment_id=experiment_id)
    
    #trainer.train(experiment_id=experiment_id)
    trainer.train_no_val(experiment_id=experiment_id)
    mlflow.end_run()


if __name__ == '__main__':

    args = default_argument_parser().parse_args()
    print("Command Line Args", args)

    main(args)
