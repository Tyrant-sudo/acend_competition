from mindspore import ops
import os
import datetime
import time
import argparse
import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore import dtype as mstype
from mindspore import save_checkpoint, jit, data_sink
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindflow.common import get_warmup_cosine_annealing_lr
from mindflow.pde import SteadyFlowWithLoss
from mindflow.loss import WaveletTransformLoss
from mindflow.cell import ViT
from mindflow.utils import load_yaml_config, print_log, log_config

from src import AirfoilDataset, plot_u_and_cp, get_ckpt_summary_dir, plot_u_v_p, calculate_test_error

import warnings

set_seed(0)
np.random.seed(0)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description='Airfoil 2D_steady Simulation')
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--context_mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Support context mode: 'GRAPH', 'PYNATIVE'")
    parser.add_argument('--train_mode', type=str, default='train', choices=["train", "test", "finetune"],
                        help="Support run mode: 'train', 'test', 'finetune'")
    parser.add_argument('--device_id', type=int, default=0, help="ID of the target device")
    parser.add_argument('--device_target', type=str, default='GPU', choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--config_file_path", type=str, default="./configs/vit.yaml")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument('--ckpt_path', type=str, default='summary_dir_testVit/summary_D/ViT_D_bs_32/ckpt_dir/epoch_500.ckpt',
                    help='Path to the checkpoint file')
    input_args = parser.parse_args()
    return input_args

if __name__ == '__main__':
    log_config('./logs', 'vit')
    print_log(f'pid: {os.getpid()}')
    print_log(datetime.datetime.now())

    args = parse_args()
    
    # 修改模式为测试模式
    args.train_mode = 'test'
    config = load_yaml_config(args.config_file_path)
    model_params = config["model"]
    compute_dtype = mstype.float32
    data_params = config["data"]
    max_value_list = data_params['max_value_list']
    min_value_list = data_params['min_value_list']
    dataset = AirfoilDataset(max_value_list, min_value_list)
    mode = args.train_mode
    batch_size = data_params['batch_size']

    train_dataset, test_dataset = dataset.create_dataset(train_dataset_path=data_params['train_dataset_path'],
                                                        test_dataset_path=data_params['test_dataset_path'],
                                                        finetune_dataset_path=data_params['finetune_dataset_path'],
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        mode=mode,
                                                        finetune_size=data_params['finetune_size'],
                                                        drop_remainder=True)

    context.set_context(mode=context.GRAPH_MODE if args.context_mode.upper().startswith("GRAPH") \
        else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id)
    print_log(f"Running in {args.context_mode.upper()} mode, using device id: {args.device_id}.")
    start_time = time.time()

    # 加载模型
    model = ViT(in_channels=model_params['in_channels'],
                out_channels=model_params['out_channels'],
                encoder_depths=model_params['encoder_depth'],
                encoder_embed_dim=model_params['encoder_embed_dim'],
                encoder_num_heads=model_params['encoder_num_heads'],
                decoder_depths=model_params['decoder_depth'],
                decoder_embed_dim=model_params['decoder_embed_dim'],
                decoder_num_heads=model_params['decoder_num_heads'],
                compute_dtype=compute_dtype
                )
    param_dict = load_checkpoint(args.ckpt_path)  # 指定您的模型文件路径
    load_param_into_net(model, param_dict)
    print_log("Load pre-trained model successfully")
    
    # 运行测试
    model.set_train(False)
    calculate_test_error(test_dataset, model)
    print_log("End-to-End total time: {} s".format(time.time() - start_time))
