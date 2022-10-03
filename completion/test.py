import logging
import os
import sys
import importlib
import argparse
import numpy as np
import h5py
import subprocess

from numpy.lib.index_tricks import AxisConcatenator
import munch
import yaml
# from utils.vis_utils import plot_single_pcd
# from utils.train_utils import *
from vis_utils import plot_single_pcd
from train_utils import *
from dataset import MVP_CP

import warnings
warnings.filterwarnings("ignore")

device = 'cuda'
device_ids = [0]


def test(cluster=False):
    # TODO the extra test set was only available during the competition
    # dataset_test = MVP_CP(prefix="test")
    dataset_test = MVP_CP(prefix="val",cluster=cluster)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')

    modelPath = args.load_model
    if(cluster):
        net = torch.nn.DataParallel(model_module.Model(args))

        from polyaxon_client.tracking import get_data_paths
        data_paths = get_data_paths()
        modelPath = os.path.join(data_paths['data1'], "USShapeCompletion/MVP", args.load_model)
    else:
        net = torch.nn.DataParallel(model_module.Model(args), device_ids=device_ids)
    net.to(device)
    net.module.load_state_dict(torch.load(modelPath)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()

    logging.info('Testing...')
    with torch.no_grad():
        results_list = []
        for i, data in enumerate(dataloader_test):

            # inputs_cpu = data
            label, inputs_cpu, gt = data

            inputs = inputs_cpu.float().to(device)
            inputs = inputs.transpose(2, 1).contiguous()

            result_dict = net(inputs, prefix="test")
            results_list.append(result_dict['result'].cpu().numpy())

            if i % args.step_interval_to_print == 0:
                logging.info('test [%d/%d]' % (i, dataset_length / args.batch_size))

        all_results = np.concatenate(results_list, axis=0)
        print(all_results.shape)

        with h5py.File(log_dir + '/results.h5', 'w') as f:
            f.create_dataset('results', data=all_results)
        
        cur_dir = os.getcwd()
        cmd = "cd %s; zip -r submission.zip results.h5 ; cd %s" % (log_dir, cur_dir)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        _, _ = process.communicate()
        print("Submission file has been saved to %s/submission.zip" % (log_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument(
        "--cluster",
        dest="cluster",
        default=False,
        action="store_true",
        help="If set, training on cluster will be enabled",
    )

    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')

    if(arg.cluster):
        from polyaxon_client.tracking import get_outputs_path
        output_directory = get_outputs_path()
        log_dir = os.path.join(output_directory, os.path.dirname(args.load_model))
    else:
        log_dir = os.path.dirname(args.load_model)

    # in case on the cluster the log_dir does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])

    test(arg.cluster)
