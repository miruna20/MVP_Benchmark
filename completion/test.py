import logging
import os
import sys
import importlib
import argparse
import numpy as np
import h5py
import subprocess

import munch
import yaml
from train_utils import *
from dataset import verse2020_lumbar
import time
import wandb
import warnings

warnings.filterwarnings("ignore")

device = 'cuda'
device_ids = [0]

def test():
    logging.info(str(args))

    prefix = args.data_to_test
    dataset_test = verse2020_lumbar(train_path=args.path_to_train_dataset,
                                    val_path=args.path_to_val_dataset,
                                    test_path=args.path_to_test_dataset,
                                    apply_trafo=args.apply_trafo,
                                    sigma = args.sigma,
                                    Xray_labelmap=args.use_labelmaps,
                                    prefix=prefix,
                                    num_partial_scans_per_mesh=args.num_partial_scans_per_mesh)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')

    modelPath = args.load_model


    net = torch.nn.DataParallel(model_module.Model(args), device_ids=device_ids)
    net.to(device)
    net.module.load_state_dict(torch.load(modelPath)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()

    # metrics we would like to compute
    if(args.eval_emd):
        metrics = ['cd_p', 'cd_p_arch', 'cd_t', 'cd_t_arch', 'emd', 'emd_arch', 'f1', 'f1_arch']
    else:
        metrics = ['cd_p','cd_p_arch', 'cd_t','cd_t_arch', 'f1','f1_arch']

    # dictionary with all of the metrics
    test_loss_meters = {m: AverageValueMeter() for m in metrics}

    # number of samples per class
    number_samples_per_class = dataset_test.number_per_classes
    num_partial_scans_per_category = dataset_test.num_partial_scans_per_mesh

    # number of categories
    num_categories = len(number_samples_per_class)

    # metrics for each category present (in our case 5 lumbar levels x 4 metrics we would like to compute)
    test_loss_cat = torch.zeros([num_categories, len(metrics)], dtype=torch.float32).cuda()

    # number of samples from each category
    cat_num = torch.ones([num_categories, 1], dtype=torch.float32).cuda()

    cat_name = ['L1', 'L2', 'L3', 'L4', 'L5']

    for class_ in range(0, len(number_samples_per_class)):
        cat_num[class_] = number_samples_per_class[class_] * num_partial_scans_per_category

    logging.info('Testing on ' + prefix + ' dataset')

    with torch.no_grad():
        results_list = []
        emd = []
        emd_arch = []
        emd_body = []
        cd_p = []
        cd_p_arch = []
        cd_p_body = []  # Add list for vertebral body CD_p
        cd_t = []
        cd_t_arch = []
        cd_t_body = []  # Add list for vertebral body CD_t
        f1 = []
        f1_arch = []
        f1_body = []  # Add list for vertebral body F1
        gts_aligned = []
        inputs_aligned = []
        for i, data in enumerate(dataloader_test):

            label, partial_pcd, labelmap, gt = data

            partial_pcd = partial_pcd.float().to(device)
            partial_pcd = partial_pcd.transpose(2, 1).contiguous()

            if args.use_labelmaps:
                labelmap = labelmap.float().to(device)
                labelmap = labelmap.transpose(2, 1).contiguous()

            gt = gt.float().to(device)

            result_dict = net(partial_pcd, labelmap, gt, prefix=prefix)

            # Update average metrics
            for k, v in test_loss_meters.items():
                v.update(result_dict[k].mean().item())

            # Sum up results for each metric
            for j, l in enumerate(label):
                for ind, m in enumerate(metrics):
                    test_loss_cat[int(l), ind] += result_dict[m][int(j)]

            # Append shape completion results
            results_list.append(result_dict['result'].cpu().numpy())
            if args.eval_emd:
                emd.append(result_dict['emd'].cpu().numpy())
                emd_arch.append(result_dict['emd_arch'].cpu().numpy())
                emd_body.append(result_dict['emd_body'].cpu().numpy())  # Append vertebral body EMD
            cd_p.append(result_dict['cd_p'].cpu().numpy())
            cd_p_arch.append(result_dict['cd_p_arch'].cpu().numpy())
            cd_p_body.append(result_dict['cd_p_body'].cpu().numpy())  # Append vertebral body CD_p
            cd_t.append(result_dict['cd_t'].cpu().numpy())
            cd_t_arch.append(result_dict['cd_t_arch'].cpu().numpy())
            cd_t_body.append(result_dict['cd_t_body'].cpu().numpy())  # Append vertebral body CD_t
            f1.append(result_dict['f1'].cpu().numpy())
            f1_arch.append(result_dict['f1_arch'].cpu().numpy())
            f1_body.append(result_dict['f1_body'].cpu().numpy())  # Append vertebral body F1
            gts_aligned.append(result_dict['gt'].cpu().numpy())
            inputs_aligned.append(result_dict['inputs'].cpu().numpy())

            if i % args.step_interval_to_print == 0:
                logging.info('test [%d/%d]' % (i, dataset_length / args.batch_size))

        logging.info('Loss per category:')
        category_log = ''
        table_data = []

        for i in range(num_categories):
            category_log += '\ncategory name: %s' % (cat_name[i])
            row_data = {
                "Category Name": cat_name[i]
            }
            for ind, m in enumerate(metrics):
                scale_factor = 1 if m in ['f1', 'f1_arch', 'f1_body'] else 10000
                category_log += ' %s: %f' % (m, test_loss_cat[i, ind] / cat_num[i] * scale_factor)
                value = test_loss_cat[i, ind] / cat_num[i] * scale_factor
                row_data[m] = value.cpu().numpy()[0]

            table_data.append(row_data)

        logging.info(category_log)
        #wandb.log({"category_metrics": wandb.Table(data=table_data, columns=["Category Name"] + metrics)})

        logging.info('Overview results:')
        overview_log = ''
        overview_results = []
        for metric, meter in test_loss_meters.items():
            scale_factor = 1 if metric in ['f1', 'f1_arch', 'f1_body'] else 10000
            overview_log += '%s: %f ' % (metric, meter.avg * scale_factor)
            overview_results.append((metric,meter.avg * scale_factor))

        #wandb.log({"Overview Results": wandb.Table(data=overview_results,columns=["Metric","Value"])})

        logging.info(overview_log)

        # Concatenate all results
        all_results = np.concatenate(results_list, axis=0)
        if args.eval_emd:
            all_emd = np.concatenate(emd, axis=0)
            all_emd_arch = np.concatenate(emd_arch, axis=0)
            all_emd_body = np.concatenate(emd_body, axis=0)  # Concatenate vertebral body EMD
        all_cd_p = np.concatenate(cd_p, axis=0)
        all_cd_p_arch = np.concatenate(cd_p_arch, axis=0)
        all_cd_p_body = np.concatenate(cd_p_body, axis=0)  # Concatenate vertebral body CD_p
        all_cd_t = np.concatenate(cd_t, axis=0)
        all_cd_t_arch = np.concatenate(cd_t_arch, axis=0)
        all_cd_t_body = np.concatenate(cd_t_body, axis=0)  # Concatenate vertebral body CD_t
        all_f1 = np.concatenate(f1, axis=0)
        all_f1_arch = np.concatenate(f1_arch, axis=0)
        all_f1_body = np.concatenate(f1_body, axis=0)  # Concatenate vertebral body F1
        all_gt = np.concatenate(gts_aligned, axis=0)
        all_inputs = np.concatenate(inputs_aligned, axis=0)

        # Write results to HDF5 file
        with h5py.File(log_dir + '/results.h5', 'w') as f:
            f.create_dataset('results', data=all_results)
            if args.eval_emd:
                f.create_dataset('emd', data=all_emd)
                f.create_dataset('emd_arch', data=all_emd_arch)
                f.create_dataset('emd_body', data=all_emd_body)  # Write vertebral body EMD
            f.create_dataset('cd_p', data=all_cd_p)
            f.create_dataset('cd_p_arch', data=all_cd_p_arch)
            f.create_dataset('cd_p_body', data=all_cd_p_body)  # Write vertebral body CD_p
            f.create_dataset('cd_t', data=all_cd_t)
            f.create_dataset('cd_t_arch', data=all_cd_t_arch)
            f.create_dataset('cd_t_body', data=all_cd_t_body)  # Write vertebral body CD_t
            f.create_dataset('f1', data=all_f1)
            f.create_dataset('f1_arch', data=all_f1_arch)
            f.create_dataset('f1_body', data=all_f1_body)  # Write vertebral body F1
            f.create_dataset('gt', data=all_gt)
            # f.create_dataset('input_aligned', data=all_inputs)  # Uncomment if needed

        # Create submission zip file
        cur_dir = os.getcwd()
        cmd = "cd %s; zip -r submission.zip results.h5 ; cd %s" % (log_dir, cur_dir)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, _ = process.communicate()
        print("Submission file has been saved to %s/submission.zip" % (log_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)

    torch.cuda.empty_cache()
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    #wandb.login(key="845cb3b94791a8d541b28fd3a9b2887374fe8b2c")
    #run = wandb.init(project="Multimodal Shape Completion",tags=['inference'])
    #wandb.config.update(args)

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')

    output_directory = "results"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(output_directory,args.name_folder_to_save_results)

    # in case on the log_dir does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])
    #logging.info(f"Project URL: {run.get_project_url()}")
    #logging.info(f"Run URL: { run.get_url()}")
    #logging.info(f"Local directory: {run.dir}")

    test()
    #wandb.finish()
