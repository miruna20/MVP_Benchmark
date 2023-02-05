import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import logging

# in the paper they consider 26 partial scans per mesh

class verse2020_lumbar(data.Dataset):

    def __init__(self, train_path, val_path, test_path, prefix="train", cluster=False, num_partial_scans_per_mesh=16):
        logging.info("Using vertebrae dataset")
        if prefix == "train":
            self.file_path = train_path

        elif prefix == "val":
            self.file_path = val_path

        elif prefix == "test":
            self.file_path = test_path
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        if (cluster):
            from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path
            data_paths_polyaxon = get_data_paths()
            self.file_path = os.path.join(data_paths_polyaxon['data1'], "USShapeCompletion", "MVP", self.file_path)
        self.prefix = prefix

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])

        print(self.input_data.shape)

        self.gt_data = np.array(input_file['complete_pcds'][()])
        self.labels = np.array(input_file['labels'][()])
        self.number_per_classes = np.array(input_file['number_per_class'][()])
        self.num_partial_scans_per_mesh = num_partial_scans_per_mesh

        print(self.gt_data.shape, self.labels.shape)

        input_file.close()
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))
        complete = torch.from_numpy((self.gt_data[index // self.num_partial_scans_per_mesh]))
        label = (self.labels[index])
        return label, partial, complete


class MVP_CP(data.Dataset):
    def __init__(self, prefix="train", cluster=False):
        logging.info("Using objects dataset")

        if prefix == "train":
            # self.file_path = 'completion/data/MVP_Train_CP.h5'
            # self.file_path = 'completion/data/first250Vertebrae.h5'
            self.file_path = 'completion/data/vertebrae_train_lumbar.h5'

        elif prefix == "val":
            self.file_path = 'completion/data/MVP_Test_CP.h5'
            # self.file_path = 'completion/data/next250Vertebrae.h5'
            # self.file_path = 'completion/data/vertebrae_val_lumbar.h5'

        elif prefix == "test":
            # self.file_path = 'completion/data/MVP_ExtraTest_Shuffled_CP.h5'
            self.file_path = 'completion/data/vertebrae_test_lumbar.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        if (cluster):
            from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path
            data_paths_polyaxon = get_data_paths()
            self.file_path = os.path.join(data_paths_polyaxon['data1'], "USShapeCompletion", "MVP", self.file_path)
        self.prefix = prefix

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])

        print(self.input_data.shape)

        # if prefix is not "test":
        self.gt_data = np.array(input_file['complete_pcds'][()])
        self.labels = np.array(input_file['labels'][()])
        print(self.gt_data.shape, self.labels.shape)

        input_file.close()
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))

        """
        if self.prefix is not "test":
            complete = torch.from_numpy((self.gt_data[index // nrPartialScansPerMesh]))
            label = (self.labels[index])
            return label, partial, complete
        else:
            return partial
        """
        complete = torch.from_numpy((self.gt_data[index // 26]))
        label = (self.labels[index])
        return label, partial, complete
