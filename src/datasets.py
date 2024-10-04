"""
The code is adapted from the following repository:
https://github.com/clinicalml/human_ai_deferral
"""

from abc import ABC, abstractmethod
import logging
import sys
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

sys.path.append("../")
from PIL import Image
from torch.utils.data import Dataset

logging.getLogger("PIL").setLevel(logging.WARNING)
import os
import requests
import tarfile
import urllib
import random
import shutil
import torchvision.transforms as transforms
from src.networks import DenseNet121_CE, ModelPredictAAE, Linear_net_sig
from torchtext import data
from sentence_transformers import SentenceTransformer
import torchvision.datasets as datasets


class BaseDataset(ABC):
    """Abstract method for learning to defer methods"""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """must at least have data_dir, test_split, val_split, batch_size, transforms"""
        pass

    @abstractmethod
    def generate_data(self):
        """generates the data loader, called on init

        should generate the following must:
            self.data_train_loader
            self.data_val_loader
            self.data_test_loader
            self.d (dimension)
            self.n_dataset (number of classes in target)
        """
        pass


class GenericImageExpertDataset(Dataset):
    def __init__(self, images, targets, expert_preds, transforms_fn, to_open=False):
        """

        Args:
            images (list): List of images
            targets (list): List of labels
            expert_preds (list): List of expert predictions
            transforms_fn (function): Function to apply to images
            to_open (bool): Whether to open images or not (RGB reader)
        """
        self.images = images
        self.targets = np.array(targets)
        self.expert_preds = np.array(expert_preds)
        self.transforms_fn = transforms_fn
        self.to_open = to_open

    def __getitem__(self, index):
        """Take the index of item and returns the image, label, expert prediction and index in original dataset"""
        label = self.targets[index]
        if self.transforms_fn is not None and self.to_open:
            image_paths = self.images[index]
            image = Image.open(image_paths).convert("RGB")
            image = self.transforms_fn(image)
        elif self.transforms_fn is not None:
            image = self.transforms_fn(self.images[index])
        else:
            image = self.images[index]
        expert_pred = self.expert_preds[index]
        return torch.FloatTensor(image), label, expert_pred

    def __len__(self):
        return len(self.targets)


class GenericDatasetDeferral(BaseDataset):
    def __init__(
        self,
        data_train,
        data_test=None,
        test_split=0.2,
        val_split=0.1,
        batch_size=128,
        transforms=None,
        seed=42,
    ):
        """

        data_train: training data expectd as dict with keys 'data_x', 'data_y', 'hum_preds'
        data_test: test data expectd as dict with keys 'data_x', 'data_y', 'hum_preds'
        test_split: fraction of training data to use for test
        val_split: fraction of training data to use for validation
        batch_size: batch size for dataloaders
        transforms: transforms to apply to images
        """
        self.data_train = data_train
        self.data_test = data_test
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.train_split = 1 - test_split - val_split
        self.transforms = transforms
        self.generate_data()
        self.seed = seed

    def generate_data(self):
        train_x = self.data_train["data_x"]
        train_y = self.data_train["data_y"]
        train_hum_preds = self.data_train["hum_preds"]
        if self.data_test is not None:
            test_x = self.data_test["data_x"]
            test_y = self.data_test["data_y"]
            test_h = self.data_test["hum_preds"]
            train_size = int((1 - self.val_split) * self.total_samples)
            val_size = int(self.val_split * self.total_samples)
            train_x, val_x = torch.utils.data.random_split(
                train_x,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )
            train_y, val_y = torch.utils.data.random_split(
                train_y,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )
            train_h, val_h = torch.utils.data.random_split(
                train_hum_preds,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )
            self.data_train = torch.utils.data.TensorDataset(
                train_x.dataset.data[train_x.indices],
                train_y.dataset.data[train_y.indices],
                train_h.dataset.data[train_h.indices],
            )
            self.data_val = torch.utils.data.TensorDataset(
                val_x.dataset.data[val_x.indices],
                val_y.dataset.data[val_y.indices],
                val_h.dataset.data[val_h.indices],
            )
            self.data_test = torch.utils.data.TensorDataset(test_x, test_y, test_h)

        else:
            train_size = int(self.train_split * self.total_samples)
            val_size = int(self.val_split * self.total_samples)
            test_size = self.total_samples - train_size - val_size
            train_x, val_x, test_x = torch.utils.data.random_split(
                train_x,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed),
            )
            train_y, val_y, test_y = torch.utils.data.random_split(
                train_y,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed),
            )
            train_h, val_h, test_h = torch.utils.data.random_split(
                train_hum_preds,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            self.data_train = torch.utils.data.TensorDataset(
                train_x.dataset.data[train_x.indices],
                train_y.dataset.data[train_y.indices],
                train_h.dataset.data[train_h.indices],
            )
            self.data_val = torch.utils.data.TensorDataset(
                val_x.dataset.data[val_x.indices],
                val_y.dataset.data[val_y.indices],
                val_h.dataset.data[val_h.indices],
            )
            self.data_test = torch.utils.data.TensorDataset(
                test_x.dataset.data[test_x.indices],
                test_y.dataset.data[test_y.indices],
                test_h.dataset.data[test_h.indices],
            )

        self.data_train_loader = torch.utils.data.DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        self.data_val_loader = torch.utils.data.DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        self.data_test_loader = torch.utils.data.DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )


class ChestXrayDataset(BaseDataset):
    """Chest X-ray dataset from NIH with multiple radiologist annotations per point from Google Research"""

    def __init__(
        self,
        non_deferral_dataset,
        use_data_aug,
        data_dir,
        label_chosen,
        test_split=0.2,
        val_split=0.1,
        batch_size=512,
        get_embeddings=False,
        transforms=None,
    ):
        """
        See https://nihcc.app.box.com/v/ChestXray-NIHCC and
        non_deferral_dataset (bool): if True, the dataset is the non-deferral dataset, meaning it is the full NIH dataset without the val-test of the human labeled, otherwise it is the deferral dataset that is only 4k in size total
        data_dir: where to save files for model
        label_chosen (int in 0,1,2,3): if non_deferral_dataset = False: which label to use between 0,1,2,3 which correspond to Fracture, Pneumotheras,  Airspace Opacity, and Nodule/Mass; if true: then it's NoFinding or not, Pneumotheras, Effusion, Nodule/Mass
        use_data_aug: whether to use data augmentation (bool)
        test_split: percentage of test data
        val_split: percentage of data to be used for validation (from training set)
        batch_size: batch size for training
        transforms: data transforms
        """
        self.non_deferral_dataset = non_deferral_dataset
        self.data_dir = data_dir
        self.use_data_aug = use_data_aug
        self.label_chosen = label_chosen
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.n_dataset = 2
        self.get_embeddings = get_embeddings
        self.d = 1024
        self.train_split = 1 - test_split - val_split
        self.transforms = transforms
        self.generate_data()

    def generate_data(self):
        """
        generate data for training, validation and test sets
        """

        links = [
            "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
            "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
            "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
            "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
            "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
            "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
            "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
            "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
            "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
            "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
            "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
            "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
        ]
        max_links = 12  # 12 is the limit
        links = links[:max_links]

        if not os.path.exists(self.data_dir + "/images_nih"):
            logging.info("Downloading NIH dataset")
            for idx, link in enumerate(links):
                if not os.path.exists(
                    self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                ):
                    fn = self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                    logging.info("downloading " + fn + "...")
                    urllib.request.urlretrieve(link, fn)  # download the zip file

            logging.info("Download complete. Please check the checksums")

            # make directory
            if not os.path.exists(self.data_dir + "/images_nih"):
                os.makedirs(self.data_dir + "/images_nih")

            # extract files
            for idx in range(max_links):
                fn = self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                logging.info("Extracting " + fn + "...")
                # os.system('tar -zxvf '+fn+' -C '+self.data_dir+'/images_nih')
                file = tarfile.open(fn)
                file.extractall(self.data_dir + "/images_nih")
                file.close()
                fn = self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                # os.remove(fn)
                logging.info("Done")
        else:
            # double check that all files are there and extracted
            # get number of files in directory
            # if not equal to 102120, then download again
            num_files = len(
                [
                    name
                    for name in os.listdir(self.data_dir + "/images_nih")
                    if os.path.isfile(os.path.join(self.data_dir + "/images_nih", name))
                ]
            )
            if num_files < 102120:  # acutal is 112120
                logging.info("Files missing. Re-downloading...")
                shutil.rmtree(self.data_dir + "/images_nih")

                for idx, link in enumerate(links):
                    # check if file exists
                    fn = self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                    if not os.path.exists(
                        self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                    ):
                        logging.info("downloading " + fn + "...")
                        urllib.request.urlretrieve(link, fn)

                logging.info("Download complete. Please check the checksums")

                # make directory
                if not os.path.exists(self.data_dir + "/images_nih"):
                    os.makedirs(self.data_dir + "/images_nih")

                # extract files
                for idx in range(max_links):
                    fn = self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                    logging.info("Extracting " + fn + "...")
                    # os.system('tar -zxvf '+fn+' -C '+self.data_dir+'/images_nih')
                    file = tarfile.open(fn)
                    file.extractall(self.data_dir + "/images_nih")
                    file.close()
                    fn = self.data_dir + "/images_%02d.tar.gz" % (idx + 1)
                    # os.remove(fn)
                    logging.info("Done")

        # DOWNLOAD CSV DATA FOR LABELS

        if (
            not os.path.exists(
                self.data_dir + "/four_findings_expert_labels_individual_readers.csv"
            )
            or not os.path.exists(
                self.data_dir + "/four_findings_expert_labels_test_labels.csv"
            )
            or not os.path.exists(
                self.data_dir + "/four_findings_expert_labels_validation_labels.csv"
            )
            or not os.path.exists(self.data_dir + "/Data_Entry_2017_v2020.csv")
        ):
            logging.info("Downloading readers NIH data")
            r = requests.get(
                "https://storage.googleapis.com/gcs-public-data--healthcare-nih-chest-xray-labels/four_findings_expert_labels/individual_readers.csv",
                allow_redirects=True,
            )

            with open(
                self.data_dir + "/four_findings_expert_labels_individual_readers.csv",
                "wb",
            ) as f:
                f.write(r.content)
            r = requests.get(
                "https://storage.googleapis.com/gcs-public-data--healthcare-nih-chest-xray-labels/four_findings_expert_labels/test_labels.csv",
                allow_redirects=True,
            )
            with open(
                self.data_dir + "/four_findings_expert_labels_test_labels.csv", "wb"
            ) as f:
                f.write(r.content)
            r = requests.get(
                "https://storage.googleapis.com/gcs-public-data--healthcare-nih-chest-xray-labels/four_findings_expert_labels/validation_labels.csv",
                allow_redirects=True,
            )
            with open(
                self.data_dir + "/four_findings_expert_labels_validation_labels.csv",
                "wb",
            ) as f:
                f.write(r.content)

            logging.info("Finished Downloading  readers NIH data")
            r = requests.get(
                "https://raw.githubusercontent.com/raj713335/AI-IN-MEDICINE-SPECIALIZATION/master/DATA/Data_Entry_2017_v2020.csv"
            )
            with open(self.data_dir + "/Data_Entry_2017_v2020.csv", "wb") as f:
                f.write(r.content)

            try:
                readers_data = pd.read_csv(
                    self.data_dir
                    + "/four_findings_expert_labels_individual_readers.csv"
                )
                test_data = pd.read_csv(
                    self.data_dir + "/four_findings_expert_labels_test_labels.csv"
                )
                validation_data = pd.read_csv(
                    self.data_dir + "/four_findings_expert_labels_validation_labels.csv"
                )
                all_dataset_data = pd.read_csv(
                    self.data_dir + "/Data_Entry_2017_v2020.csv"
                )

            except:
                logging.error("Failed to load readers NIH data")
                raise

        else:
            logging.info("Loading readers NIH data")
            try:
                readers_data = pd.read_csv(
                    self.data_dir
                    + "/four_findings_expert_labels_individual_readers.csv"
                )
                test_data = pd.read_csv(
                    self.data_dir + "/four_findings_expert_labels_test_labels.csv"
                )
                validation_data = pd.read_csv(
                    self.data_dir + "/four_findings_expert_labels_validation_labels.csv"
                )
                all_dataset_data = pd.read_csv(
                    self.data_dir + "/Data_Entry_2017_v2020.csv"
                )
            except:
                logging.error("Failed to load readers NIH data")
                raise

        data_labels = {}
        for i in range(len(validation_data)):
            labels = [
                validation_data.iloc[i]["Fracture"],
                validation_data.iloc[i]["Pneumothorax"],
                validation_data.iloc[i]["Airspace opacity"],
                validation_data.iloc[i]["Nodule or mass"],
            ]
            # covert YES to 1 and otherwise to 0
            labels = [1 if x == "YES" else 0 for x in labels]
            data_labels[validation_data.iloc[i]["Image Index"]] = labels
        for i in range(len(test_data)):
            labels = [
                test_data.iloc[i]["Fracture"],
                test_data.iloc[i]["Pneumothorax"],
                test_data.iloc[i]["Airspace opacity"],
                test_data.iloc[i]["Nodule or mass"],
            ]
            # covert YES to 1 and otherwise to 0
            labels = [1 if x == "YES" else 0 for x in labels]
            data_labels[test_data.iloc[i]["Image Index"]] = labels

        data_human_labels = {}
        for i in range(len(readers_data)):
            labels = [
                readers_data.iloc[i]["Fracture"],
                readers_data.iloc[i]["Pneumothorax"],
                readers_data.iloc[i]["Airspace opacity"],
                readers_data.iloc[i]["Nodule/mass"],
            ]
            # covert YES to 1 and otherwise to 0
            labels = [1 if x == "YES" else 0 for x in labels]
            if readers_data.iloc[i]["Image ID"] in data_human_labels:
                data_human_labels[readers_data.iloc[i]["Image ID"]].append(labels)
            else:
                data_human_labels[readers_data.iloc[i]["Image ID"]] = [labels]

        # for each key in data_human_labels, we have a list of lists, sample only one list from each key
        data_human_labels = {
            k: random.sample(v, 1)[0] for k, v in data_human_labels.items()
        }

        labels_categories = [
            "Fracture",
            "Pneumothorax",
            "Airspace opacity",
            "Nodule/mass",
        ]
        self.label_to_idx = {
            labels_categories[i]: i for i in range(len(labels_categories))
        }

        image_to_patient_id = {}
        for i in range(len(readers_data)):
            image_to_patient_id[readers_data.iloc[i]["Image ID"]] = readers_data.iloc[
                i
            ]["Patient ID"]

        patient_ids = list(set(image_to_patient_id.values()))

        data_all_nih_label = {}
        # the original dataset has the following labels ['Atelectasis' 'Cardiomegaly' 'Consolidation' 'Edema' 'Effusion' 'Emphysema' 'Fibrosis' 'Hernia' 'Infiltration' 'Mass' 'No Finding' 'Nodule' 'Pleural_Thickening' 'Pneumonia' 'Pneumothorax']
        for i in range(len(all_dataset_data)):
            if not all_dataset_data["Patient ID"][i] in patient_ids:
                labels = [0, 0, 0, 0]
                if "Pneumothorax" in all_dataset_data["Finding Labels"][i]:
                    labels[1] = 1
                if "Effusion" in all_dataset_data["Finding Labels"][i]:
                    labels[2] = 1
                if (
                    "Mass" in all_dataset_data["Finding Labels"][i]
                    or "Nodule" in all_dataset_data["Finding Labels"][i]
                ):
                    labels[3] = 1
                if "No Finding" in all_dataset_data["Finding Labels"][i]:
                    labels[0] = 0
                else:
                    labels[0] = 1
                data_all_nih_label[all_dataset_data["Image Index"][i]] = labels

        # depending on non_deferral_dataset
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        if self.non_deferral_dataset == True:
            # iterate over key, value in data_all_nih_label
            data_y = []
            data_expert = []
            image_paths = []
            for key, value in list(data_all_nih_label.items()):
                image_path = self.data_dir + "/images_nih/" + key
                # check if the file exists
                if os.path.isfile(image_path):
                    data_y.append(value[self.label_chosen])
                    image_paths.append(self.data_dir + "/images_nih/" + key)
                    data_expert.append(value[self.label_chosen])  # nonsense expert

            data_y = np.array(data_y)
            data_expert = np.array(data_expert)
            image_paths = np.array(image_paths)

            random_seed = random.randrange(10000)

            test_size = int(self.test_split * len(image_paths))
            val_size = int(self.val_split * len(image_paths))
            train_size = len(image_paths) - test_size - val_size

            train_x, val_x, test_x = torch.utils.data.random_split(
                image_paths,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(random_seed),
            )
            train_y, val_y, test_y = torch.utils.data.random_split(
                data_y,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(random_seed),
            )
            train_h, val_h, test_h = torch.utils.data.random_split(
                data_expert,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(random_seed),
            )

            data_train = GenericImageExpertDataset(
                train_x.dataset[train_x.indices],
                train_y.dataset[train_y.indices],
                train_h.dataset[train_h.indices],
                transform_train,
                to_open=True,
            )

            data_val = GenericImageExpertDataset(
                val_x.dataset[val_x.indices],
                val_y.dataset[val_y.indices],
                val_h.dataset[val_h.indices],
                transform_test,
                to_open=True,
            )
            data_test = GenericImageExpertDataset(
                test_x.dataset[test_x.indices],
                test_y.dataset[test_y.indices],
                test_h.dataset[test_h.indices],
                transform_test,
                to_open=True,
            )
            self.data_train_loader = torch.utils.data.DataLoader(
                data_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
            )
            self.data_val_loader = torch.utils.data.DataLoader(
                data_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )
            self.data_test_loader = torch.utils.data.DataLoader(
                data_test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )

        else:
            # split patient_ids into train and test, val
            random.shuffle(patient_ids, random.random)
            # split using 80% for trarain, 10% for test and 10% for validation
            train_patient_ids = patient_ids[: int(len(patient_ids) * self.train_split)]
            test_patient_ids = patient_ids[
                int(len(patient_ids) * self.train_split) : int(
                    len(patient_ids) * (self.train_split + self.test_split)
                )
            ]
            val_patient_ids = patient_ids[
                int(len(patient_ids) * (self.train_split + self.test_split)) :
            ]
            # go from patient ids to image ids
            train_image_ids = np.array(
                [k for k, v in image_to_patient_id.items() if v in train_patient_ids]
            )
            val_image_ids = np.array(
                [k for k, v in image_to_patient_id.items() if v in val_patient_ids]
            )
            test_image_ids = np.array(
                [k for k, v in image_to_patient_id.items() if v in test_patient_ids]
            )
            # remove images that are not in the directory
            train_image_ids = np.array(
                [
                    k
                    for k in train_image_ids
                    if os.path.isfile(self.data_dir + "/images_nih/" + k)
                ]
            )
            val_image_ids = np.array(
                [
                    k
                    for k in val_image_ids
                    if os.path.isfile(self.data_dir + "/images_nih/" + k)
                ]
            )
            test_image_ids = np.array(
                [
                    k
                    for k in test_image_ids
                    if os.path.isfile(self.data_dir + "/images_nih/" + k)
                ]
            )

            logging.info("Finished splitting data into train, test and validation")
            # print sizes
            logging.info("Train size: {}".format(len(train_image_ids)))
            logging.info("Test size: {}".format(len(test_image_ids)))
            logging.info("Validation size: {}".format(len(val_image_ids)))

            train_y = np.array(
                [data_labels[k][self.label_chosen] for k in train_image_ids]
            )
            val_y = np.array([data_labels[k][self.label_chosen] for k in val_image_ids])
            test_y = np.array(
                [data_labels[k][self.label_chosen] for k in test_image_ids]
            )
            train_h = np.array(
                [data_human_labels[k][self.label_chosen] for k in train_image_ids]
            )
            val_h = np.array(
                [data_human_labels[k][self.label_chosen] for k in val_image_ids]
            )
            test_h = np.array(
                [data_human_labels[k][self.label_chosen] for k in test_image_ids]
            )
            train_image_ids = np.array(
                [self.data_dir + "/images_nih/" + k for k in train_image_ids]
            )
            val_image_ids = np.array(
                [self.data_dir + "/images_nih/" + k for k in val_image_ids]
            )
            test_image_ids = np.array(
                [self.data_dir + "/images_nih/" + k for k in test_image_ids]
            )

            data_train = GenericImageExpertDataset(
                train_image_ids, train_y, train_h, transform_train, to_open=True
            )
            data_val = GenericImageExpertDataset(
                val_image_ids, val_y, val_h, transform_test, to_open=True
            )
            data_test = GenericImageExpertDataset(
                test_image_ids, test_y, test_h, transform_test, to_open=True
            )

            self.data_train_loader = torch.utils.data.DataLoader(
                data_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
            )
            self.data_val_loader = torch.utils.data.DataLoader(
                data_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )
            self.data_test_loader = torch.utils.data.DataLoader(
                data_test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )

        if self.get_embeddings:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            path_model = "../exp_data/models/chextxray_dn121_3epochs.pt"
            model_linear = DenseNet121_CE(2).to(device)
            # torch load
            model_linear.load_state_dict(torch.load(path_model))

            model_linear.densenet121.classifier = nn.Sequential(
                *list(model_linear.densenet121.classifier.children())[:-1]
            )

            # get embeddings of train-val-test data
            def get_embeddings(model, data_loader):
                model.eval()
                with torch.no_grad():
                    embeddings = []
                    for i, (x, y, h) in enumerate(data_loader):
                        x = x.to(device)
                        y = y.to(device)
                        h = h.to(device)
                        x = model(x)
                        embeddings.append(x.cpu().numpy())
                return np.concatenate(embeddings, axis=0)

            train_embeddings = torch.FloatTensor(
                get_embeddings(model_linear, self.data_train_loader)
            )
            val_embeddings = torch.FloatTensor(
                get_embeddings(model_linear, self.data_val_loader)
            )
            test_embeddings = torch.FloatTensor(
                get_embeddings(model_linear, self.data_test_loader)
            )

            data_train = torch.utils.data.TensorDataset(
                train_embeddings,
                torch.from_numpy(train_y),
                torch.from_numpy(train_h),
            )
            data_val = torch.utils.data.TensorDataset(
                val_embeddings,
                torch.from_numpy(val_y),
                torch.from_numpy(val_h),
            )
            data_test = torch.utils.data.TensorDataset(
                test_embeddings,
                torch.from_numpy(test_y),
                torch.from_numpy(test_h),
            )

            self.data_train_loader = torch.utils.data.DataLoader(
                data_train,
                batch_size=3000,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )
            self.data_val_loader = torch.utils.data.DataLoader(
                data_val,
                batch_size=3000,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )
            self.data_test_loader = torch.utils.data.DataLoader(
                data_test,
                batch_size=3000,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )


# https://github.com/jcpeterson/cifar-10h
class Cifar10h(BaseDataset):
    """CIFAR-10H dataset with seperate human annotations on the test set of CIFAR-10"""

    def __init__(
        self,
        use_data_aug,
        data_dir,
        test_split=0.2,
        val_split=0.1,
        batch_size=128,
        transforms=None,
    ):
        """
        data_dir: where to save files for model
        use_data_aug: whether to use data augmentation (bool)
        test_split: percentage of test data
        val_split: percentage of data to be used for validation (from training set)
        batch_size: batch size for training
        transforms: data transforms
        """
        self.data_dir = data_dir
        self.use_data_aug = use_data_aug
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.n_dataset = 10
        self.train_split = 1 - test_split - val_split
        self.transforms = transforms
        self.generate_data()

    def metrics_cifar10h(self, exp_preds, labels):
        correct = 0
        total = 0
        j = 0
        self.class_conditional_acc = [0] * 10
        class_counts = [0] * 10
        for i in range(len(exp_preds)):
            total += 1
            j += 1
            correct += exp_preds[i] == labels[i]
            self.class_conditional_acc[labels[i]] += exp_preds[i] == labels[i]
            class_counts[labels[i]] += 1
        for i in range(0, 10):
            self.class_conditional_acc[i] = (
                100 * self.class_conditional_acc[i] / class_counts[i]
            )
        self.human_accuracy = 100 * correct / total

    def generate_data(self):
        """
        generate data for training, validation and test sets
        : "airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9
        """
        # download cifar10h data
        # check if file already exists
        # check if file already exists
        if not os.path.exists(self.data_dir + "/cifar10h-probs.npy"):
            logging.info("Downloading cifar10h data")
            r = requests.get(
                "https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-probs.npy",
                allow_redirects=True,
            )
            with open(self.data_dir + "/cifar10h-probs.npy", "wb") as f:
                f.write(r.content)
            logging.info("Finished Downloading cifar10h data")
            try:
                cifar10h = np.load(self.data_dir + "/cifar10h-probs.npy")
            except:
                logging.error("Failed to load cifar10h data")
                raise
        else:
            logging.info("Loading cifar10h data")
            try:
                cifar10h = np.load(self.data_dir + "/cifar10h-probs.npy")
            except:
                logging.error("Failed to load cifar10h data")
                raise

        human_predictions = np.array(
            [
                np.argmax(np.random.multinomial(1, cifar10h[i]))
                for i in range(len(cifar10h))
            ]
        )

        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

        if self.use_data_aug:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: F.pad(
                            x.unsqueeze(0), (4, 4, 4, 4), mode="reflect"
                        ).squeeze()
                    ),
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        dataset = "cifar10"
        kwargs = {"num_workers": 0, "pin_memory": True}

        train_dataset_all = datasets.__dict__[dataset.upper()](
            "../data", train=False, download=True, transform=transform_test
        )
        labels_all = train_dataset_all.targets
        self.metrics_cifar10h(human_predictions, labels_all)

        test_size = int(self.test_split * len(train_dataset_all))
        val_size = int(self.val_split * len(train_dataset_all))
        train_size = len(train_dataset_all) - test_size - val_size

        train_x = train_dataset_all.data
        train_y = train_dataset_all.targets
        train_y = np.array(train_y)
        random_seed = random.randrange(10000)

        train_x, val_x, test_x = torch.utils.data.random_split(
            train_x,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_y, val_y, test_y = torch.utils.data.random_split(
            train_y,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_h, val_h, test_h = torch.utils.data.random_split(
            human_predictions,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )

        data_train = GenericImageExpertDataset(
            train_x.dataset[train_x.indices],
            train_y.dataset[train_y.indices],
            train_h.dataset[train_h.indices],
            transform_train,
        )
        data_val = GenericImageExpertDataset(
            val_x.dataset[val_x.indices],
            val_y.dataset[val_y.indices],
            val_h.dataset[val_h.indices],
            transform_test,
        )
        data_test = GenericImageExpertDataset(
            test_x.dataset[test_x.indices],
            test_y.dataset[test_y.indices],
            test_h.dataset[test_h.indices],
            transform_test,
        )

        self.data_train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        self.data_val_loader = torch.utils.data.DataLoader(
            data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        self.data_test_loader = torch.utils.data.DataLoader(
            data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )


class CifarSynthExpert:
    """simple class to describe our synthetic expert on CIFAR-10    k: number of classes expert can predict, n_classes: number of classes (10 for CIFAR-10)"""

    def __init__(self, k, n_classes):
        self.k = k
        self.n_classes = n_classes

    def predict(self, labels):
        batch_size = len(labels)
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i] <= self.k - 1:
                outs[i] = labels[i]
            else:
                prediction_rand = random.randint(0, self.n_classes - 1)
                outs[i] = prediction_rand
        return outs


class CifarSynthDataset(BaseDataset):
    """This is the CifarK synthetic expert on top of Cifar-10 from Consistent Estimators for Learning to Defer (https://arxiv.org/abs/2006.01862)"""

    def __init__(
        self,
        expert_k,
        use_data_aug,
        test_split=0.2,
        val_split=0.1,
        batch_size=128,
        n_dataset=10,
        transforms=None,
    ):
        """
        expert_k: number of classes expert can predict
        use_data_aug: whether to use data augmentation (bool)
        test_split: NOT USED FOR CIFAR, since we have a fixed test set
        val_split: percentage of data to be used for validation (from training set)
        batch_size: batch size for training
        transforms: data transforms
        """
        self.expert_k = expert_k
        self.use_data_aug = use_data_aug
        self.n_dataset = n_dataset
        self.expert_fn = CifarSynthExpert(expert_k, self.n_dataset).predict
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.train_split = 1 - test_split - val_split
        self.transforms = transforms
        self.generate_data()

    def generate_data(self):
        """
        generate data for training, validation and test sets
        """
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

        if self.use_data_aug:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: F.pad(
                            x.unsqueeze(0), (4, 4, 4, 4), mode="reflect"
                        ).squeeze()
                    ),
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        if self.n_dataset == 10:
            dataset = "cifar10"
        elif self.n_dataset == 100:
            dataset = "cifar100"

        kwargs = {"num_workers": 8, "pin_memory": True}

        train_dataset_all = datasets.__dict__[dataset.upper()](
            "../data", train=True, download=True, transform=transform_train
        )
        train_size = int((1 - self.val_split) * len(train_dataset_all))
        val_size = len(train_dataset_all) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset_all, [train_size, val_size]
        )

        test_dataset = datasets.__dict__["cifar10".upper()](
            "../data", train=False, transform=transform_test, download=True
        )

        dataset_train = GenericImageExpertDataset(
            np.array(train_dataset.dataset.data)[train_dataset.indices],
            np.array(train_dataset.dataset.targets)[train_dataset.indices],
            self.expert_fn(
                np.array(train_dataset.dataset.targets)[train_dataset.indices]
            ),
            transform_train,
        )
        dataset_val = GenericImageExpertDataset(
            np.array(val_dataset.dataset.data)[val_dataset.indices],
            np.array(val_dataset.dataset.targets)[val_dataset.indices],
            self.expert_fn(np.array(val_dataset.dataset.targets)[val_dataset.indices]),
            transform_test,
        )
        dataset_test = GenericImageExpertDataset(
            test_dataset.data,
            test_dataset.targets,
            self.expert_fn(test_dataset.targets),
            transform_test,
        )

        self.data_train_loader = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        self.data_val_loader = torch.utils.data.DataLoader(
            dataset=dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        self.data_test_loader = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )


# https://github.com/jcpeterson/cifar-10h
class HateSpeech(BaseDataset):
    """Hatespeech dataset from Davidson et al. 2017"""

    def __init__(
        self,
        data_dir,
        embed_texts,
        include_demographics,
        expert_type,
        device,
        synth_exp_param=[0.7, 0.7],
        test_split=0.2,
        val_split=0.1,
        batch_size=1000,
        transforms=None,
    ):
        """
        data_dir: where to save files for dataset (folder path)
        embed_texts (bool): whether to embedd the texts or raw text return
        include_demographics (bool): whether to include the demographics for each example, defined as either AA or not.
        if True, then the data loader will return a tuple of (data, label, expert_prediction, demographics)
        expert_type (str): either 'synthetic' which makes error depending on synth_exp_param for not AA or AA, or 'random_annotator' which defines human as random annotator
        synth_exp_param (list): list of length 2, first element is the probability of error for AA, second is for not AA
        test_split: percentage of test data
        val_split: percentage of data to be used for validation (from training set)
        batch_size: batch size for training
        transforms: data transforms
        """
        self.embed_texts = embed_texts
        self.include_demographics = include_demographics
        self.expert_type = expert_type
        self.synth_exp_param = synth_exp_param
        self.data_dir = data_dir
        self.device = device
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.n_dataset = 3  # number of classes in dataset
        self.train_split = 1 - test_split - val_split
        self.transforms = transforms
        self.generate_data()

    def generate_data(self):
        """
        generate data for training, validation and test sets

        """
        # download dataset if it doesn't exist
        if not os.path.exists(self.data_dir + "/hatespeech_labeled_data.csv"):
            logging.info("Downloading HateSpeech dataset")
            r = requests.get(
                "https://github.com/t-davidson/hate-speech-and-offensive-language/raw/master/data/labeled_data.csv",
                allow_redirects=True,
            )
            with open(self.data_dir + "/hatespeech_labeled_data.csv", "wb") as f:
                f.write(r.content)
            logging.info("Finished Downloading HateSpeech Data data")
            try:
                hatespeech_data = pd.read_csv(
                    self.data_dir + "/hatespeech_labeled_data.csv"
                )
            except:
                logging.error("Failed to load HateSpeech data")
                raise
        else:
            logging.info("Loading HateSpeech data")
            try:
                hatespeech_data = pd.read_csv(
                    self.data_dir + "/hatespeech_labeled_data.csv"
                )
            except:
                logging.error("Failed to load HateSpeech data")
                raise
            # download aae file
        if not os.path.exists(self.data_dir + "/model_count_table.txt"):
            logging.info("Downloading AAE detection")
            r = requests.get(
                "https://github.com/slanglab/twitteraae/raw/master/model/model_count_table.txt",
                allow_redirects=True,
            )
            with open(self.data_dir + "/model_count_table.txt", "wb") as f:
                f.write(r.content)
        if not os.path.exists(self.data_dir + "/model_vocab.txt"):
            logging.info("Downloading AAE detection")
            r = requests.get(
                "https://github.com/slanglab/twitteraae/raw/master/model/model_vocab.txt",
                allow_redirects=True,
            )
            with open(self.data_dir + "/model_vocab.txt", "wb") as f:
                f.write(r.content)
        self.model_file_path = self.data_dir + "/model_count_table.txt"
        self.vocab_file_path = self.data_dir + "/model_vocab.txt"
        self.model_aae = ModelPredictAAE(self.model_file_path, self.vocab_file_path)
        # predict demographics for the deata
        hatespeech_data["demographics"] = hatespeech_data["tweet"].apply(
            lambda x: self.model_aae.predict_lang(x)
        )

        self.label_to_category = {
            0: "hate_speech",
            1: "offensive_language",
            2: "neither",
        }
        # create a new column that creates a distribution over the labels
        distribution_over_labels = []
        for i in range(len(hatespeech_data)):
            label_counts = [
                hatespeech_data.iloc[i]["hate_speech"],
                hatespeech_data.iloc[i]["offensive_language"],
                hatespeech_data.iloc[i]["neither"],
            ]
            label_distribution = np.array(label_counts) / sum(label_counts)
            distribution_over_labels.append(label_distribution)
        hatespeech_data["label_distribution"] = distribution_over_labels
        human_prediction = []
        if self.expert_type == "synthetic":
            for i in range(len(hatespeech_data)):
                if hatespeech_data.iloc[i]["demographics"] == 0:
                    correct_human = np.random.choice(
                        [0, 1], p=[1 - self.synth_exp_param[0], self.synth_exp_param[0]]
                    )

                else:
                    correct_human = np.random.choice(
                        [0, 1], p=[1 - self.synth_exp_param[1], self.synth_exp_param[1]]
                    )
                if correct_human:
                    human_prediction.append(hatespeech_data.iloc[i]["class"])
                else:
                    human_prediction.append(np.random.choice([0, 1, 2]))
        else:
            for i in range(len(hatespeech_data)):
                # sample from label distribution
                label_distribution = hatespeech_data.iloc[i]["label_distribution"]
                label = np.random.choice([0, 1, 2], p=label_distribution)
                human_prediction.append(label)

        hatespeech_data["human_prediction"] = human_prediction

        train_x = hatespeech_data["tweet"].to_numpy()
        train_y = hatespeech_data["class"].to_numpy()
        train_h = hatespeech_data["human_prediction"].to_numpy()
        train_d = hatespeech_data["demographics"].to_numpy()
        random_seed = random.randrange(10000)

        if self.embed_texts:
            logging.info("Embedding texts")
            # TODO: cache the embeddings, so no need to regenerate them
            model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            embeddings = model.encode(train_x)
            train_x = np.array(embeddings)
            test_size = int(self.test_split * len(hatespeech_data))
            val_size = int(self.val_split * len(hatespeech_data))
            train_size = len(hatespeech_data) - test_size - val_size
            train_y = torch.tensor(train_y)
            train_h = torch.tensor(train_h)
            train_d = torch.tensor(train_d)
            train_x = torch.from_numpy(train_x).float()

            self.d = train_x.shape[1]
            train_x, val_x, test_x = torch.utils.data.random_split(
                train_x,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(random_seed),
            )
            train_y, val_y, test_y = torch.utils.data.random_split(
                train_y,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(random_seed),
            )
            train_h, val_h, test_h = torch.utils.data.random_split(
                train_h,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(random_seed),
            )

            data_train = torch.utils.data.TensorDataset(
                train_x.dataset[train_x.indices],
                train_y.dataset[train_y.indices],
                train_h.dataset[train_h.indices],
            )
            data_val = torch.utils.data.TensorDataset(
                val_x.dataset[val_x.indices],
                val_y.dataset[val_y.indices],
                val_h.dataset[val_h.indices],
            )
            data_test = torch.utils.data.TensorDataset(
                test_x.dataset[test_x.indices],
                test_y.dataset[test_y.indices],
                test_h.dataset[test_h.indices],
            )

            self.data_train_loader = torch.utils.data.DataLoader(
                data_train, batch_size=self.batch_size, shuffle=True
            )
            self.data_val_loader = torch.utils.data.DataLoader(
                data_val, batch_size=self.batch_size, shuffle=False
            )
            self.data_test_loader = torch.utils.data.DataLoader(
                data_test, batch_size=self.batch_size, shuffle=False
            )

        else:
            # NOT YET SUPPORTED, SPACY GIVES ERRORS
            # pytorch text loader
            self.text_field = data.Field(
                sequential=True, lower=True, include_lengths=True, batch_first=True
            )
            label_field = data.Field(
                sequential=False, use_vocab=False, batch_first=True
            )
            human_field = data.Field(
                sequential=False, use_vocab=False, batch_first=True
            )
            demographics_field = data.Field(
                sequential=False, use_vocab=False, batch_first=True
            )

            fields = [
                ("text", self.text_field),
                ("label", label_field),
                ("human", human_field),
            ]  # , ('demographics', self.demographics_field)]
            examples = [
                data.Example.fromlist([train_x[i], train_y[i], train_h[i]], fields)
                for i in range(train_x.shape[0])
            ]
            hatespeech_dataset = data.Dataset(examples, fields)
            self.text_field.build_vocab(
                hatespeech_dataset,
                min_freq=3,
                vectors="glove.6B.100d",
                unk_init=torch.Tensor.normal_,
                max_size=20000,
            )
            label_field.build_vocab(hatespeech_dataset)
            human_field.build_vocab(hatespeech_dataset)
            demographics_field.build_vocab(hatespeech_dataset)
            train_data, valid_data, test_data = hatespeech_dataset.split(
                split_ratio=[self.train_split, self.val_split, self.test_split],
                random_state=random.seed(random_seed),
            )
            (
                self.data_train_loader,
                self.data_val_loader,
                self.data_test_loader,
            ) = data.BucketIterator.splits(
                (train_data, valid_data, test_data),
                batch_size=self.batch_size,
                sort_key=lambda x: len(x.text),
                sort_within_batch=True,
                device=self.device,
            )

    def model_setting(self, model_nn):
        # build model
        INPUT_DIM = len(self.text_field.vocab)
        EMBEDDING_DIM = 100  # fixed
        PAD_IDX = self.text_field.vocab.stoi[self.text_field.pad_token]

        # model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
        # model = CNN_rej(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, 3, DROPOUT, PAD_IDX)

        pretrained_embeddings = self.text_field.vocab.vectors

        model_nn.embedding.weight.data.copy_(pretrained_embeddings)
        UNK_IDX = self.text_field.vocab.stoi[self.text_field.unk_token]

        model_nn.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model_nn.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
        return INPUT_DIM, EMBEDDING_DIM, PAD_IDX


class SyntheticData(BaseDataset):
    """Synthetic dataset introduced in our work"""

    def __init__(
        self,
        train_samples=1000,
        test_samples=1000,
        data_distribution="mix_of_guassians",
        d=10,
        mean_scale=1,
        expert_deferred_error=0,
        expert_nondeferred_error=0.5,
        machine_nondeferred_error=0,
        num_of_guassians=10,
        val_split=0.1,
        batch_size=1000,
        transforms=None,
    ):
        """

        total_samples: total number of samples in the dataset
        data_distribution: the distribution of the data. mix_of_guassians, or uniform
        d: dimension of the data
        mean_scale: the scale of the means of the guassians, or uniform
        expert_deferred_error: the error of the expert when the data is deferred
        expert_nondeferred_error: the error of the expert when the data is nondeferred
        machine_nondeferred_error: the error of the machine when the data is nondeferred
        num_of_guassians: the number of guassians in the mix of guassians
        """
        self.val_split = val_split
        self.batch_size = batch_size
        self.train_split = 1 - val_split
        self.transforms = transforms
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.total_samples = train_samples + test_samples
        self.data_distribution = data_distribution
        self.d = d
        self.n_dataset = 2
        self.mean_scale = mean_scale
        self.expert_deferred_error = expert_deferred_error
        self.expert_nondeferred_error = expert_nondeferred_error
        self.machine_nondeferred_error = machine_nondeferred_error
        self.num_of_guassians = num_of_guassians
        self.generate_data()

    def generate_data(self):
        if self.data_distribution == "uniform":
            data_x = torch.rand((self.total_samples, self.d)) * self.mean_scale
        else:
            mix = D.Categorical(
                torch.ones(
                    self.num_of_guassians,
                )
            )
            comp = D.Independent(
                D.Normal(
                    torch.randn(self.num_of_guassians, self.d),
                    torch.rand(self.num_of_guassians, self.d),
                ),
                1,
            )
            gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)
            data_x = gmm.sample((self.total_samples,)) * self.mean_scale
        # get random labels
        mean_rej_prop = 0
        # make sure ramdom rejector rejects between 20 and 80% of the time (variable)
        while not (mean_rej_prop >= 0.2 and mean_rej_prop <= 0.8):
            net_rej_opt = Linear_net_sig(self.d)
            with torch.no_grad():
                outputs = net_rej_opt(data_x)
                predicted = torch.round(outputs.data)
                mean_rej_prop = np.mean(
                    [predicted[i][0] for i in range(len(predicted))]
                )
                # get rejector preds on x
        opt_rej_preds = []
        with torch.no_grad():
            outputs = net_rej_opt(data_x)
            predicted = torch.round(outputs.data)
            opt_rej_preds = [predicted[i][0] for i in range(len(predicted))]
        # get classifier that is 1 at least 20% and at most 80% on non-deferred side
        mean_class_prop = 0
        net_mach_opt = Linear_net_sig(self.d)
        while not (mean_class_prop >= 0.2 and mean_class_prop <= 0.8):
            net_mach_opt = Linear_net_sig(self.d)
            with torch.no_grad():
                outputs = net_mach_opt(data_x)
                predicted = torch.round(outputs.data)
                predicted_class = [
                    predicted[i][0] * (1 - opt_rej_preds[i])
                    for i in range(len(predicted))
                ]
                mean_class_prop = np.sum(predicted_class) / (
                    len(opt_rej_preds) - np.sum(opt_rej_preds)
                )
        # get classifier preds on x
        opt_mach_preds = []
        with torch.no_grad():
            outputs = net_mach_opt(data_x)
            predicted = torch.round(outputs.data)
            opt_mach_preds = [predicted[i][0] for i in range(len(predicted))]

        # get random labels
        data_y = torch.randint(low=0, high=2, size=(self.total_samples,))
        # make labels consistent with net_mach_opt on non-deferred side with error specified
        for i in range(len(data_y)):
            if opt_rej_preds[i] == 0:
                coin = np.random.binomial(1, 1 - self.machine_nondeferred_error, 1)[0]
                if coin == 1:
                    data_y[i] = opt_mach_preds[i]

        # make expert 1-expert_deferred_error accurate on deferred side and 1-expert_nondeferred_error accurate otherwise
        human_predictions = [0] * len(data_y)
        for i in range(len(data_y)):
            if opt_rej_preds[i] == 1:
                coin = np.random.binomial(1, 1 - self.expert_deferred_error, 1)[0]
                if coin == 1:
                    human_predictions[i] = data_y[i]
                else:
                    human_predictions[i] = 1 - data_y[i]

            else:
                coin = np.random.binomial(1, 1 - self.expert_nondeferred_error, 1)[0]
                if coin == 1:
                    human_predictions[i] = data_y[i]
                else:
                    human_predictions[i] = 1 - data_y[i]

        human_predictions = torch.tensor(human_predictions)
        # split into train, val, test
        train_size = int(self.train_samples * self.train_split)
        val_size = int(self.train_samples * self.val_split)
        test_size = len(data_x) - train_size - val_size  # = self.test_samples

        self.train_x, self.val_x, self.test_x = torch.utils.data.random_split(
            data_x,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        self.train_y, self.val_y, self.test_y = torch.utils.data.random_split(
            data_y,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        self.train_h, self.val_h, self.test_h = torch.utils.data.random_split(
            human_predictions,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        logging.info("train size: ", len(self.train_x))
        logging.info("val size: ", len(self.val_x))
        logging.info("test size: ", len(self.test_x))
        self.data_train = torch.utils.data.TensorDataset(
            self.train_x.dataset.data[self.train_x.indices],
            self.train_y.dataset.data[self.train_y.indices],
            self.train_h.dataset.data[self.train_h.indices],
        )
        self.data_val = torch.utils.data.TensorDataset(
            self.val_x.dataset.data[self.val_x.indices],
            self.val_y.dataset.data[self.val_y.indices],
            self.val_h.dataset.data[self.val_h.indices],
        )
        self.data_test = torch.utils.data.TensorDataset(
            self.test_x.dataset.data[self.test_x.indices],
            self.test_y.dataset.data[self.test_y.indices],
            self.test_h.dataset.data[self.test_h.indices],
        )

        self.data_train_loader = torch.utils.data.DataLoader(
            self.data_train, batch_size=self.batch_size, shuffle=True
        )
        self.data_val_loader = torch.utils.data.DataLoader(
            self.data_val, batch_size=self.batch_size, shuffle=False
        )
        self.data_test_loader = torch.utils.data.DataLoader(
            self.data_test, batch_size=self.batch_size, shuffle=False
        )

        # double check if the solution we got is actually correct
        error_optimal_ = 0
        for i in range(len(data_y)):
            if opt_rej_preds[i] == 1:
                error_optimal_ += human_predictions[i] != data_y[i]
            else:
                error_optimal_ += opt_mach_preds[i] != data_y[i]
        error_optimal_ = error_optimal_ / len(data_y)
        self.error_optimal = error_optimal_
        logging.info(
            f"Data optimal: Accuracy Train {100 - 100 * error_optimal_:.3f} with rej {mean_rej_prop * 100} \n \n"
        )



class OracleSyntheticData(BaseDataset):
    """Synthetic dataset introduced in our work"""

    def __init__(
        self,
        train_samples=1000,
        test_samples=1000,
        data_distribution="mix_of_guassians",
        d=10,
        mean_scale=1,
        expert_deferred_error=0,
        expert_nondeferred_error=0.5,
        machine_nondeferred_error=0,
        num_of_guassians=10,
        val_split=0.1,
        batch_size=1000,
        transforms=None,
    ):
        """

        total_samples: total number of samples in the dataset
        data_distribution: the distribution of the data. mix_of_guassians, or uniform
        d: dimension of the data
        mean_scale: the scale of the means of the guassians, or uniform
        expert_deferred_error: the error of the expert when the data is deferred
        expert_nondeferred_error: the error of the expert when the data is nondeferred
        machine_nondeferred_error: the error of the machine when the data is nondeferred
        num_of_guassians: the number of guassians in the mix of guassians
        """
        self.val_split = val_split
        self.batch_size = batch_size
        self.train_split = 1 - val_split
        self.transforms = transforms
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.total_samples = train_samples + test_samples
        self.data_distribution = data_distribution
        self.d = d
        self.n_dataset = 2
        self.mean_scale = mean_scale
        self.expert_deferred_error = expert_deferred_error
        self.expert_nondeferred_error = expert_nondeferred_error
        self.machine_nondeferred_error = machine_nondeferred_error
        self.num_of_guassians = num_of_guassians
        self.generate_data()

    def generate_data(self):
        if self.data_distribution == "uniform":
            data_x = torch.rand((self.total_samples, self.d)) * self.mean_scale
        else:
            mix = D.Categorical(
                torch.ones(
                    self.num_of_guassians,
                )
            )
            comp = D.Independent(
                D.Normal(
                    torch.randn(self.num_of_guassians, self.d),
                    torch.rand(self.num_of_guassians, self.d),
                ),
                1,
            )
            gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)
            data_x = gmm.sample((self.total_samples,)) * self.mean_scale
        # get random labels
        mean_rej_prop = 0
        # make sure ramdom rejector rejects between 20 and 80% of the time (variable)
        while not (mean_rej_prop >= 0.2 and mean_rej_prop <= 0.8):
            self.net_rej_opt = Linear_net_sig(self.d)
            with torch.no_grad():
                outputs = self.net_rej_opt(data_x)
                predicted = torch.round(outputs.data)
                mean_rej_prop = np.mean(
                    [predicted[i][0] for i in range(len(predicted))]
                )
                # get rejector preds on x
        opt_rej_preds = []
        with torch.no_grad():
            outputs = self.net_rej_opt(data_x)
            predicted = torch.round(outputs.data)
            opt_rej_preds = [predicted[i][0] for i in range(len(predicted))]
        # get classifier that is 1 at least 20% and at most 80% on non-deferred side
        mean_class_prop = 0
        self.net_mach_opt = Linear_net_sig(self.d)
        while not (mean_class_prop >= 0.2 and mean_class_prop <= 0.8):
            self.net_mach_opt = Linear_net_sig(self.d)
            with torch.no_grad():
                outputs = self.net_mach_opt(data_x)
                predicted = torch.round(outputs.data)
                predicted_class = [
                    predicted[i][0] * (1 - opt_rej_preds[i])
                    for i in range(len(predicted))
                ]
                mean_class_prop = np.sum(predicted_class) / (
                    len(opt_rej_preds) - np.sum(opt_rej_preds)
                )
        # get classifier preds on x
        opt_mach_preds = []
        with torch.no_grad():
            outputs = self.net_mach_opt(data_x)
            predicted = torch.round(outputs.data)
            opt_mach_preds = [predicted[i][0] for i in range(len(predicted))]

        # get random labels
        data_y = torch.randint(low=0, high=2, size=(self.total_samples,))
        # make labels consistent with net_mach_opt on non-deferred side with error specified
        for i in range(len(data_y)):
            if opt_rej_preds[i] == 0:
                coin = np.random.binomial(1, 1 - self.machine_nondeferred_error, 1)[0]
                if coin == 1:
                    data_y[i] = opt_mach_preds[i]

        # make expert 1-expert_deferred_error accurate on deferred side and 1-expert_nondeferred_error accurate otherwise
        human_predictions = [0] * len(data_y)
        for i in range(len(data_y)):
            if opt_rej_preds[i] == 1:
                coin = np.random.binomial(1, 1 - self.expert_deferred_error, 1)[0]
                if coin == 1:
                    human_predictions[i] = data_y[i]
                else:
                    human_predictions[i] = 1 - data_y[i]

            else:
                coin = np.random.binomial(1, 1 - self.expert_nondeferred_error, 1)[0]
                if coin == 1:
                    human_predictions[i] = data_y[i]
                else:
                    human_predictions[i] = 1 - data_y[i]

        human_predictions = torch.tensor(human_predictions)
        # split into train, val, test
        train_size = int(self.train_samples * self.train_split)
        val_size = int(self.train_samples * self.val_split)
        test_size = len(data_x) - train_size - val_size  # = self.test_samples

        self.train_x, self.val_x, self.test_x = torch.utils.data.random_split(
            data_x,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        self.train_y, self.val_y, self.test_y = torch.utils.data.random_split(
            data_y,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        self.train_h, self.val_h, self.test_h = torch.utils.data.random_split(
            human_predictions,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        logging.info("train size: ", len(self.train_x))
        logging.info("val size: ", len(self.val_x))
        logging.info("test size: ", len(self.test_x))
        self.data_train = torch.utils.data.TensorDataset(
            self.train_x.dataset.data[self.train_x.indices],
            self.train_y.dataset.data[self.train_y.indices],
            self.train_h.dataset.data[self.train_h.indices],
        )
        self.data_val = torch.utils.data.TensorDataset(
            self.val_x.dataset.data[self.val_x.indices],
            self.val_y.dataset.data[self.val_y.indices],
            self.val_h.dataset.data[self.val_h.indices],
        )
        self.data_test = torch.utils.data.TensorDataset(
            self.test_x.dataset.data[self.test_x.indices],
            self.test_y.dataset.data[self.test_y.indices],
            self.test_h.dataset.data[self.test_h.indices],
        )

        self.data_train_loader = torch.utils.data.DataLoader(
            self.data_train, batch_size=self.batch_size, shuffle=True
        )
        self.data_val_loader = torch.utils.data.DataLoader(
            self.data_val, batch_size=self.batch_size, shuffle=False
        )
        self.data_test_loader = torch.utils.data.DataLoader(
            self.data_test, batch_size=self.batch_size, shuffle=False
        )

        # double check if the solution we got is actually correct
        error_optimal_ = 0
        for i in range(len(data_y)):
            if opt_rej_preds[i] == 1:
                error_optimal_ += human_predictions[i] != data_y[i]
            else:
                error_optimal_ += opt_mach_preds[i] != data_y[i]
        error_optimal_ = error_optimal_ / len(data_y)
        self.error_optimal = error_optimal_
        logging.info(
            f"Data optimal: Accuracy Train {100 - 100 * error_optimal_:.3f} with rej {mean_rej_prop * 100} \n \n"
        )





# https://osf.io/2ntrf/
# https://www.pnas.org/doi/10.1073/pnas.2111547119


class GalaxyZoo(BaseDataset):
    def __init__(
        self,
        data_dir,
        test_split=0.2,
        val_split=0.1,
        batch_size=1000,
        transforms=None,
    ):
        """
        Must go to  https://osf.io/2ntrf/ , click on OSF Storage, download zip, unzip it, and write the path of the folder in data_dir
        data_dir: where to save files for model
        noise_version: noise version to use from 080,095, 110,125 (From imagenet16h paper)
        use_data_aug: whether to use data augmentation (bool)
        test_split: percentage of test data
        val_split: percentage of data to be used for validation (from training set)
        batch_size: batch size for training
        transforms: data transforms
        """
        self.data_dir = data_dir
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.n_dataset = 2
        self.train_split = 1 - test_split - val_split
        self.transforms = transforms
        self.d = 1024
        self.generate_data()

    def generate_data(self):
        """
        generate data for training, validation and test sets

        """
        # check if the folder data_dir has everything we need

        if not os.path.exists(self.data_dir + "/galaxyzoo/training_solutions_rev1.csv"):
            raise ValueError(
                "cant find csv, Please download the data from https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/overview"
                " , unzip it, and construct the path of the folder in data_dir"
            )
        if not os.path.exists(self.data_dir + "/galaxyzoo/images_training_rev1"):
            raise ValueError(
                "cant find csv, Please download the data from https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/overview"
                " , unzip it, and construct the path of the folder in data_dir"
            )

        # load the csv file
        df = pd.read_csv(self.data_dir + "/galaxyzoo/training_solutions_rev1.csv")
        # split the task b/w smooth galaxy vs non-smooth galaxy and star
        df["Class1.23"] = df["Class1.2"] + df["Class1.3"]
        # set the target label as the majority of the experts
        df["TARGET"] = np.argmax(df[["Class1.1", "Class1.23"]], axis=1)
        # get a random human prediction from the 30 annotators
        df["hum_pred"] = df["Class1.1"].transform(
            lambda x: random.choices([0, 1], weights=[x, 1 - x])[0]
        )
        # get a 10k sample as in Okati et al.
        random_seed = random.randrange(10000)
        df = (
            df[["GalaxyID", "TARGET", "hum_pred"]]
            .sample(n=10000, random_state=random_seed)
            .reset_index(drop=True)
            .copy()
        )
        print(df["TARGET"].mean())
        print(df["hum_pred"].mean())
        print(np.unique(df["TARGET"], return_counts=True))
        # get unique categories
        categories = df["TARGET"].unique()
        # get mapping from category to index
        imagenames_categories = dict(zip(df["GalaxyID"].astype(str), df["TARGET"]))
        # for each image name, get the random human prediction
        image_name_to_single_participant_classification = {}
        for image_name in df["GalaxyID"].unique():
            image_name_to_single_participant_classification[str(image_name)] = df[
                df["GalaxyID"] == image_name
            ]["hum_pred"].values[0]
        # remove png extension
        image_names = df["GalaxyID"].astype(str).to_list()
        image_paths = np.array(
            ["data/galaxyzoo/images_training_rev1/" + x + ".jpg" for x in image_names]
        )
        # get label for image names
        image_names_labels = np.array([imagenames_categories[x] for x in image_names])
        # get prediction for image names
        image_names_human_predictions = np.array(
            [image_name_to_single_participant_classification[x] for x in image_names]
        )

        transform_train = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        test_size = int(self.test_split * len(image_paths))
        val_size = int(self.val_split * len(image_paths))
        train_size = len(image_paths) - test_size - val_size

        train_x, val_x, test_x = torch.utils.data.random_split(
            image_paths,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_y, val_y, test_y = torch.utils.data.random_split(
            image_names_labels,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_h, val_h, test_h = torch.utils.data.random_split(
            image_names_human_predictions,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )

        data_train = GenericImageExpertDataset(
            train_x.dataset[train_x.indices],
            train_y.dataset[train_y.indices],
            train_h.dataset[train_h.indices],
            transform_train,
            to_open=True,
        )
        data_val = GenericImageExpertDataset(
            val_x.dataset[val_x.indices],
            val_y.dataset[val_y.indices],
            val_h.dataset[val_h.indices],
            transform_test,
            to_open=True,
        )
        data_test = GenericImageExpertDataset(
            test_x.dataset[test_x.indices],
            test_y.dataset[test_y.indices],
            test_h.dataset[test_h.indices],
            transform_test,
            to_open=True,
        )

        self.data_train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=32,
            pin_memory=True,
        )
        self.data_val_loader = torch.utils.data.DataLoader(
            data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        self.data_test_loader = torch.utils.data.DataLoader(
            data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )


