from torchvision import datasets, transforms, models
import autoaugment
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import Accuracy
import tqdm
from model import resnet18, resnet34, resnet50
import wandb
import cfg
import time
import json
import logging
import os
import shutil
import torch
from emotion import Emotion
import numpy as np
import scipy.misc

from io import BytesIO  # Python 3.x


def load_model(model, model_filepath, device):
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    return model


def save_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def log_metrics(acc, loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "train_loss": loss,
              "train_acc": acc}, step=example_ct)


def log_metrics_val(metrics):
    wandb.log(metrics)


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def get_model_accuracy(eval_data, model_path):
    print(eval_data)
    test_data, num_classes = load_data(eval_data, train=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, num_workers=12, shuffle=False,
                                              pin_memory=False)

    model = create_model(eval_data, num_classes)
    assert os.path.exists(
        model_path), 'Expected model in path: {}'.format(model_path)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    val_metrics = evaluate(model, test_loader)
    print(f"{model_path.split('/')[-1]} : {val_metrics}")
    json_path = model_path[:-3] + '.json'
    with open(json_path, 'w') as f:
        json.dump(val_metrics, f)

    return float(val_metrics['acc'] / 100)

############################## TRAINING/EVAL ##############################


def evaluate(model, data_loader, loss_fn=F.cross_entropy, device=torch.device("cuda:0"), suffix="", *args):
    model.eval()
    model.to(device)

    # summary for current eval loop
    summ = []
    print("Validation")

    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in data_loader:
            # if params.cuda:
            data_batch = data_batch.to(device)         # (B,3,32,32)
            labels_batch = labels_batch.to(device)     # (B,)

            # compute model output
            output_batch = model(data_batch)
            # print(output_batch, labels_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            # calculate accuracy
            output_batch = np.argmax(output_batch, axis=1)
            acc = 100.0 * np.sum(output_batch == labels_batch) / \
                float(labels_batch.shape[0])

            summary_batch = {f'val_acc{suffix}': acc.item(
            ), f'val_loss{suffix}': loss.item()}
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                    for x in summ]) for metric in summ[0]}
    return metrics_mean


def get_file_size(file_path):
    """ Get file in size in MB"""
    size = os.path.getsize(file_path)
    return size / (1024*1024)


def create_model(dataset_name, num_classes, arch='resnet'):
    dataset_name = dataset_name.split('_')[0]
    if dataset_name in ['emotion']:
        if 'resnet34' in arch:
            model = resnet34(num_classes=num_classes).cuda()
        elif 'resnet18' in arch:
            model = resnet18(num_classes=num_classes).cuda()
        elif 'resnet50' in arch:
            model = resnet50(num_classes=num_classes).cuda()
        else:
            raise ValueError('{} is an invalid architecture!'.format(arch))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model


def cross_entropy_loss(logits, distill_target, gt_target, temperature):
    """
    :param logits: tensor of shape (N, C); the predicted logits
    :param gt_target: long tensor of shape (N,); the gt class labels
    :returns: cross entropy loss
    """
    return F.cross_entropy(logits, gt_target, reduction='mean')


############################## DATA LOADING ##############################

def load_data(mode='train', seed=42):
    if mode == 'train' or mode == 'val':
        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(p=0.25),

            transforms.RandomChoice([
                transforms.AutoAugment(),
                autoaugment.RandAugment(),
                autoaugment.AugMix(),
            ]),
            transforms.RandomChoice([
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.GaussianBlur(kernel_size=5),
                transforms.RandomAffine(
                    degrees=(0, 360), translate=(0.1, 0.3)),
                transforms.RandomPerspective(),
            ]),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])

    num_classes = 7
    dataset = Emotion(cfg.DATASET_ROOT, train=mode, transform=transform)

    return dataset, num_classes
