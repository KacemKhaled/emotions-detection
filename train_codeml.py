from threading import currentThread
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import tqdm
# from tqdm import tqdm
from torch.backends import cudnn
import pickle
import argparse
from utils import *
from torchmetrics import Accuracy
import logging
import os
import wandb
import cfg
import pandas as pd


logging.basicConfig(level=logging.INFO)
cudnn.benchmark = True


class ParamsCfg():
    def __init__(self, cuda, learning_rate, num_epochs) -> None:
        self.cuda = cuda
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs


############################## DATASET AND TRAINING CODE ##################################
def train(model, loader, test_loader, optimizer, scheduler, loss_fn, save_path, num_epochs=50, print_every=100):
    """
    Trains the provided model

    :param model: the student model to train with distillation
    :param loader: the data loader for distillation; the dataset was created with make_distillation_dataset
    :param loss_fn: the loss function to use for distillation
    :param num_epochs: the number of epochs to train for
    """
    loss_avg = RunningAverage()
    acc_avg = RunningAverage()
    examples_ct = 0

    best_val = 0
    best_model = model

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        # {'acc':acc, 'loss':loss}
        val_metrics = evaluate(model, test_loader)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in val_metrics.items())
        if val_metrics['val_acc'] > best_val:
            best_model = model
            best_val = val_metrics['val_acc']
            # save checkppoint
            ckpt_path = args.save_path.replace('.pt', '_ckpt.pt')
            print('\n\nSaving checkpoint model to: {}\n'.format(ckpt_path))
            torch.save(best_model.state_dict(), ckpt_path)
        print(f"Epoch : {epoch} \n\t- Eval metrics : {metrics_string}")
        log_metrics_val(val_metrics)
        wandb.log({'epoch': epoch})
        model.train()
        # with tqdm.tqdm(total=len(loader), miniters=int(len(loader)/1000)) as t:
        for i, (bx, by) in enumerate(loader):
            bx = bx.cuda()
            by = by.cuda()

            # forward pass
            logits = model(bx)
            loss = loss_fn(logits, by)

            acc = 100.0 * np.sum(np.argmax(logits.detach().cpu().numpy(), axis=1)
                                 == by.cpu().numpy()) / float(by.cpu().numpy().shape[0])

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # update the average loss
            loss_avg.update(loss.item())
            acc_avg.update(acc.item())

            examples_ct += len(bx)
            if i % print_every == 0:
                # print(i, loss.item())
                log_metrics(acc, loss, examples_ct, epoch)

    val_metrics = evaluate(model,  test_loader)     # {'acc':acc, 'loss':loss}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in val_metrics.items())
    print("- Eval metrics : " + metrics_string)
    log_metrics_val(val_metrics)
    model.eval()


################################################################

def cross_entropy_loss(logits, gt_target):
    """
    :param logits: tensor of shape (N, C); the predicted logits
    :param gt_target: long tensor of shape (N,); the gt class labels
    :returns: cross entropy loss
    """
    return F.cross_entropy(logits, gt_target, reduction='mean')


def predict(dataset, mymodel, batch_size=64, num_workers=4, device=torch.device("cuda:0")):
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                         pin_memory=True)

    mymodel_preds = []
    mymodel.eval()
    mymodel.to(device)

    for (bx, by) in tqdm.tqdm(loader, mininterval=1.0):
        # bx = bx.cuda()
        bx = bx.to(device)
        with torch.no_grad():
            mymodel_logits = mymodel(bx)
            mymodel_pred = torch.softmax(mymodel_logits, dim=1)

            _, preds = torch.max(mymodel_pred, 1)
            mymodel_preds.append(preds)

    mymodel_preds = torch.cat(mymodel_preds, dim=0).cpu().numpy()

    return mymodel_preds


def main(args):
    run = wandb.init(
        project=cfg.WB_PROJECT,
        entity=cfg.WB_ENTITY,
        name=args.exp_name,
        tags=['mymodel', args.dataset],
        config=args, settings=wandb.Settings(start_method="fork"))

    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_filepath = args.save_path

    train_data, _ = load_data(mode='train')

    val_data, num_classes = load_data(mode='val')
    test_data, num_classes = load_data(mode='test')

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=cfg.NUM_WORKERS, shuffle=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, num_workers=cfg.NUM_WORKERS, shuffle=False, pin_memory=False)

    loss = cross_entropy_loss

    ################# TRAINING ####################
    print('\nTraining model on: {}'.format(args.dataset))

    mymodel = create_model(args.dataset, num_classes, arch=args.arch)
    num_epochs = args.epochs
    lr = args.lr
    optimizer = torch.optim.SGD(mymodel.parameters(
    ), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs*len(train_loader))  # //2

    if not args.prediction_only:
        train(mymodel, train_loader, test_loader, optimizer, scheduler,
              loss, num_epochs=num_epochs, save_path=args.save_path)
        print('\n\nDone! Saving model to: {}\n'.format(args.save_path))
        torch.save(mymodel.state_dict(), args.save_path)

    # # Load a model from path.
    mymodel = load_model(
        model=mymodel, model_filepath=args.save_path, device=cuda_device)
    print(f"Model exists, loading the model {args.save_path}")
    mymodel.eval()
    test_data, num_classes = load_data(mode='test')
    preds = predict(test_data, mymodel)
    idx_to_cat = {0: 'angry', 1: 'sad', 2: 'disgusted',
                  3: 'neutral', 4: 'fearful', 5: 'happy', 6: 'surprised'}
    print(preds)
    pred2 = pd.DataFrame(
        np.vstack([test_data.data.values[:, 0], [idx_to_cat[x] for x in preds]])).T
    path_prediction = args.save_path.replace('.pt', '.csv')
    pred2.to_csv(path_prediction, index=False, header=['ID', 'label'])
    table = wandb.Table(dataframe=pred2)
    table_artifact = wandb.Artifact("prediction_artifact", type="dataset")
    table_artifact.add(table, "table")
    # Log the raw csv file within an artifact to preserve our data
    table_artifact.add_file(path_prediction)
    # Log the table to visualize with a run...
    run.log({"table": table})

    # and Log as an Artifact to increase the available row limit!
    run.log_artifact(table_artifact)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN')
    parser.add_argument('--dataset', type=str, help='dataset')
    parser.add_argument('--save_path', type=str, help='path for saving model')
    parser.add_argument('--num_gpus', type=int,
                        help='number of GPUs for training', default=1)
    parser.add_argument('--exp_name', type=str,
                        help='name for exepriment', default='train')
    parser.add_argument('--epochs', type=int,
                        help='number of epochs for training', default=10)
    parser.add_argument('--lr', type=float,
                        help='learning rate for training', default=0.05)
    parser.add_argument('--batch_size', type=int,
                        help='batch size for training', default=32)
    parser.add_argument('--arch', type=str,
                        help='architecture for the model', default='resnet')
    parser.add_argument('--prediction_only', action='store_true',
                        help='prediction only', default=False)

    args = parser.parse_args()
    print(args)

    main(args)
