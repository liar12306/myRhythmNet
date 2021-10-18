import os
import config
import glob
import torch
import numpy as np
import pandas as pd
import engine_vipl
from torch.utils.data import DataLoader
from src.utils.loadData import RhythmNetDataSet
from torch.nn import L1Loss
from src.utils.plot_scripts import plot_train_test_curves, bland_altman_plot, gt_vs_est, create_plot_for_tensorboard
from src.utils.model_utils import plot_loss, load_model_if_checkpointed, save_model_checkpoint

from src.models.rhythmNet import RhythmNet
from src.models.resNet import ResNet18
import time
from loss_func.rhythmnet_loss import RhythmNetLoss


# Needed in VIPL dataset where each data item has a different number of frames/maps
def collate_fn(batch):
    batched_st_map, batched_targets = [], []
    # for data in batch:
    #     batched_st_map.append(data["st_maps"])
    #     batched_targets.append(data["target"])
    # # torch.stack(batched_output_per_clip, dim=0).transpose_(0, 1)
    return batch


def hr_error(ground_true, predict):
    return abs(predict - ground_true)


def rmse(loss):
    return np.sqrt(np.mean(loss ** 2))


def mae(loss):
    return np.mean(loss)


def mer(ground_true, loss):
    return np.mean(loss / ground_true) * 100


def std(loss, hr_mae):
    return np.sqrt(np.mean((loss - hr_mae) ** 2))

def r(ground_true,predict):
    g = ground_true - np.mean(ground_true)
    p = predict-np.mean(predict)
    return np.sum(g*p)/(np.sqrt(np.sum(g**2))*np.sqrt(np.sum(p**2)))


def compute_criteria(target_hr_list, predicted_hr_list):
    hr_loss = hr_error(target_hr_list, predicted_hr_list)
    hr_mae = mae(hr_loss)
    hr_rmse = rmse(hr_loss)
    hr_mer = mer(target_hr_list, hr_loss)
    hr_std = std(hr_loss, hr_mae)
    pearson = r(target_hr_list, predicted_hr_list)


    return {"MAE": hr_mae, "RMSE": hr_rmse, "STD": hr_std, "MER": hr_mer,"r":pearson}


def load_data(data_list_path):
    data_roots = pd.read_csv(data_list_path)["root"].values
    test_data_roots = data_roots[:300]
    train_data_roots = data_roots[300:]
    train_set = RhythmNetDataSet(train_data_roots)
    test_set = RhythmNetDataSet(train_data_roots)
    return train_set, test_set


def run_training():
    net = ResNet18()

    net.to(config.DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.8,
        patience=5,
        verbose=True
    )
    lossFunction = RhythmNetLoss()
    loss_function = L1Loss()
    data_list_path = config.PROJECT_ROOT + config.DATA_ROOT + "rhythmnet_data.csv"

    train_set, test_set = load_data(data_list_path)


    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_fn
    )
    train_loss_per_epoch = []

    for epoch in range(config.EPOCHS):
        start_time = time.time()
        target_hr_list, predicted_hr_list, train_loss = engine_vipl.train_NG(net, train_loader, optimizer,loss_function)
        end_time = time.time()
        cost_time = int(end_time - start_time)
        m = cost_time // 60
        s = cost_time % 60
        metrics = compute_criteria(target_hr_list, predicted_hr_list)
        print(f"\nFinished [Epoch: {epoch + 1}/{config.EPOCHS}]",
              "\nTraining Loss: {:.3f} |".format(train_loss),
              "MAE : {:.3f} |".format(metrics["MAE"]),
              "RMSE : {:.3f} |".format(metrics["RMSE"]),
              "STD : {:.3f} |".format(metrics["STD"]),
              "MER : {:.3f}% |".format(metrics["MER"]),
              "r : {:.3f} |".format(metrics["r"]),
              "time: {}:{} s".format(m, s))
        # "Pearsonr : {:.3f} |".format(metrics["Pearson"]), )
        train_loss_per_epoch.append(train_loss)

    mean_loss = np.mean(train_loss_per_epoch)
    print(f"Avg Training Loss: {np.mean(mean_loss)} for {config.EPOCHS} epochs")

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config.BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_fn
    )

    target_hr_list, predicted_hr_list, test_loss = engine_vipl.eval_NG(net, test_loader,loss_function)
    metrics = compute_criteria(target_hr_list, predicted_hr_list)
    print("\nTest Loss: {:.3f} |".format(test_loss),
          "MAE : {:.3f} |".format(metrics["MAE"]),
          "RMSE : {:.3f} |".format(metrics["RMSE"]),
          "STD : {:.3f} |".format(metrics["STD"]),
          "MER : {:.3f}% |".format(metrics["MER"]),
          "r : {:.3f} |".format(metrics["r"])
    )




if __name__ == '__main__':
    run_training()

