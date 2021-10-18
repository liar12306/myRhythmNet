from tqdm import tqdm
import torch
import config
import numpy as np
from torch import nn
from utils.model_utils import save_model_checkpoint

# have not GRU
def train_NG(model, data_loader, optimizer,loss_function):
    model.train()
    fin_loss = 0.0

    target_hr_list = []
    predicted_hr_list = []

    for batch in data_loader:
        for data in batch:
            maps = data["st_maps"].to(config.DEVICE)
            labels = data["target"].to(config.DEVICE)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(maps)
                loss = loss_function(outputs,labels)
                loss.backward()
                optimizer.step()
            target_hr_list.append(labels.mean().item())
            predicted_hr_list.append(outputs.mean().item())
            fin_loss += loss.item()
    return np.array(target_hr_list),np.array(predicted_hr_list),fin_loss/(len(data_loader)*config.BATCH_SIZE)

def eval_NG(model, data_loader,loss_function):
    model.train()
    fin_loss = 0.0

    target_hr_list = []
    predicted_hr_list = []

    for batch in data_loader:
        for data in batch:
            maps = data["st_maps"].to(config.DEVICE)
            labels = data["target"].to(config.DEVICE)
            outputs = model(maps)
            loss = loss_function(outputs,labels)
            target_hr_list.append(labels.mean().item())
            predicted_hr_list.append(outputs.mean().item())
            fin_loss += loss.item()
    return np.array(target_hr_list),np.array(predicted_hr_list),fin_loss/(len(data_loader)*config.BATCH_SIZE)


# with GRU
def train_WG(model, data_loader, optimizer, loss_fn):
    model.train()
    fin_loss = 0
    loss = 0.0

    target_hr_list = []
    predicted_hr_list = []
    batched_data = []
    for batch in data_loader:
        for data in batch:
            # an item of the data is available as a dictionary
            for (key, value) in data.items():
                data[key] = value.to(config.DEVICE)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs, gru_outputs = model(data["st_maps"])
                # w/o GRU
                # loss = loss_fn(outputs.squeeze(0), data["target"])
                loss = loss_fn(outputs.squeeze(0), gru_outputs, data["target"])
                loss.backward()
                optimizer.step()
            # "For each face video, the avg of all HR (bpm) of individual clips are computed as the final HR result
            # target_hr_batch = list(data["target"].mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
            target_hr_list.append(data["target"].mean().item())

            # predicted_hr_batch = list(outputs.squeeze(2).mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
            predicted_hr_list.append(outputs.squeeze(0).mean().item())
            fin_loss += loss.item()

    return np.array(target_hr_list), np.array(predicted_hr_list), fin_loss / (len(data_loader)*config.BATCH_SIZE)


def eval_WG(model, data_loader,loss_fn):
    model.eval()
    fin_loss = 0
    target_hr_list = []
    predicted_hr_list = []
    with torch.no_grad():
        for batch in data_loader:
            for data in batch:
                for (key, value) in data.items():
                    data[key] = value.to(config.DEVICE)

                # with torch.set_grad_enabled(False):
                outputs, gru_outputs = model(**data)
                # loss w/o GRU
                # loss = loss_fn(outputs.squeeze(0), data["target"])
                # loss with GRU
                loss = loss_fn(outputs.squeeze(0), gru_outputs, data["target"])
                fin_loss += loss.item()
                # target_hr_batch = list(data["target"].mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
                target_hr_list.append(data["target"].mean().item())

                # predicted_hr_batch = list(outputs.squeeze(2).mean(dim=1, keepdim=True).squeeze(1).detach().cpu().numpy())
                predicted_hr_list.append(outputs.squeeze(0).mean().item())

        return target_hr_list, predicted_hr_list, fin_loss / (len(data_loader)*config.BATCH_SIZE)
