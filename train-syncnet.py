import os
from models import SyncNet_color as SyncNet
from models.Dataloader_syncnet import Data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import wandb
from torch import optim

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def train(device, model, train_data_loader, test_data_loader, optimizer,
          epochs = 2000000000, syncnet_eval_interval = 100):
    global_step = 1
    for i in range(epochs):
        for _, (x, spec, y) in enumerate(train_data_loader):
            #print(global_step)
            model.train()
            optimizer.zero_grad()

            x = x.to(device)

            spec = spec.to(device)

            a, v = model(spec, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()

            prev_loss = -1
            if global_step % syncnet_eval_interval == 0:
                with torch.no_grad():
                    curr_loss = eval_model(test_data_loader, device, model)
                if curr_loss<=prev_loss or prev_loss==-1:
                    torch.save(model.state_dict(), f'./checkpoints/syncnet{global_step}.pth')
                prev_loss = curr_loss
            wandb.log({'Train': loss.item()})
            global_step+=1

def eval_model(test_data_loader, device, model):
    losses = []
    for _, (x, spec, y) in enumerate(test_data_loader):

        model.eval()

        # Transform data to CUDA device
        x = x.to(device)

        spec = spec.to(device)

        a, v = model(spec, x)
        y = y.to(device)

        loss = cosine_loss(a, v, y)
        losses.append(loss.item())

    averaged_loss = sum(losses) / len(losses)
    wandb.log({'Eval':averaged_loss})
    return averaged_loss


if __name__ == '__main__':
    train_dataset = Data('./preprocessfps25/train')
    train_dataloader = DataLoader(train_dataset, batch_size=64, drop_last=False, pin_memory=True, shuffle=True, num_workers=4)

    test_dataset = Data('./preprocessfps25/eval')
    test_dataloader = DataLoader(test_dataset, batch_size=64, pin_memory=True, drop_last=False, shuffle=False, num_workers=4)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = SyncNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    
    wandb.init(project = 'syncnet')
    train(device, model, train_dataloader, test_dataloader, optimizer)
