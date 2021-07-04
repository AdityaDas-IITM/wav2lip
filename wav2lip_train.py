from models import SyncNet_color as SyncNet
from models import Wav2Lip as Wav2Lip, Wav2Lip_disc_qual
from models.Dataloader_wav2lip import Data
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from torch.utils.data import DataLoader
import random
import os
import wandb
import cv2

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    d = torch.clamp(d, 0.00000001, 0.99999999)
    loss = logloss(d.unsqueeze(1), y)
    return loss

recon_loss = nn.L1Loss()

def r1_reg(logits, inp):
    batch_size = inp.size(0)
    grad_dout = torch.autograd.grad(outputs=logits.sum(), inputs=inp, create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout = grad_dout.pow(2)
    reg = 0.5*grad_dout.view(batch_size, -1).sum(1).mean(0)
    return reg
     
syncnet = SyncNet().to('cuda' if torch.cuda.is_available() else 'cpu')
syncnet.load_state_dict(torch.load('./checkpoints/syncnet_best.pth'))
for p in syncnet.parameters():
    p.requires_grad = False
def get_sync_loss(mel, g, syncnet_T):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

syncnet_wt = 0.3
disc_wt = 0.07

def train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer, checkpoint_interval=500, epochs=2000000000):
    global_step = 0
    global syncnet_wt, disc_wt

    for _ in range(epochs):
        for step, (x, indiv_mels, mel, gt) in enumerate(train_data_loader):
            running_sync_loss, running_l1_loss, disc_loss, running_perceptual_loss = 0., 0., 0., 0.
            running_disc_real_loss, running_disc_fake_loss = 0., 0.

            disc.train()
            model.train()

            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            ### Train generator now. Remove ALL grads. 
            optimizer.zero_grad()
            disc_optimizer.zero_grad()

            g = model(indiv_mels, x)
            if syncnet_wt != 0:
                sync_loss = get_sync_loss(mel, g, 5)
            else:
                sync_loss = 0

            perceptual_loss = disc.perceptual_forward(g)

            l1loss = recon_loss(g, gt)

            loss = syncnet_wt * sync_loss + disc_wt * perceptual_loss + \
                                    (1. - syncnet_wt - disc_wt) * l1loss

            loss.backward()
            optimizer.step()

            ### Remove all gradients before Training disc
            disc_optimizer.zero_grad()

            pred1 = disc(gt)
            disc_real_loss = F.binary_cross_entropy_with_logits(pred1, torch.ones((len(pred1), 1)).to(device))
            
            g2 = g.detach()
            g2.requires_grad_()
            pred2 = disc(g2)
            disc_fake_loss = F.binary_cross_entropy_with_logits(pred2, torch.zeros((len(pred2), 1)).to(device))
            reg_loss = r1_reg(pred2, g2)
            disc_loss = disc_real_loss + disc_fake_loss + 0.07*reg_loss

            disc_loss.backward()
            disc_optimizer.step()

            running_disc_real_loss += disc_real_loss.item()
            running_disc_fake_loss += disc_fake_loss.item()

            running_l1_loss += l1loss.item()
            running_perceptual_loss += perceptual_loss.item()
            if syncnet_wt != 0:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss+=0

            if global_step == 1 or global_step % checkpoint_interval == 0:
                torch.save(model.state_dict(), f'./checkpoints/wav2lip{global_step}.pth')

            if global_step % checkpoint_interval == 0:
                save_imgs(x, g, gt, global_step)
                with torch.no_grad():
                    average_sync_loss = eval_model(test_data_loader, global_step, device, model, disc)

                    if average_sync_loss < .75:
                        syncnet_wt = 0.03
                
            global_step += 1
            wandb.log({'train_l1':l1loss, 'train_sync':sync_loss, 'train_preceptual':perceptual_loss})
            
def save_imgs(x, g, gt, step): # x = Bx6xTxHxW, g = Bx3xTxHxW, gt = Bx3xTxHxW
    x = (x[:5].detach().cpu().numpy().transpose(0,2,3,4,1)*255).astype(np.uint8)
    g = (g[:5].detach().cpu().numpy().transpose(0,2,3,4,1)*255).astype(np.uint8)
    gt = (gt[:5].detach().cpu().numpy().transpose(0,2,3,4,1)*255).astype(np.uint8)
    roll1, roll2 = x[:,:,:,:,:3], x[:,:,:,:,3:]
    #shapes = BxTxHxWx3
    collage = np.concatenate([roll1, roll2, gt, g], axis = 2)
    collage = np.concatenate([collage[:,i] for i in range(collage.shape[1])], axis = 2)
    for i in range(5):
        img = collage[i]
        cv2.imwrite(f'./outputs/{i}.jpg', img)
    
    wandb.log({'Images':[wandb.Image(f'./outputs/{i}.jpg',  caption=step) for i in range(5)]})
    


def eval_model(test_data_loader, global_step, device, model, disc):
    global syncnet_wt, disc_wt
    running_sync_loss, running_l1_loss, running_disc_real_loss, running_disc_fake_loss, running_perceptual_loss = [], [], [], [], []
    for step, (x, indiv_mels, mel, gt) in enumerate((test_data_loader)):
        model.eval()
        disc.eval()

        x = x.to(device)
        mel = mel.to(device)
        indiv_mels = indiv_mels.to(device)
        gt = gt.to(device)

        pred = disc(gt)
        disc_real_loss = F.binary_cross_entropy_with_logits(pred, torch.ones((len(pred), 1)).to(device))

        g = model(indiv_mels, x)
        pred = disc(g)
        disc_fake_loss = F.binary_cross_entropy_with_logits(pred, torch.zeros((len(pred), 1)).to(device))

        running_disc_real_loss.append(disc_real_loss.item())
        running_disc_fake_loss.append(disc_fake_loss.item())

        sync_loss = get_sync_loss(mel, g, 5)
        
        perceptual_loss = disc.perceptual_forward(g)

        l1loss = recon_loss(g, gt)

        loss = syncnet_wt * sync_loss + disc_wt * perceptual_loss + \
                                (1. - syncnet_wt - disc_wt) * l1loss

        running_l1_loss.append(l1loss.item())
        running_sync_loss.append(sync_loss.item())
        running_perceptual_loss.append(perceptual_loss.item())

    wandb.log({'eval_l1':sum(running_l1_loss)/len(running_l1_loss), 'eval_sync':sum(running_sync_loss)/len(running_sync_loss), 
                'eval_preceptual':sum(running_perceptual_loss)/len(running_perceptual_loss)})
    return sum(running_sync_loss) / len(running_sync_loss)



if __name__ == '__main__':
    train_dataset = Data('./preprocessfps25/train')
    test_dataset = Data('./preprocessfps25/eval')

    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4, drop_last = False, pin_memory=True)

    test_data_loader = DataLoader(test_dataset, batch_size=32,num_workers=4, drop_last = False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Wav2Lip().to(device)
    disc = Wav2Lip_disc_qual().to(device)

    optimizer = optim.Adam(model.parameters(),lr=1e-4, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(disc.parameters(),lr=1e-4, betas=(0.5, 0.999))
    wandb.init(project = 'wav2lip')
    train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer)