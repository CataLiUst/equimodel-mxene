import os 
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))           # path to experiments/
ROOT = os.path.abspath(os.path.join(ROOT, "."))             # repo root
sys.path.insert(0, ROOT)                                    # so local packages import
sys.path.insert(0, os.path.join(ROOT, "equiformer_v2/ocp")) # so 'ocpmodels' resolves

import argparse

from datetime import datetime
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from pytorch_lightning import seed_everything

from tqdm import tqdm
from datetime import datetime

from equiformer_v2.nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20 as EquiformerV2

from utils import * 
from model_configs import *
from types import SimpleNamespace


def get_scheduler(optimizer, lambda_type, warmup_factor, warmup_epochs, lr_min_factor, total_epochs):
    """
    Create a learning rate scheduler with warmup and cosine decay.
    
    Args:
        optimizer: The optimizer to attach the scheduler to.
        scheduler_params: Dictionary containing scheduler parameters.
        
    Returns:
        torch.optim.lr_scheduler.LambdaLR: Configured scheduler.
    """


    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return warmup_factor + (1 - warmup_factor) * (epoch / warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            if lambda_type == "cosine":
                return lr_min_factor + 0.5 * (1 - lr_min_factor) * (1 + torch.cos(torch.pi * progress))
            else:
                raise ValueError(f"Unsupported lambda_type: {lambda_type}")
    
    return LambdaLR(optimizer, lr_lambda)


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)
device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dist.get_rank() == 0:
    print(f"Torch version {torch.__version__}, device='{device}'")

# environment setup:
SEED = 42  

CHECKPOINT_PATH = f"checkpoints/{datetime.now().strftime('%y%m%d_%H%M')}_{SEED}_equiformerv2_orig_worep_100e_25for"
DATA_PATH = "./cataliust_ti2c-mxene_train_without_rep.h5"
TEST_DATA_PATH = "./cataliust_ti2c-mxene_test_without_rep.h5"

# hyperparams:
num_epochs = 100
lr = 0.0004
weight_decay = 1e-3
loss_forces_weight = 25
model_config = model_config_orig
model_config = {**model_config, "avg_num_nodes": 86.7569, "avg_degree": 18.64958191066762} # avg_num_nodes is NOT used and is given here for the sake of completeness
 
batch_size = 1
eval_batch_size = 1

__file__ = os.path.abspath('')

seed_everything(SEED, workers=True)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


train_dataset = DFTDatasetH5(DATA_PATH, split="train", verify_splits=True)
val_dataset   = DFTDatasetH5(DATA_PATH, split="val", verify_splits=True)
test_dataset  = DFTDatasetH5(TEST_DATA_PATH)

# create DistributedSampler for each dataset:
train_sampler = DistributedSampler(train_dataset, shuffle=True)
val_sampler = DistributedSampler(val_dataset)
test_sampler = DistributedSampler(test_dataset)

# adjust the DataLoaders:
train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler, shuffle=False, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, eval_batch_size, sampler=val_sampler, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, eval_batch_size, sampler=test_sampler, shuffle=False, collate_fn=custom_collate_fn) 


if dist.get_rank() == 0:
    print(f"\n\nDataset size:\ntrain: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")


def train_and_validate(model, train_loader, val_loader, criterion, optimizer,
                       num_epochs=10, loss_forces_weight=10, scheduler=None,
                       best_val_metric=float('inf'), checkpoint_path=''):
    if dist.get_rank() == 0:
        print("\n\nTraining the model")

    train_ls, train_ls_f, train_ls_e, val_ls, val_ls_f, val_ls_e = [], [], [], [], [], []
    t0 = datetime.now()

    # global step for per-batch logging if you want it later
    global_step = 0

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_loss_forces = 0.0
        train_loss_energy = 0.0
        total_atoms = 0
        total_batch_systems = 0

        # tqdm for training loop (only on main process)
        train_iter = train_loader
        if dist.get_rank() == 0:
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for data in train_iter:
            data = {k: v.to(device) for k, v in data.items()}
            data = SimpleNamespace(**data)
            optimizer.zero_grad()

            eff_batch_size = data.natoms.sum().item()
            batch_systems = data.natoms.shape[0]
            total_atoms += eff_batch_size
            total_batch_systems += batch_systems

            pred_energy, pred_forces = model(data)
            loss_forces = criterion(pred_forces, data.forces)
            loss_energy = criterion(pred_energy, data.energy)
            loss = loss_forces_weight * loss_forces + (1 / eff_batch_size) * loss_energy

            loss.backward()
            optimizer.step()

            train_loss_forces += loss_forces.item() * eff_batch_size
            train_loss_energy += loss_energy.item() * batch_systems

            if dist.get_rank() == 0:
                train_iter.set_postfix({
                    "forces_loss": f"{loss_forces.item():.4f}",
                    "energy_loss": f"{loss_energy.item():.4f}"
                })
           
            global_step += 1

        if scheduler is not None:
            scheduler.step()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        train_loss_tensor = torch.tensor([train_loss, train_loss_forces, train_loss_energy,
                                          total_atoms, total_batch_systems], device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss_forces = train_loss_tensor[1].item() / train_loss_tensor[-2].item()
        train_loss_energy = train_loss_tensor[2].item() / train_loss_tensor[-1].item()
        train_loss = train_loss_forces + train_loss_energy

        # validation phase:
        model.eval()
        val_loss = 0.0
        val_loss_forces = 0.0
        val_loss_energy = 0.0
        total_val_atoms = 0
        total_val_batch_systems = 0

        val_iter = val_loader
        if dist.get_rank() == 0:
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)

        with torch.no_grad():
            for data in val_iter:
                data = {k: v.to(device) for k, v in data.items()}
                data = SimpleNamespace(**data)

                pred_energy, pred_forces = model(data)
                loss_forces = criterion(pred_forces, data.forces)
                loss_energy = criterion(pred_energy, data.energy)
                loss = loss_forces + loss_energy

                eff_batch_size = data.natoms.sum().item()
                batch_systems = data.natoms.shape[0]
                total_val_atoms += eff_batch_size
                total_val_batch_systems += batch_systems

                val_loss_forces += loss_forces.item() * eff_batch_size
                val_loss_energy += loss_energy.item() * batch_systems

                if dist.get_rank() == 0:
                    val_iter.set_postfix({
                        "forces_loss": f"{loss_forces.item():.4f}",
                        "energy_loss": f"{loss_energy.item():.4f}"
                    })

        val_loss_tensor = torch.tensor([val_loss, val_loss_forces, val_loss_energy,
                                        total_val_atoms, total_val_batch_systems], device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss_forces = val_loss_tensor[1].item() / val_loss_tensor[-2].item()
        val_loss_energy = val_loss_tensor[2].item() / val_loss_tensor[-1].item()
        val_loss = val_loss_forces + val_loss_energy

        # collect losses
        train_ls.append(train_loss)
        train_ls_f.append(train_loss_forces)
        train_ls_e.append(train_loss_energy)
        val_ls.append(val_loss)
        val_ls_f.append(val_loss_forces)
        val_ls_e.append(val_loss_energy)

        if dist.get_rank() == 0:
            # save arrays and checkpoints
            np.save(checkpoint_path + "/train_ls.npy", np.array(train_ls))
            np.save(checkpoint_path + "/train_ls_f.npy", np.array(train_ls_f))
            np.save(checkpoint_path + "/train_ls_e.npy", np.array(train_ls_e))
            np.save(checkpoint_path + "/val_ls.npy", np.array(val_ls))
            np.save(checkpoint_path + "/val_ls_f.npy", np.array(val_ls_f))
            np.save(checkpoint_path + "/val_ls_e.npy", np.array(val_ls_e))

            if val_loss < best_val_metric:
                best_val_metric = val_loss
                save_checkpoint(model, optimizer, epoch, best_val_metric, checkpoint_path, is_best=True)
                print(f"Epoch {epoch+1}: New best model saved with validation loss {val_loss:.4f}")

            if epoch == num_epochs - 1:
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, is_best=False)
                print(f"Epoch {epoch+1}: Model saved with validation loss {val_loss:.4f}")

            t1 = datetime.now()
            print(f"Epoch {epoch+1} ({t1 - t0}): "
                  f"Train Loss: {train_loss_forces:.4f} + {train_loss_energy:.4f} = {train_loss:.4f}, "
                  f"Validation Loss: {val_loss_forces:.4f} + {val_loss_energy:.4f} = {val_loss:.4f}")

    return train_ls, train_ls_f, train_ls_e, val_ls, val_ls_f, val_ls_e

def test(model, test_loader, criterion):
    if dist.get_rank() == 0:
        print("\n\nTesting the model")
    model.eval()

    test_loss = torch.tensor(0.0, device=device)
    test_loss_forces = torch.tensor(0.0, device=device)
    test_loss_energy = torch.tensor(0.0, device=device)
    total_test_atoms = torch.tensor(0, device=device)
    total_test_batch_systems = torch.tensor(0, device=device)

    with torch.no_grad():
        for data in test_loader:
            data = {k: v.to(device) for k, v in data.items()}
            data = SimpleNamespace(**data)

            pred_energy, pred_forces= model(data)
            loss_forces = criterion(pred_forces, data.forces)
            loss_energy = criterion(pred_energy, data.energy)
            loss = loss_forces + loss_energy

            eff_batch_size = data.natoms.sum().item()
            batch_systems = data.natoms.shape[0]
            total_test_atoms += eff_batch_size
            total_test_batch_systems += batch_systems

            test_loss_forces += loss_forces * eff_batch_size
            test_loss_energy += loss_energy * batch_systems

    dist.all_reduce(test_loss_forces, op=dist.ReduceOp.SUM)
    dist.all_reduce(test_loss_energy, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_test_atoms, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_test_batch_systems, op=dist.ReduceOp.SUM)

    test_loss_forces = test_loss_forces.item() / total_test_atoms.item()
    test_loss_energy = test_loss_energy.item() / total_test_batch_systems.item()
    test_loss = test_loss_forces + test_loss_energy

    if dist.get_rank() == 0:
        print(f'Test Loss: {test_loss_forces:.4f} + {test_loss_energy:.4f} = {test_loss:.4f}')
        return test_loss, test_loss_forces, test_loss_energy
    else:
        return None, None, None
    
if dist.get_rank() == 0:
    print("\n\nInstantiating the model")

model = EquiformerV2(None, None, None, **model_config).to(device)

model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

if dist.get_rank() == 0:
    count_parameters(model)

checkpoint_path = CHECKPOINT_PATH
if dist.get_rank() == 0:
    os.makedirs(checkpoint_path, exist_ok=False)

best_val_metric = float('inf')  # for tracking the best validation loss

world_size = dist.get_world_size()

if dist.get_rank() == 0:        
    print(f"\n\nWorld size: {world_size}")

# criterion and optimizer:
criterion = nn.L1Loss(reduction="mean")
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.01*lr)

# train and validate the model:
train_ls, train_ls_f, train_ls_e, val_ls, val_ls_f, val_ls_e = \
    train_and_validate(model, train_loader, val_loader, criterion, optimizer,
                       loss_forces_weight=loss_forces_weight, num_epochs=num_epochs, scheduler=scheduler,
                       best_val_metric=best_val_metric, checkpoint_path=checkpoint_path)

# load the best model:
model, optimizer, epoch, best_val_metric = load_checkpoint(CHECKPOINT_PATH+"/best_model.pth", model, optimizer)
if dist.get_rank() == 0:        
    print(f"Model successfully loaded. Best val metric: {best_val_metric}")

# sanity check: test on the validation set again
if dist.get_rank() == 0:        
    print(f"\nSanity check: testing on the validation set")
_val_ls, _val_ls_forces, _val_ls_energy = test(model, val_loader, criterion)

# test the model:
test_ls, test_ls_forces, test_ls_energy = test(model, test_loader, criterion)


dist.destroy_process_group()