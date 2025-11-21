# For two gpus run: torchrun --standalone --nproc_per_node=2 testing_equiformer.py
import os 
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))          # path to experiments/
ROOT = os.path.abspath(os.path.join(ROOT, "."))            # repo root

sys.path.insert(0, ROOT)                                    # so local packages import
sys.path.insert(0, os.path.join(ROOT, "equiformer_v2/ocp")) # so 'ocpmodels' resolves



import argparse
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm


from equiformer_v2.nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20 as EquiformerV2
from utils import *
from model_configs import * 

# --------------------- DDP / torchrun compatibility -------------------
p = argparse.ArgumentParser(add_help=True)

# DDP args (you already had these)
p.add_argument("--local_rank", type=int, default=None)                      # old style
p.add_argument("--local-rank", dest="local_rank_dash", type=int, default=None)  # torchrun sometimes passes this

# ------------------------- NEW: training/config args -------------------------

p.add_argument(
    "--data_root",
    type=str,
    default="YOUR PATH",
    help="Change: Root directory where the HDF5 datasets live.",
)


p.add_argument(
    "--select_test_dataset",
    choices=["with_rep", "without_rep"],
    default="without_rep",
    help="Choose test dataset, with or without artificial repetitions",
)
p.add_argument(
    "--model",
    choices=["small", "orig"],
    default="orig",
    help="Select model config",
)


p.add_argument("--load_OC20_pt", action="store_true",
               help="Load fixed pretrained weights (uses a hardcoded path).")


p.add_argument("--seed",
    type=int,
    default = 1996,
    help="set the seed"

)


p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint  pth")


args, _ = p.parse_known_args()

#------------------------------------------------


# Check that combination of args is allowed
validate_args(args)


LOCAL_RANK = (
    args.local_rank
    if args.local_rank is not None
    else (args.local_rank_dash if args.local_rank_dash is not None
          else int(os.environ.get("LOCAL_RANK", 0)))
)
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

if WORLD_SIZE > 1:
    dist.init_process_group(backend="nccl")

if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)
device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")


is_dist = (WORLD_SIZE > 1) and dist.is_initialized()
is_main = (not is_dist) or (dist.get_rank() == 0)


if is_main:
    print("\nCONFIG:")
    print("-----------------------------------------------------------------------------------------------")
    print(
        f"[cfg] ckpt={args.ckpt}, data={args.select_test_dataset}, model={args.model}, seed={args.seed}, use_PT={args.load_OC20_pt}"
    )
    print("-----------------------------------------------------------------------------------------------\n")
    print(f"[init] torch={torch.__version__} world_size={WORLD_SIZE} local_rank={LOCAL_RANK} device={device}\n")


quit()
seed_everything(args.seed, workers=True)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# ------------------------------- Data ---------------------------------

data_root = Path(args.data_root)
DATA_PATH_TEST  = data_root / f"cataliust_ti2c-mxene_test_{args.select_test_dataset}.h5"
test_dataset  = DFTDatasetH5(DATA_PATH_TEST)

_model_map = {
    "small": model_config_small,
    "orig": model_config_orig,
}
model_config_to_use = _model_map[args.model]

# This is what the pretrained model requried
if args.load_OC20_pt:
    model_config_to_use["max_num_elements"] = 90


# Statistics based on the dataset
if args.select_test_dataset == "with_rep": 
    model_config_to_use = {**model_config_to_use, "avg_num_nodes": 520.46036, "avg_degree": 18.6495819106676}
elif args.select_test_dataset == "without_rep":
    model_config_to_use = {**model_config_to_use, "avg_num_nodes": 86.7569, "avg_degree": 18.6488219106676}



# Samplers conditional on DDP
if is_dist:
    test_sampler  = DistributedSampler(test_dataset,  shuffle=True)
else:
    test_sampler  = None


pin_memory = torch.cuda.is_available()


test_loader = DataLoader(
    test_dataset, batch_size = 1,
    sampler=test_sampler, shuffle=False,
    collate_fn=custom_collate_fn, num_workers=16, pin_memory=pin_memory
)



if is_main:
    print(f"\n\nDataset size: test: {len(test_dataset)}")


# ------------------------------ Model ---------------------------------

model = EquiformerV2(None, None, None, **model_config_to_use).to(device)

model, _, best_epoch, best_val_metric = load_checkpoint(args.ckpt, model)

if is_main:
    print(f"Loaded model at epoch: {best_epoch} with best val metric being: {best_val_metric}")
    count_parameters(model)


if is_dist:
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=False)



def test(model, test_loader, criterion):
    if is_main:
        print("\n\n Begin testing")
    
    model.eval()

    # Accumulators (all on device for DDP all_reduce)
    test_loss_forces   = torch.tensor(0.0, device=device)  # atom-weighted L1 for forces
    test_loss_energy   = torch.tensor(0.0, device=device)  # system-averaged L1 for energy
    total_test_atoms   = torch.tensor(0.0, device=device)
    total_test_systems = torch.tensor(0.0, device=device)

    # For energy-per-atom MAE = sum |dE| / sum natoms
    sum_abs_dE         = torch.tensor(0.0, device=device)  # Σ |ΔE|
    sum_atoms_for_Ea   = torch.tensor(0.0, device=device)  # Σ natoms (same as total_test_atoms but keep separate for clarity)

    with torch.no_grad():
        tbar = tqdm(test_loader, disable=not is_main, dynamic_ncols=True, desc="Testing")
        for data in tbar:
            data = {k: v.to(device, non_blocking=True) for k, v in data.items()}
            data = SimpleNamespace(**data)

            # Forward
            pred_energy, pred_forces = model(data)  # shapes typically [B,1] and [sumN,3] or similar
            # Ensure shapes are compatible for losses / scalars
            pe = pred_energy.view(-1)
            te = data.energy.view(-1)

            # Losses (your definitions)
            loss_forces = criterion(pred_forces, data.forces)  # mean over atoms in batch
            loss_energy = criterion(pe, te)                    # mean over systems in batch

            # Bookkeeping weights
            eff_atoms = float(data.natoms.sum().item())        # total atoms in batch
            batch_sys = float(data.natoms.shape[0])            # number of systems in batch

            # Accumulate your existing metrics
            test_loss_forces   += loss_forces * eff_atoms
            test_loss_energy   += loss_energy * batch_sys
            total_test_atoms   += eff_atoms
            total_test_systems += batch_sys

            # --- New: energy-per-atom MAE components ---
            abs_dE = (pe - te).abs()                           # [B]
            sum_abs_dE       += abs_dE.sum()                   # Σ |ΔE|
            sum_atoms_for_Ea += data.natoms.float().sum()      # Σ natoms

    # DDP reductions
    if is_dist:
        for t in [test_loss_forces, test_loss_energy, total_test_atoms,
                  total_test_systems, sum_abs_dE, sum_atoms_for_Ea]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    # Final metrics
    test_loss_forces_val = (test_loss_forces   / total_test_atoms).item()    # atom-weighted L1 on forces
    test_loss_energy_val = (test_loss_energy   / total_test_systems).item()  # system-averaged L1 on energy
    test_loss_total      = test_loss_forces_val + test_loss_energy_val

    # New: energy-per-atom MAE (eV/atom)
    mae_e_per_atom = (sum_abs_dE / sum_atoms_for_Ea).item() if sum_atoms_for_Ea.item() > 0 else float('nan')

    if is_main:
        print(f"Test Loss (forces + energy/sys): {test_loss_forces_val:.6f} + {test_loss_energy_val:.6f} = {test_loss_total:.6f}")
        print(f"MAE(E per atom): {mae_e_per_atom:.6f} ")
        return test_loss_total, test_loss_forces_val, test_loss_energy_val, mae_e_per_atom
    else:
        return None, None, None, None




criterion = nn.L1Loss(reduction="mean")


if __name__ == "__main__":
    test(model, test_loader, criterion)

    # Cleanup DDP 
    if is_dist:
        dist.destroy_process_group()