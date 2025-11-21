import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.utils.data import Dataset
from pathlib import Path

from collections import OrderedDict

class DFTDatasetH5(Dataset):
    """
    PyTorch Dataset for variable-length atomistic data stored in HDF5.

    If `indices` is None, will try to read a split from the HDF5:
      - if `split` is provided, load f["splits/<split>"]
      - else try 'train' -> 'val'
      - else fall back to all samples
    """
    def __init__(self, h5_path, indices=None, split=None):
        self.h5_path = h5_path
        self.split_name = None

        with h5py.File(h5_path, "r") as f:
            self.offsets   = f["offsets"][:]      # int64, slice boundaries
            self.num_atoms = f["num_atoms"][:]    # int32, sequence lengths
            self.energy    = f["energy"][:]       # float64
            self.cell      = f["cell"][:]         # (N,3,3) float64

            self._N_total = len(self.offsets) - 1

            if indices is not None:
                self.indices = np.asarray(indices, dtype=np.int64)
                self.split_name = split  # may be None if indices provided directly
            else:
                # Try to load indices from the file
                loaded = False
                if "splits" in f:
                    grp = f["splits"]

                    if split is not None:
                        ds = grp.get(split)
                        if ds is None:
                            raise KeyError(f"Requested split '{split}' not found in HDF5 ('splits' group exists but no '{split}').")
                        self.indices = ds[:].astype(np.int64, copy=False)
                        self.split_name = split
                        loaded = True
                    else:
                        for cand in ("train", "val"):
                            if cand in grp:
                                self.indices = grp[cand][:].astype(np.int64, copy=False)
                                self.split_name = cand
                                loaded = True
                                break

                if not loaded:
                    # Fall back to all samples
                    self.indices = np.arange(self._N_total, dtype=np.int64)
                    self.split_name = "all"

        self._h5 = None  # lazy-open per worker

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
            self._pos = self._h5["positions"]  # float64
            self._frc = self._h5["forces"]     # float64
            self._sym = self._h5["symbols"]    # int64

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        self._ensure_open()
        j = int(self.indices[i])                         # original sample index
        s, e = self.offsets[j], self.offsets[j+1]
        pos  = torch.from_numpy(self._pos[s:e]).float()  # (L,3) 
        frc  = torch.from_numpy(self._frc[s:e]).float()  # (L,3) 
        sym  = torch.from_numpy(self._sym[s:e])          # (L,)   int64
        E    = torch.tensor(self.energy[j]).float()      # scalar
        cell = torch.from_numpy(self.cell[j]).float()    # (3,3)
        L    = int(self.num_atoms[j])
        return {
            "pos": pos,
            "forces": frc,
            "atomic_numbers": sym,
            "energy": E.unsqueeze(0),
            "cell": cell,
            "natoms": L,
            "pbc": torch.tensor([True, True, True]),
        }

    def __repr__(self):
        return (f"{self.__class__.__name__}(path='{self.h5_path}', "
                f"split='{self.split_name}', size={len(self)})")
    

def custom_collate_fn(batch):
    """
    Collate function to batch graph data for EquiformerV2.
    Args:
        batch (list of dict): List of individual data samples.
    Returns:
        dict: Batched data.
    """
    # Extract fields
    pos = torch.cat([b["pos"] for b in batch], dim=0)
    atomic_numbers = torch.cat([b["atomic_numbers"] for b in batch], dim=0)
    cell = torch.stack([b["cell"] for b in batch if b["cell"] is not None]) if "cell" in batch[0] else None
    pbc = torch.stack([b["pbc"] for b in batch if b["pbc"] is not None]) if "pbc" in batch[0] else None
    natoms = torch.tensor([b["natoms"] for b in batch], dtype=torch.long)
    forces = torch.cat([b["forces"] for b in batch], dim=0)
    energy = torch.cat([b["energy"] for b in batch], dim=0)
    
    # Build the batch index
    batch_index = torch.cat([
        torch.full((b["pos"].shape[0],), i, dtype=torch.long) for i, b in enumerate(batch)
    ], dim=0)

    return {
        "pos": pos,
        "atomic_numbers": atomic_numbers,
        "cell": cell,
        "pbc": pbc,
        "natoms": natoms,
        "batch": batch_index, 
        "forces": forces, 
        "energy": energy
    }


def count_parameters(model):
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()
    print(f'num_params is: {num_params}')


def save_checkpoint(model, optimizer, epoch, best_val_metric, checkpoint_path, is_best=False):
    """
    Save model checkpoint.

    Parameters:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): The current epoch number.
        best_val_metric (float): The best validation metric achieved so far.
        checkpoint_path (str): Directory where the checkpoint will be saved.
        is_best (bool): Whether this checkpoint is the best model so far.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_metric': best_val_metric,
    }
    
    # If this is the best model so far, save it separately
    if is_best:
        best_model_file = os.path.join(checkpoint_path, 'best_model.pth')
        torch.save(state, best_model_file)
    
    else:
        # Save the checkpoint
        checkpoint_file = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(state, checkpoint_file)


def load_checkpoint(checkpoint_path, model, optimizer=None, load_optimizer=True):
    """
    Load model checkpoint.

    Parameters:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the state_dict into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state_dict into (if required).
        load_optimizer (bool): Whether to load the optimizer state_dict (default: True).

    Returns:
        model (torch.nn.Module): Model with loaded weights.
        optimizer (torch.optim.Optimizer, optional): Optimizer with loaded state_dict (if provided).
        epoch (int): The epoch at which the checkpoint was saved.
        best_val_metric (float): The best validation metric at the time of saving the checkpoint.
    """
    # Load the checkpoint from file
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Load model state_dict
    state_dict = checkpoint['model_state_dict']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # remove 'module.' prefix
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(state_dict)

    # Load optimizer state_dict if applicable
    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Extract additional information from the checkpoint
    epoch = checkpoint.get('epoch', -1)
    best_val_metric = checkpoint.get('best_val_metric', None)

    return model, optimizer, epoch, best_val_metric


def load_PT_equiformer_weights(path, model):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = ckpt["state_dict"]  # <-- confirmed key from inspection

    # remove double "module.module." prefix
    new_state = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module.module."):
            new_state[k[len("module.module."):]] = v
        elif k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[ckpt] load strict=False | missing={len(missing)} unexpected={len(unexpected)}")
    if missing and len(missing) < 12:
        print("Missing keys:", missing)
    if unexpected and len(unexpected) < 12:
        print("Unexpected keys:", unexpected)

    return model


def validate_args(args):
    # (model, select_test_dataset, load_OC20_pt)
    allowed = {
        ("orig",  "without_rep", False),
        ("orig",  "without_rep", True),   # PT only allowed here
        ("small", "without_rep", False),
        ("small", "with_rep",   False),
    }

    key = (args.model, args.select_test_dataset, bool(args.load_OC20_pt))

    if key not in allowed:
        raise ValueError(
            "Invalid argument combination:\n"
            f"  --model {args.model}, --select-test-dataset {args.select_test_dataset}, "
            f"--load_OC20_pt {bool(args.load_OC20_pt)}\n\n"
            "Allowed combinations are:\n"
            "  • model=orig,  select-test-dataset=without_rep, load_OC20_pt=False\n"
            "  • model=orig,  select-test-dataset=without_rep, load_OC20_pt=True\n"
            "  • model=small, select-test-dataset=without_rep, load_OC20_pt=False\n"
            "  • model=small, select-test-dataset=with_rep,   load_OC20_pt=False\n"
            "\nNote: small + PT (load_OC20_pt=True) is not allowed."
        )

   
    if args.load_OC20_pt:
        pt_path = os.path.join(args.data_root, "eq2_31M_ec4_allmd.pt")

        if not os.path.isfile(pt_path):
            raise FileNotFoundError(
                f"Pretrained checkpoint not found:\n"
                f"  {pt_path}\n\n"
                "Make sure eq2_31M_ec4_allmd.pt is inside the folder given by --data_root.\n"
            )
