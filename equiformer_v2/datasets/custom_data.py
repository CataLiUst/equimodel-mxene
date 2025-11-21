import torch
import numpy as np
import os

from torch.utils.data import Dataset
from pathlib import Path


class LargeDFTDataset(Dataset):
    # max_atoms = 116 # max_atoms = max([f.shape[1] for f in forces_array])
    symbol_to_idx = {'C': 0, 'H': 1, 'O': 2, 'Ti': 3}
    # symbol_to_atomic_number = {
    #     'C': 6,
    #     'H': 1,
    #     'O': 8,
    #     'Ti': 22
    # }
    idx_to_atomic_number = {
        0: 6,   # C
        1: 1,   # H
        2: 8,   # O
        3: 22   # Ti
    }

#     @staticmethod
#     def pc_normalize(pc, mask):
#         centroid = torch.mean(pc[mask], dim=0)
#         pc[mask] = pc[mask] - centroid
# #         m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1)))
# #         pc = pc / m
#         return pc
    
    def __init__(self, partition, transform: callable = None, shuffle=False, normalize=False, num_points=65, prefix='', num_samples=None):
        super().__init__()

        assert partition in ('train', 'val', 'test')

        self.partition = partition
        self.transform = transform
        self.shuffle = shuffle
        self.normalize = normalize

        # dset_root = Environment.dset_path
        dset_root = Path("/home/x_pavme/experiments/dft-training/catalyst/tetrasphere/tetrasphere/datasets/CO2_Ti2C-MXene")
        # split_name = f"{partition}"
        # file_name = dset_root / "" / (split_name + ".npy")

        # load and store each array in lists:
        self.positions = torch.from_numpy(np.load(dset_root / f"{prefix}positions_{partition}.npy")).float()
        self.forces = torch.from_numpy(np.load(dset_root / f"{prefix}forces_{partition}.npy")).float()
        self.energy = torch.from_numpy(np.load(dset_root / f"{prefix}energy_{partition}.npy")).float()
        self.nn_indices = torch.from_numpy(np.load(dset_root / f"{prefix}nn_indices_{partition}.npy")).long()
        self.symbols = torch.from_numpy(np.load(dset_root / f"{prefix}symbols_{partition}.npy")).long()
        self.masks = torch.from_numpy(np.load(dset_root / f"{prefix}mask_{partition}.npy")).bool()
        self.nn_vecs = None
        self.cells = None
        
        if os.path.exists(dset_root / f"{prefix}nn_vecs_{partition}.npy"):
            self.nn_vecs = torch.from_numpy(np.load(dset_root / f"{prefix}nn_vecs_{partition}.npy")).float()
        if os.path.exists(dset_root / f"{prefix}cell_{partition}.npy"):
            self.cells = torch.from_numpy(np.load(dset_root / f"{prefix}cell_{partition}.npy")).float()

        # handle number of samples if specified:
        if num_samples is not None:
            assert isinstance(num_samples, int) and num_samples <= len(self.positions)
            self.positions = self.positions[:num_samples]
            self.symbols = self.symbols[:num_samples]
            self.forces = self.forces[:num_samples]
            self.energy = self.energy[:num_samples]
            self.nn_indices = self.nn_indices[:num_samples]
            self.masks = self.masks[:num_samples]
            if self.nn_vecs is not None:
                self.nn_vecs = self.nn_vecs[:num_samples] 
            if self.cells is not None:
                self.cells = self.cells[:num_samples]

    def __getitem__(self, item):
        pcd = self.positions[item]
        forces = self.forces[item]
        energy = self.energy[item]
        nn_indices = self.nn_indices[item]
        symbols = self.symbols[item]
        mask = self.masks[item]

        # shuffle point indices if required
        pt_idxs = torch.arange(0, pcd.shape[0])
        if self.partition == "train" and self.shuffle:
            pt_idxs = pt_idxs[torch.randperm(len(pt_idxs))]

        pcd = pcd[pt_idxs][:].clone()

        # do NOT center/normalize the positions (2024/12/09):
        # if self.normalize: 
        #     pcd[:, 0:3] = self.pc_normalize(pcd[:, 0:3], mask)

        if self.transform is not None:
            pcd = self.transform(pcd)

        to_return = [pcd, symbols, forces, energy, nn_indices, mask]

        if self.nn_vecs is not None:
                to_return.append(self.nn_vecs[item])
        if self.cells is not None:
                to_return.append(self.cells[item])

        return to_return

    def __len__(self):
        return len(self.positions)


class LargeDFTDatasetEquiformer(LargeDFTDataset):
    def __getitem__(self, item):
        mask = self.masks[item]            # [max_atoms]
        pcd = self.positions[item][mask]   # [num_atoms, 3]
        forces = self.forces[item][mask]   # [num_atoms, 3]
        energy = self.energy[item]         # scalar
        symbols = self.symbols[item][mask] # [num_atoms], currently 0->C,1->H,2->O,3->Ti
        natoms = mask.sum()
        # print(natoms)
        # exit()

        # shuffle points if required:
        pt_idxs = torch.arange(0, pcd.shape[0])
        if self.partition == "train" and self.shuffle:
            pt_idxs = pt_idxs[torch.randperm(len(pt_idxs))]

        pcd = pcd[pt_idxs].clone()
        forces = forces[pt_idxs]
        symbols = symbols[pt_idxs]
        mask = mask[pt_idxs]

        if self.transform is not None:
            pcd = self.transform(pcd)

        # convert from symbol indices to actual atomic numbers:
        # C:6, H:1, O:8, Ti:22
        symbol_to_atomic_number = torch.tensor([6, 1, 8, 22]).to(pcd.device)
        symbols = symbol_to_atomic_number[symbols]

        # construct the dictionary as the model input:
        data = {
            'pos': pcd,                 # [num_atoms, 3]
            'atomic_numbers': symbols,  # [num_atoms]
            'energy': energy,           # scalar energy for the configuration
            'forces': forces,           # [num_atoms, 3]
            'natoms': natoms
        }

        if self.cells is not None:
            cell = self.cells[item]   # [3,3]
            data['cell'] = cell
            # if all directions are periodic:
            data['pbc'] = torch.tensor([True, True, True])

        return data


def map_batch_to_device(batch, device):
    """
    Maps all tensors in a batch dictionary to the specified device.

    Args:
        batch (dict): Dictionary containing batch data.
        device (torch.device): Target device (e.g., 'cuda:0' or 'cpu').

    Returns:
        dict: Updated batch dictionary with all tensors on the target device.
    """
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)  # Move tensor to the device
    return batch