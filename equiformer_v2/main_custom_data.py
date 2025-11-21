import torch
from torch.utils.data import DataLoader
from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20 as EquiformerV2
from datasets.custom_data import LargeDFTDatasetEquiformer
from utils import count_parameters
from types import SimpleNamespace


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
        "batch": batch_index
    }

# Initialize dataset and dataloader
dataset = LargeDFTDatasetEquiformer(partition='train', prefix="new_padded_")
dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn)


# Define model configuration
# model_config = {
#     'num_layers': 20,
#     # 'hidden_dim': 1536,
#     'num_heads': 6,
#     'num_species': 4,  # Adjust based on your dataset
#     'cutoff': 5.0,
#     'max_neighbors': 32,
#     'activation': 'relu',
#     # Add other parameters as required
# # }
# model_config = {
#     # "name": "equiformer_v2",
#     "use_pbc": True,
#     "regress_forces": True,
#     "otf_graph": True,
#     "max_neighbors": 20,
#     "max_radius": 12.0,
#     "max_num_elements": 116,
#     "num_layers": 8,
#     "sphere_channels": 128,
#     "attn_hidden_channels": 64,  # [64, 96] Determines the hidden size of message passing
#     "num_heads": 8,
#     "attn_alpha_channels": 64,   # Not used when `use_s2_act_attn` is True
#     "attn_value_channels": 16,
#     "ffn_hidden_channels": 128,
#     "norm_type": "layer_norm_sh",  # ['rms_norm_sh', 'layer_norm', 'layer_norm_sh']
#     "lmax_list": [4],
#     "mmax_list": [2],
#     "grid_resolution": 18,  # [18, 16, 14, None]
#     "num_sphere_samples": 128,
#     "edge_channels": 128,
#     "use_atom_edge_embedding": True,
#     "share_atom_edge_embedding": False,  # If `True`, `use_atom_edge_embedding` must be `True`
#     "distance_function": "gaussian",
#     "num_distance_basis": 512,  # Not used
#     "attn_activation": "silu",
#     "use_s2_act_attn": False,  # [False, True]
#     "use_attn_renorm": True,   # Attention re-normalization
#     "ffn_activation": "silu",  # ['silu', 'swiglu']
#     "use_gate_act": False,     # [True, False]
#     "use_grid_mlp": True,      # [False, True]
#     "use_sep_s2_act": True,    # Separable S2 activation
#     "alpha_drop": 0.1,         # [0.0, 0.1]
#     "drop_path_rate": 0.1,     # [0.0, 0.05]
#     "proj_drop": 0.0,
#     "weight_init": "uniform",  # ['uniform', 'normal']
# }

model_config = {
    "use_pbc": True, 
    "regress_forces": True,
    "otf_graph": True,  
    "max_neighbors": 20,  
    "max_radius": 12.0,  
    "max_num_elements": 23,  
    "num_layers": 8,
    "sphere_channels": 128,
    "attn_hidden_channels": 64,
    "num_heads": 8,
    "attn_alpha_channels": 64,
    "attn_value_channels": 16,
    "ffn_hidden_channels": 128,
    "norm_type": "layer_norm_sh",
    "lmax_list": [4],
    "mmax_list": [2],
    "grid_resolution": 18,
    "num_sphere_samples": 128,
    "edge_channels": 128,
    "use_atom_edge_embedding": True,
    "share_atom_edge_embedding": False,
    "distance_function": "gaussian",
    "num_distance_basis": 512,
    "attn_activation": "silu",
    "use_s2_act_attn": False,
    "use_attn_renorm": True,
    "ffn_activation": "silu",
    "use_gate_act": False,
    "use_grid_mlp": True,
    "use_sep_s2_act": True,
    "alpha_drop": 0.1,
    "drop_path_rate": 0.1,
    "proj_drop": 0.0,
    "weight_init": "uniform",
}

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
# model = EquiformerV2(**model_config)
model = EquiformerV2(None, None, None, **model_config).to(device)
count_parameters(model)


# # Batch size
# batch_size = 2

# # Define the number of atoms per graph
# num_atoms_per_graph = [83, 99] 

# # Total number of atoms in the batch
# num_atoms = sum(num_atoms_per_graph)

# # Atomic positions
# pos = torch.rand(num_atoms, 3, device=device)  # Random 3D coordinates for all atoms

# # Atomic numbers (elements): Randomly assigned within [1, 10]
# atomic_numbers = torch.randint(1, 10, (num_atoms,), device=device)

# # Periodic boundary condition information
# cell = torch.eye(3).unsqueeze(0).repeat(2, 1, 1).to(device)  # Identity matrix for both graphs
# pbc = torch.tensor([[True, True, True], [True, True, True]], dtype=torch.bool, device=device)  # No periodicity

# # Create the batch tensor
# batch = torch.cat([
#     torch.full((n,), i, dtype=torch.long) for i, n in enumerate(num_atoms_per_graph)
# ]).to(device)

# # Number of atoms in each graph
# natoms = torch.tensor(num_atoms_per_graph, dtype=torch.long, device=device)

# print(pos.device)
# print(atomic_numbers.device)
# print(cell.device)
# print(pbc.device)
# print(batch.device)
# print(natoms.device)
# print()


# # Compile the batch dictionary
# data = {
#     'pos': pos,
#     'atomic_numbers': atomic_numbers,
#     'cell': cell,
#     'pbc': pbc,
#     'natoms': natoms,
#     "batch": batch
# }

# data = SimpleNamespace(**data)

# Ensure all tensors are on the correct device

for data in dataloader:
    data = {k: v.to(device) for k, v in data.items()}
    print(data.keys())
    data = SimpleNamespace(**data)
    # print(data.pos.shape, data.pbc.shape, data.cell.shape, data.natoms, data.batch.shape)
    # energy, forces = model(data)
    # print(data.pos.device)
    print(data.atomic_numbers)
    # print(data.cell.device)
    # print(data.pbc.device)
    # print(data.batch.device)
    # print(data.natoms.device)
    # exit()
    break

# Forward pass through the model
model.eval()
with torch.no_grad():
    energy, forces = model(data)

print(energy.shape, forces.shape)

# print(model)