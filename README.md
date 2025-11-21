# Equivariant Modelling for Catalysis on 2D MXenes

This repository contains the code for the "Equivariant Modelling for Catalysis on 2D MXenes" (Melnyk et al., 2025) paper. 

[[bibtex]](https://github.com/CataLiUst/equimodel-mxene#citation) [[dataset]](https://github.com/CataLiUst/equimodel-mxene#dataset) [[trained_models]](https://github.com/CataLiUst/equimodel-mxene#trained-models) 
<!-- [[dataset]](https://doi.org/10.5281/zenodo.17652071) [[trained_models]](https://doi.org/10.5281/zenodo.17654434) -->

**Note:**
For ease of installation, the repository contains the original EquiformerV2 code from [EquiformerV2 GitHub Repository](https://github.com/atomicarchitects/equiformer_v2). The main modification we have made is the removal of dataset-specific statistics in the energy estimation block 
`equiformer_v2/nets/equiformer_v2/equiformer_v2_oc20.py` L466.  


## Dataset

This project includes two versions of the Ti₂C-MXene dataset, both stored in HDF5 format:

- **Without repetitions**  
  `cataliust_ti2c-mxene_train_without_rep.h5`  
  `cataliust_ti2c-mxene_test_without_rep.h5`

- **With repetitions**  
  `cataliust_ti2c-mxene_train_with_rep.h5`  
  `cataliust_ti2c-mxene_test_with_rep.h5`

These can be downloaded at [dataset_link](https://doi.org/10.5281/zenodo.17652071) (CC BY-NC-SA 4.0 license). Place the dataset files inside your project’s `data_root` directory.


## Trained Models

We provide four trained model checkpoints (`.pth`), covering all combinations of models and datasets on which they we trained:

- **original PT without rep**: `orig_pt_without_rep_seed42.pth`
- **original without repetitions**: `orig_without_rep_seed42.pth`
- **small with repetitions**: `small_with_rep_seed42.pth`
- **small without repetitions**: `small_without_rep_seed42.pth`

These checkpoints can be downloaded at [trained_models_link](https://doi.org/10.5281/zenodo.17654434) (CC BY-NC-SA 4.0 license).


## Pretrained OC20 Model (*Original PT*)

To enable loading the OC20-pretrained EquiformerV2 model (referred to as *original PT* in our paper) for training and evaluation, download the pretrained folder from [EquiformerV2 GitHub Repository](https://github.com/atomicarchitects/equiformer_v2).

In our experiments, we use the 31M All+MD (`eq2_31M_ec4_allmd.pt`) checkpoint. 


Place the downloaded folder inside your project’s `data_root` directory.

## Requirements

To install the environment, use either **mamba** or **conda**:
  <!-- (we have been using mamba for its speed): -->

<!-- <div align="center"> -->
```text
[mamba/conda] env create -f env_equiformer_py310.yml -n <your_env_name>
[mamba/conda] activate <your_env_name>
```
<!-- </div> -->


## Running the Training Script
To run an example training script for the *original* model and our dataset *without repetitions*, 
specify the correct `DATA_PATH` and `TEST_DATA_PATH` in `training_equiformer_orig_worep.py` and run

```bash
python -m torch.distributed.launch --nproc_per_node=NGPUs training_equiformer_orig_worep.py
```


## Running the Evaluation Script

To test a model, run `testing_equiformer.py` script. We recommend running it via DDP, in which case you can run 

<!-- <div align="center"> -->
```bash
torchrun --standalone --nproc_per_node=NGPUs testing_equiformer.py --ckpt path/to/checkpoint.pth
```
<!-- </div> -->
where you specify the number of GPUs you have available in `NGPUs`.

Alternatively, one can run 
<!-- <div align="center"> -->
```bash
python -m torch.distributed.launch --nproc_per_node=NGPUs testing_equiformer.py --ckpt path/to/checkpoint.pth
```
<!-- </div> -->


There are additional parser arguments that one can write:
<div align="center">

| Argument | Type&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Description |
|----------|------------------------------|-------------|
| `--ckpt` | str (Required) | Path to the model checkpoint to evaluate |
| `--data_root` | str  | Path to data folder |
| `--model {small, orig}` | str | Choose model configuration |
| `--select_test_dataset {with_rep, without_rep}` | str | Select dataset variant |
| `--load_OC20_pt` | flag | Enable loading the OC20 pretrained checkpoint |
| `--seed N` | int | Random seed |

</div>


## Citation
If you find our work useful, please cite the paper
```
@inproceedings{
  melnyk2025equivariant,
  title={{Equivariant Modelling for Catalysis on 2D MXenes}},
  author={Pavlo Melnyk and Anmar Karmush and Ania Beatriz Rodr{\'\i}guez-Barrera and M{\r{a}}rten Wadenb{\"a}ck and Michael Felsberg and Johanna Rosen and Jonas Bj{\"o}rk},
  booktitle={EurIPS 2025 Workshop on SIMBIOCHEM},
  year={2025}
}
```

