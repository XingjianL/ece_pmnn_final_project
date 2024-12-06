# ece_pmnn_final_project

Project for ECE 592 Physics Modeling with Neural Networks

We try to use neural ode method to fit a underwater robot dynamics consisting of 21 states (position, rotation, velocities, and thrusts) that can be used to solve future robot states given the current robot states. The best performing model is about 9.8 cm position error which is better than a equivalent sized traditional model of 13.4 cm.

### Dataset and full project
The dataset and full project is available at [https://github.com/XingjianL/ece_pmnn_final_project](https://github.com/XingjianL/ece_pmnn_final_project)

### Dependencies
`PyTorch` for data loading and model definitions

`matplotlib` for plotting

`torchdiffeq` for ode solver

`tqdm` for displaying training progress

### train/val scripts
Note: check commit history for other training configurations

`Train_single_batch.py` is used for training neural ode models (currently configured for ReLU training)

`Train.py` is NOT used (ignore this), was trying to load windowed batches of multiple sequences, but the library does not support uneven time steps so we use single sequence for training above.

`Val.py` is used to generate L2 and MSE loss for the validation sequences (currently configured for Baseline model)

### non training/val scripts
`tools/data_loader.py` loads and parse the dataset with pytorch dataloader

`model/*.py` different models

`parse_csv_data_to_sequences.py` parse the simulation output to individual sequence files
