# DistributedREN Documentation

## Overview

This repository contains the code accompanying the paper titled "Unconstrained learning of networked nonlinear systems via free parametrization of stable interconnected operators" authored by Leonardo Massai, Danilo Saccani, Luca Furieri, and Giancarlo Ferrari Trecate.

For inquiries about the code, please contact:

- Danilo Saccani: danilo.saccani@epfl.ch
- Leonardo Massai: l.massai@epfl.ch

## Repository Contents

The repository is organized as follows:

1. `main_data_generator.m`: This MATLAB script is responsible for performing the simulations to collect data. It contains the parameters for the simulations and executes the Simulink file named `tank_simulator.slx`. The script generates the dataset in the file `dataset_sysID_3tanks.mat`.

2. `tank_simulator.slx`: This Simulink model is used in conjunction with `main_data_generator.m` to perform simulations and generate the dataset.

3. `models.py`: This Python script contains the classes for all the Neural Networks used in the paper. These include Recurrent Neural Networks, Recurrent Equilibrium Networks with different properties, and the class for the proposed approach.

4. `main_sysid_3tanks.ipynb`: This Jupyter Notebook is responsible for training the distributed operator described in the paper as well as single operators for the system identification task.

5. `run_model.py`: This Python script is used to run the trained model on the test dataset and visualize the results.

## Dependencies

The main dependencies required to run the Python code are:

- numpy
- scipy
- matplotlib
- torch

## Usage

1. Run `main_data_generator.m` to perform simulations and generate the dataset.

2. Use the generated dataset (`dataset_sysID_3tanks.mat`) for training and testing the models.

3. Train the distributed operator using `main_sysid_3tanks.ipynb`.

4. Test the trained model using `run_model.py` to visualize the results.

## License
This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by] 

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg



## Citation

If you use this code in your research, please cite the accompanying paper: 

L. Massai, D. Saccani, L. Furieri, and G. Ferrari-Trecate. (2023). Unconstrained learning of networked nonlinear systems via free parametrization of stable interconnected operators. arXiv preprint
