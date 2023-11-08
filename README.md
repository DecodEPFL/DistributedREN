# DistributedREN Documentation

## Overview

This repository contains the code accompanying the paper titled "Unconstrained learning of networked nonlinear systems via free parametrization of stable interconnected operators" authored by Leonardo Massai, Danilo Saccani, Luca Furieri, and Giancarlo Ferrari Trecate.

For inquiries about the code, please contact:

- Danilo Saccani: danilo.saccani@epfl.ch
- Leonardo Massai: l.massai@epfl.ch

## Repository Contents

The repository is organized as follows:

1. `main_simulink_3tanks.m`: This MATLAB script is responsible for performing the simulations to collect data. It contains the parameters for the simulations and executes the Simulink file named `sim_3tank.slx`. The script generates the dataset in the file `dataset_sysID_3tanks.mat`.

2. `sim_3tank.slx`: This Simulink model is used in conjunction with `main_simulink_3tanks.m` to perform simulations and generate the dataset.

3. `models.py`: This Python script contains the classes for all the Neural Networks used in the paper. These include Recurrent Neural Networks, Recurrent Equilibrium Networks with different properties, and the class for the proposed approach.

4. `REN_SYSD_3tanks.py`: This Python script is responsible for training the distributed operator described in the paper.

## Dependencies

The main dependencies required to run the code are:

- numpy
- scipy
- matplotlib
- torch

## Usage

1. Run `main_simulink_3tanks.m` to perform simulations and generate the dataset.

2. Use the generated dataset (`dataset_sysID_3tanks.mat`) for training and testing the models.

3. Train the distributed operator using `REN_SYSD_3tanks.py`.

## Citation

If you use this code in your research, please cite the accompanying paper: 
L. Massai, D. Saccani, L. Furieri, and G. Ferrari-Trecate. (2023). Unconstrained learning of networked nonlinear systems via free parametrization of stable interconnected operators. arXiv preprint
