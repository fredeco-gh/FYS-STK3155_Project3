# FYS-STK3155_Project3

## Group members

Frederik Callin Østern, Heine Elias Husdal

## Project description

In this project, we investigate the use of neural networks to solve differential equations in physics. In particular, we demonstrate a method for finding the energy eigenstates and eigenvalues of quantum systems, by applying it to find the excited states of the quantum harmonic oscillator problem in 1 dimension. The procedure is inspired by Physics Informed Neural Networks (PINN's), though with a modified residual cost function and an extra cost function ensuring orthogonality between different states. 

Our results are compared with a simple finite difference procedure to find the solutions, as well as with the analytical solutions. 


## Folder structure

`code/` - Folder with all the code files, including:

- `core/` - Folder containing code for training neural networks, as well as the grid search. It contains: 

- `interfaces.py` - file implementing the skeleton for a physics informed neural network and its cost function. 
- `neural_network.py` - file implementing a feed forward neural network with PyTorch
- `training.py` - file defining a function to train the neural network. 

- `param_search/` - Folder containing code to perform the broad search for the best hyperparameters. It contains: 

- `A_train_ground_state.py` - file containing code for training and loading the ground state wavefunction. 
- `B_broad_sweep_wandb.py` - file containing code for the broad sweep search over different hyperparameters the first excited state

- `tise1d/` - Folder containing code for the quantum harmonic oscillator problem. It contains: 

- `analytic_ho.py` - file with functions for the analytical solution to the harmonic oscillator problem, as well as code to compare NN results with the analytical one. 

- `numerical_ho.py` - file with code for finding the numerical finite difference solution to the Schrödinger equation
- `tise1d.py` - code implementing a physics informed neural network for the time independent Schrödinger equation, including the desired cost function. 

- `utils/` - Folder with simple utilities. 

Notes

- The notebooks in `code/` reproduce the experiments and figures in the report.
- Most notebooks take a while to run, especially those involving hyperparameter searches. To load our computed results directly, set the top variable `LOAD_FROM_FILE = True` in each of the notebooks.
- The core implementation is under `code/utils/` and is importable from the notebooks.

## Running the code

1. Set up a python virtual environment

2. Install packages: `pip install -r requirements.txt`

3. Run `.ipynb` notebooks described above.

## Use of LLMs in this Project

In this project, we have utilized Large Language Models (LLMs) such as ChatGPT to assist with various aspects of project. In particular, it was helpful in the following areas:

- Assisting with and understanding LaTeX syntax for the report.

- Becoming familiar with how to use the library "pickle" to save Python objects.  

- Finding some relevant articles about physics informed neural networks, in particular, article [1] and [3] in the report. 