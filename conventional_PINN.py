'''
PINN for two coupled oscillators
Rory Clements, 30/1/25

Create and train a PINN for two coupled oscillators for variable boundary conditions.
This PINN implementation uses a conventional densely-connected neural network, exclusively
using the tanh activation function. This script does allow the user to vary the boundary
conditions during training in order to attempt the decoupling of the trained PINN from
a single unique set.

This script will create an "output" directory in the directory it is in and slowly
populate it with training and evaluation plots, once every x number of training steps.

Please note that upon re-running this script, all previous plots present in the "output"
directory will be deleted, permanently. This is done to prevent clutter due to the
potential of a very large number of figures being generated. So, if you want to save
any of the resulting plots, please create copies of them elsewhere before executing
this script again.

Changelog:
v1: - Sanitised code for publication to GitHub.

Known problems:
    -

Notes/observations/ToDo:
    - This version was used to generate the conventional PINN results for the 1D PWPINNs paper.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib import gridspec
from scipy.integrate import odeint
import torch
import torch.nn as nn
import sys
import os 
import glob
import time

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#============================
#DEFINE CLASSES AND FUNCTIONS
#============================

class dirOps:
    def __init__(self):
        self.dir_name = "output" #Set the directory name to be dir_name by default
        self.is_new = False #Variable used to communicate if a new directory has been created
        self.dir_path = None #New/existing directory path
    
    def newDir(self, dir_name="output"):
        #Allows the user to create/modify another directory using the same instance of a dirOps object
        self.dir_name = dir_name #Set the directory name to be dir_name by default
        self.is_new = False #Variable used to communicate if a new directory has been created
        self.dir_path = None #New/existing directory path
    
    def CreateDir(self):
        #Creates new directory in current directory, with default name "output"
        #Returns new directory path, even if it already exists, for use when saving files to the new directory
        script_dir = os.path.dirname(os.path.abspath(__file__)) #Get the directory of this script
        self.dir_path = os.path.join(script_dir, self.dir_name)
        try:
            if os.path.isdir(self.dir_path) == False: #Check if output folder already exists
                os.mkdir(self.dir_path) #Create new directory
                self.is_new = True #Return True for is_new since new directory has been created
                print(f"Created new directory \"{self.dir_name}\"")
            else:
                print(f"Directory \"{self.dir_name}\" already exists")
                self.is_new = False #Return False for is_new since directory already exists
        except:
            print(f"Cannot create \"{self.dir_name}\". Check directory hierarchy and privileges.")
        return self.dir_path, self.is_new
    
    def empty_dir(self):
        #Delete contents of the directory "dir_name" if it is not new, as defined by the truth of self.is_new
        script_dir = os.path.dirname(os.path.abspath(__file__)) #Get the directory of this script
        dir_path = os.path.join(script_dir, self.dir_name)
        if self.is_new == False:
            try:
                files = os.listdir(dir_path)
                for file in files: #Remove files recursively
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path): #Check entity is a file and not a directory (since we may place old runs into directories within the output directory)
                        os.remove(file_path) #Remove the file
                print(f"All files removed from \"{self.dir_name}\" successfully.")
            except:
                print(f"Error while attempting to remove file(s) from \"{self.dir_name}\".")
            else:
                print(f"\"{self.dir_name}\" cannot be emptied since it is a new directory.")

class NeuralNetwork(nn.Module):
	#Define neural network with smaller number of trainable parameters
	def __init__(self):
		super().__init__()
		
		#Initialize layers
		self.fc0 = nn.Linear(5, 64)
		self.fc1 = nn.Linear(64, 64)
		self.fc2 = nn.Linear(64, 2)

		#Initialize activation functions
		self.a1 = nn.Tanh()

	def forward(self, t):
		u = self.a1(self.fc0(t))
		
		u = self.a1(self.fc1(u))
		u = self.a1(self.fc1(u))
		u = self.a1(self.fc1(u))
		u = self.a1(self.fc1(u))

		u = self.fc2(u)
		return u

'''
class NeuralNetwork(nn.Module):
    #Define neural network with larger number of trainable parameters
    def __init__(self):
        super().__init__()
        
        #Initialize layers
        self.fc0 = nn.Linear(5, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

        #Initialize activation functions
        self.a1 = nn.Tanh()

    def forward(self, t):
        u = self.a1(self.fc0(t))
        
        u = self.a1(self.fc1(u))
        u = self.a1(self.fc2(u))
        u = self.a1(self.fc2(u))
        
        u = self.fc3(u)
        return u
'''

def IsGPU():
    #Check for GPU and use it if available
    if torch.cuda.is_available():
        compute_device = torch.device("cuda")
        print("GPU available and being used")
    else:
        compute_device = torch.device("cpu")
        print("GPU not available, using CPU")
    return compute_device

def rand_bcs(bc_means_tensor, bc_sigma_tensor):
    #Creates a tensor of boundary conditions centred around the means in bc_means_tensor with standard deviations given by bc_sigma_tensor
    bc_tensor = torch.zeros(1, bc_means_tensor.shape[1])
    rand_gen = torch.Generator(device='cpu') #Initialise random tensor seed
    rand_gen.seed() #Seed normal random tensor generator
    for bc_no in range(bc_means_tensor.shape[1]): #Iterate over each boundary conditions
        bc_tensor[0][bc_no] = torch.empty(1, 1).normal_(mean=bc_means_tensor[0][bc_no], std=bc_sigma_tensor[0][bc_no], generator=rand_gen)
    bc_tensor = bc_tensor.to(device)
    return bc_tensor

def dSdt(t, S):
    #Returns the first derivative of the vector S with respect to time
    x1, v1, x2, v2 = S #Define velocity and acceleration of the degree of freedom
    return [v1,
        -omega_1_squared*(x1-a) + k2*(x2-b),
        v2,
        -omega_2_squared*(x2-b) + k2*(x1-a)]

#==============
#INITIALIZATION
#==============

device='cuda' #Device on which to train the network. Will automatically switch to cpu if gpu is not available.
eval_device = "cuda" #Device on which to evaluate the trained network. It can be advantageous to evaluate on the CPU, since RAM is cheaper than vRAM for large tensors.
device = IsGPU() #Get status of GPU and switch compute device to it

#Create directory "output"
out_directory = dirOps() #Create out_directory object
out_directory.newDir("output")
dir_state = out_directory.CreateDir() #Get the path to the new output directory along with whether it had to be created

#Create directory "output/training"
out_directory.newDir("output/training")
train_dir = out_directory.CreateDir() #Get the path to the new output directory along with whether it had to be created
out_directory.empty_dir() #Attempt to empty directory. No action will be taken if directory is new.

#Create directory "output/evaluation"
out_directory.newDir("output/evaluation")
eval_dir = out_directory.CreateDir() #Get the path to the new output directory along with whether it had to be created
out_directory.empty_dir() #Attempt to empty directory. No action will be taken if directory is new.

#Create directory "output/error"
out_directory.newDir("output/error")
err_dir = out_directory.CreateDir() #Get the path to the new output directory along with whether it had to be created
out_directory.empty_dir() #Attempt to empty directory. No action will be taken if directory is new.

#=======================================
#DEFINE INDEPENDENT SIMULATION VARIABLES
#=======================================

#Training control variables
target_rolling_mean_loss = 1e-9 # Target rolling mean loss. In order to run the training loop to a specific number of training steps, set this to some unachievable small value, such as 1e-9.
loss_monitor = 1000 #Number of backpropagation steps to perform before displaying loss values
lr = 1e-3 #Define network learning rate

#Network loss function hyperparameters
field_weight = 1 #Controls field loss weighting
boundary_weight = 1 #Controls the loss weighting for the positional and velocity boundary condition
F_smooth_weight = 0 #Controls first derivative smoothing of F coefficients loss weighting
high_freq_weight = 0 #Controls high frequency suppression loss weighting

#Problem-specific independent variables
m1, m2 = 1, 1 #Masses of the oscillators
k1, k2, k3 = 2, 10, 3 #Spring constants
a, b = 1, 2 #Equilibrium positions of the masses
x1_0, x2_0 = a+0.1, b-0.2 #Initial displacements of the oscillators
v1_0, v2_0 = 0.2, -0.1 #Initial velocities of the oscillators

#Temporal dimension
t_start, t_stop, t_step_no = 10, 20, 10000 #Time start, stop and number of steps for PINN evaluation
t_train_start, t_train_stop, t_train_step_no = 10, 20, 100 #Time start, stop and number of steps for PINN evaluation

#===================
#BOUNDARY CONDITIONS
#===================

#Fixed boundary conditions
x1_bc_mean = x1_0 #Mean of normal distribution used to seed positional boundary condition for x1 during PINN training
x1_bc_standard_dev = 1 #Standard deviation of normal distribution used to seed positional boundary condition for x1 during PINN training
dx1dt_bc_mean = v1_0 #Mean of normal distribution used to seed velocity boundary condition for x1 during PINN training
dx1dt_bc_standard_dev = 1 #Standard deviation of normal distribution used to seed velocity boundary condition for x1 during PINN training
x2_bc_mean = x2_0 #Mean of normal distribution used to seed positional boundary condition for x2 during PINN training
x2_bc_standard_dev = 1 #Standard deviation of normal distribution used to seed positional boundary condition for x2 during PINN training
dx2dt_bc_mean = v2_0 #Mean of normal distribution used to seed velocity boundary condition for x2 during PINN training
dx2dt_bc_standard_dev = 1 #Standard deviation of normal distribution used to seed velocity boundary condition for x2 during PINN training

#=====================================
#DEFINE DEPENDENT SIMULATION VARIABLES
#=====================================

#Temporal dimension
t_train = torch.linspace(t_train_start, t_train_stop, t_train_step_no).view(-1, 1).requires_grad_(True).to(device) #Training time
t_eval = torch.linspace(t_start, t_stop, t_step_no).view(-1, 1).to(device) #Evaluation time

#Equations of motion variables
omega_1_squared = (k1+k2)/m1
omega_2_squared = (k2+k3)/m2

#Generate tensors for fixed evaluation input boundary conditions for PINN
#These must have the same tensor dimensions as time in order for the linear
#algebra to work in the network.
x1_eval = torch.tensor(x1_0).to(device).repeat(t_eval.shape)
dx1dt_eval = torch.tensor(v1_0).to(device).repeat(t_eval.shape)
x2_eval = torch.tensor(x2_0).to(device).repeat(t_eval.shape)
dx2dt_eval = torch.tensor(v2_0).to(device).repeat(t_eval.shape)
eval_bc_domain = torch.cat((x1_eval, dx1dt_eval, x2_eval, dx2dt_eval, t_eval), dim=1)

#==================
#NUMERICAL SOLUTION
#==================

S_0 = [x1_0, v1_0, x2_0, v2_0] #Vector of boundary conditions for S

x_1_numerical_position, x_1_numerical_velocity, x_2_numerical_position, x_2_numerical_velocity = odeint(dSdt, y0=S_0, t=t_eval.detach().to("cpu")[:,0], tfirst=True).T #Get the numerical solution using odeint() over the evaluation time domain

#==========
#TRAIN PINN
#==========

torch.manual_seed(123) #Fix seed for network

#Initialise tensor of normal distribution and sample size
rand_gen = torch.Generator(device='cpu') #Initialise random tensor seed

#Declare and initialise PINN
pinn = NeuralNetwork().to(device) #Declare PINN and move it to the compute device
pytorch_total_params = sum(p.numel() for p in pinn.parameters() if p.requires_grad) #Get number of trainable parameters in model
print("\nNumber of trainable parameters in model: ", pytorch_total_params, "\n")

#Train the PINN
step_history, loss_history, rolling_loss_history = [], [], [] #Empty arrays for capturing loss evolution
optimiser = torch.optim.Adam(pinn.parameters(), lr=lr)
loss_sum = 1e6 #Initial, arbitrary loss sum such that the training process will start
rolling_mean_loss = 1e6 #Initial, arbitrary loss sum such that the training process will start
step = 0 #Training step start
eval_plot_time = 0 #Clock to measure time spent evaluating and plotting results
eval_time = 0 #Clock to be used to calculate mean of evaluation time
numerical_time = 0 #Clock to be used to calculate mean of numerical evaluation time
no_evals = 0
total_timer_start = time.time() #Encapsulate code to get evaluation time

# To run the training loop to a specific number of training steps instead of to a set
# rolling mean loss, uncomment the for statement below and comment the while statement,
# along with the step increment directly below it.
for step in range(1000001): #Run training loop
#while rolling_mean_loss >= target_rolling_mean_loss: #Use if network should train to target rolling mean loss
    #step += 1
    optimiser.zero_grad()

    rand_gen.seed() #Seed normal random tensor generator

    #Generate positional velocity boundary condition seed by sampling the normal distribution
    x1_seed = torch.empty(1, 1).normal_(mean=x1_bc_mean, std=x1_bc_standard_dev, generator=rand_gen).to(device)
    dx1dt_seed = torch.empty(1, 1).normal_(mean=dx1dt_bc_mean, std=dx1dt_bc_standard_dev, generator=rand_gen).to(device)
    x2_seed = torch.empty(1, 1).normal_(mean=x2_bc_mean, std=x2_bc_standard_dev, generator=rand_gen).to(device)
    dx2dt_seed = torch.empty(1, 1).normal_(mean=dx2dt_bc_mean, std=dx2dt_bc_standard_dev, generator=rand_gen).to(device)

    #Generate variable training boundary condition inputs for the PINN
    x1_seed_train = x1_seed.repeat(t_train.shape)
    dx1dt_seed_train = dx1dt_seed.repeat(t_train.shape)
    x2_seed_train = x2_seed.repeat(t_train.shape)
    dx2dt_seed_train = dx2dt_seed.repeat(t_train.shape)
    train_domain = torch.cat((x1_seed_train, dx1dt_seed_train, x2_seed_train, dx2dt_seed_train, t_train), dim=1)

    u = pinn(train_domain) #Run the network for the entire training time domain to obtain the positions u(t).
    u_1, u_2 = torch.split(u, 1, 1)

    #First differentials wrt coordinates
    du_1dt = torch.autograd.grad(u_1, t_train, torch.ones_like(u_1), create_graph=True)[0] #First differential of f(t) wrt time
    d2u_1dt2 = torch.autograd.grad(du_1dt, t_train, torch.ones_like(du_1dt), create_graph=True)[0] #Second differential of f(t) wrt time

    #Second coordinate differentials
    du_2dt = torch.autograd.grad(u_2, t_train, torch.ones_like(u_2), create_graph=True)[0] #First differential of f(t) wrt time
    d2u_2dt2 = torch.autograd.grad(du_2dt, t_train, torch.ones_like(du_2dt), create_graph=True)[0] #Second differential of f(t) wrt time

    #Compute field losses
    field_loss = (torch.mean((m1*d2u_1dt2 + m1*omega_1_squared*(u_1-a) - k2*(u_2-b))**2) 
                + torch.mean((m2*d2u_2dt2 + m2*omega_2_squared*(u_2-b) - k2*(u_1-a))**2)).requires_grad_(True)

    #Compute field boundary losses
    boundary_loss = ((u_1[0] - x1_seed)**2 + (du_1dt[0] - dx1dt_seed)**2 + (u_2[0] - x2_seed)**2 + (du_2dt[0] - dx2dt_seed)**2).requires_grad_(True) #Positional and velocity losses at t = t_i

    loss_sum = (field_weight*field_loss + boundary_weight*boundary_loss).requires_grad_(True)
    
    rolling_loss_history.append(loss_sum.detach().item()) #Append loss sum to buffer
    if len(rolling_loss_history) >= loss_monitor:
        rolling_loss_history.pop(0) #Remove first element when buffer is full
    rolling_mean_loss = sum(rolling_loss_history) / loss_monitor #Get rolling mean loss for loss_monitor number of training steps

    #Detect collapse states on the loss hypersurface and stop training
    if loss_sum < 0:
        print("\n### Collapse state detected on loss hypersurface! Training stopped. Last loss sum value: ", loss_sum, " ###")
        break

    #Backpropagate the network and optimiser
    loss_sum.backward() #Backpropagate through the network
    optimiser.step() #Update the optimiser
    
    #Record loss every loss_monitor number of steps
    if step % loss_monitor == 0 or rolling_mean_loss <= target_rolling_mean_loss:
        step_history.append(step)
        loss_history.append(loss_sum.detach().to("cpu"))
        dp = 10 #Number of decimal places to round losses to before printing to terminal
        print(f"\nLosses at step {step}:")
        print("            Rolling mean: ", round(rolling_mean_loss, dp))
        print("                     Sum: ", round(loss_sum.detach().to("cpu").item(), dp))
        print("                   Field: ", round(field_loss.detach().to("cpu").item(), dp))
        print("              Boundaries: ", round(boundary_loss.detach().to("cpu").item(), dp))
        #print('Test time ', test_timer_stop-loss_timer_start, 's')

    #Plot the result as the training progresses
    if step % 10000 == 0 or rolling_mean_loss <= target_rolling_mean_loss: #Place a conditional such that a new plot is shown every x steps
        eval_plot_timer_start = time.time()
        no_evals += 1
        pinn.to(eval_device) #Move trained model to evaluation device for evaluation
    
        #Training boundary conditions evaluation
        
        #Generate variable evaluation input for PINN. This must be done since the size of the time and boundary condition tensors must be the same, thus the boundary condition tensor must be repeated to match the evaluation time domain
        x1_seed_eval = x1_seed.repeat(t_eval.shape)
        dx1dt_seed_eval = dx1dt_seed.repeat(t_eval.shape)
        x2_seed_eval = x2_seed.repeat(t_eval.shape)
        dx2dt_seed_eval = dx2dt_seed.repeat(t_eval.shape)
        eval_domain = torch.cat((x1_seed_eval, dx1dt_seed_eval, x2_seed_eval, dx2dt_seed_eval, t_eval), dim=1)

        eval_timer_start = time.time()
        u_eval_rand_bc = pinn(eval_domain)
        eval_timer_stop = time.time()
        eval_time += eval_timer_stop-eval_timer_start #Add time spent evaluating to cumulative total evaluation time.
        u_1_eval_rand_bc, u_2_eval_rand_bc = torch.split(u_eval_rand_bc, 1, 1)
        u_1_eval_rand_bc, u_2_eval_rand_bc = u_1_eval_rand_bc.detach().to("cpu"), u_2_eval_rand_bc.detach().to("cpu")
        
        #User defined boundary conditions evaluation        
        u_eval = pinn(eval_bc_domain)
        u_1_eval, u_2_eval = torch.split(u_eval, 1, 1)
        u_1_eval, u_2_eval = u_1_eval.detach().to("cpu"), u_2_eval.detach().to("cpu")
        
        print("Sample of training boundary conditions: ", x1_seed, dx1dt_seed, x2_seed, dx2dt_seed)

        #Evaluation plotting of time evolution and error between PINN and numerical solutions

        #Training test plots, used to ensure boundary conditions are being varied
        #Numerical solution
        numerical_timer_start = time.time()
        x_1_numerical_position_train, x_1_numerical_velocity_train, x_2_numerical_position_train, x_2_numerical_velocity_train = odeint(dSdt, y0=[x1_seed[0][0].detach().to("cpu"), dx1dt_seed[0][0].detach().to("cpu"), x2_seed[0][0].detach().to("cpu"), dx2dt_seed[0][0].detach().to("cpu")], t=t_eval.detach().to("cpu")[:,0], tfirst=True).T #Get the numerical solution using odeint() over the evaluation time domain and boundary conditions
        numerical_timer_stop = time.time()
        numerical_time += numerical_timer_stop - numerical_timer_start

        #Absolute error between numerical and PINN solutions
        x1_abs_err = np.abs(x_1_numerical_position - u_1_eval.squeeze().numpy())
        x2_abs_err = np.abs(x_2_numerical_position - u_2_eval.squeeze().numpy())

        #Training time evolution
        train_fig = plt.figure(figsize=(15, 5))
        train_ax1 = train_fig.add_subplot(111)
        train_ax1.plot(t_eval.detach().to("cpu"), u_1_eval_rand_bc.detach().to("cpu"), 'red', linewidth=2, marker=None, label='$x_{1,\\text{P}}$')
        train_ax1.plot(t_eval.detach().to("cpu"), u_2_eval_rand_bc.detach().to("cpu"), 'blue', linewidth=2, marker=None, label='$x_{2,\\text{P}}$')
        train_ax1.plot(t_eval.detach().to("cpu"), x_1_numerical_position_train, 'orange', linestyle=(0, (1, 2)), marker=None, linewidth=6, label='$x_{1,\\text{N}}$')
        train_ax1.plot(t_eval.detach().to("cpu"), x_2_numerical_position_train, 'green', linestyle=(0, (1, 2)), marker=None, linewidth=6, label='$x_{2,\\text{N}}$')
        train_ax1.set_xlabel('$t$ (s)', fontsize=40)
        train_ax1.set_ylabel('$x$ (m)', fontsize=40)
        train_ax1.tick_params(axis='both', which='major', labelsize=32)
        train_ax1.grid(True)
        train_ax1.legend(fontsize=40, bbox_to_anchor=(1, 1.05))
        train_fig.tight_layout()
        plt.savefig(os.path.join(train_dir[0], f"{step}")) #Save figure in the /output/train directory
        plt.close() #Close each plot once saved

        #Evaluation time evolution
        eval_fig = plt.figure(figsize=(15, 5))
        eval_ax1 = eval_fig.add_subplot(111)
        eval_ax1.plot(t_eval.to("cpu"), u_1_eval, 'red', linewidth=2, marker=None, label='$x_{1,\\text{P}}$')
        eval_ax1.plot(t_eval.to("cpu"), u_2_eval, 'blue', linewidth=2, marker=None, label='$x_{2,\\text{P}}$')
        eval_ax1.plot(t_eval.to("cpu"), x_1_numerical_position, 'orange', linestyle=(0, (1, 2)), marker=None, linewidth=6, label='$x_{1,\\text{N}}$')
        eval_ax1.plot(t_eval.to("cpu"), x_2_numerical_position, 'green', linestyle=(0, (1, 2)), marker=None, linewidth=6, label='$x_{2,\\text{N}}$')
        eval_ax1.set_xlabel('$t$ (s)', fontsize=40)
        eval_ax1.set_ylabel('$x$ (m)', fontsize=40)
        eval_ax1.tick_params(axis='both', which='major', labelsize=32)
        eval_ax1.grid(True)
        eval_ax1.legend(fontsize=40, bbox_to_anchor=(1, 1.05))
        eval_fig.tight_layout()
        plt.savefig(os.path.join(eval_dir[0], f"{step}")) #Save figure in the /output/evaluation directory
        plt.close() #Close each plot once saved

        #Evaluation absolute error
        err_fig = plt.figure(figsize=(15, 5))
        err_ax1 = err_fig.add_subplot(111)
        err_ax1.plot(t_eval.to("cpu"), x1_abs_err, 'red', linewidth=2, marker=None, label='$x_{1}$')
        err_ax1.plot(t_eval.to("cpu"), x2_abs_err, 'blue', linewidth=2, marker=None, label='$x_{2}$')
        err_ax1.set_xlabel('$t$ (s)', fontsize=40)
        err_ax1.set_ylabel('Absolute error (m)', fontsize=36)
        err_ax1.tick_params(axis='both', which='major', labelsize=32)
        err_ax1.grid(True)
        err_ax1.legend(fontsize=40, bbox_to_anchor=(1, 1.05))
        err_fig.tight_layout()
        plt.savefig(os.path.join(err_dir[0], f"{step}")) #Save figure in the /output/error directory
        plt.close() #Close each plot once saved

        pinn.to(device) #Move model back to training device
        eval_plot_timer_stop = time.time()
        eval_plot_time += eval_plot_timer_stop-eval_plot_timer_start #Add time spent plotting to eval_plot_clock
total_timer_stop = time.time() #Encapsulate code to get evaluation time
total_time = total_timer_stop-total_timer_start #Get total training, evaluation and plotting time
training_time = total_time - eval_plot_time #Get total time spent training only
evaluation_time_mean = eval_time / no_evals
numerical_time_mean = numerical_time / no_evals
print(f"\nTraining complete.")
print(f"Total training time: {training_time}s")
print(f"Mean evaluation time: {evaluation_time_mean}s")
print(f"Mean numerical evaluation time: {numerical_time_mean}s")
