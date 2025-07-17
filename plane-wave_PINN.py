'''
Plane-wave PINN for two coupled oscillators
Rory Clements, 18/6/25

Create and train a PINN for two coupled oscillators for variable boundary conditions.
The novelty of this PINN is that it uses a "plane-wave" activation function in the
penultimate layer, lending it well to decoupling the training from a single set of
boundary conditions.

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
    - This version was used to generate the results for the plane-wave PINNs in the 1D PWPINNs paper. 
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

#Exponential prefactor plane wave expansion
class complex_exponential(nn.Module):
    def __init__(self, in_features):
        #in_features: shape of the input
        super(complex_exponential, self).__init__()
        self.in_features = in_features

    def forward(self, F, omega, t):
        return torch.real(F * torch.exp(1j * omega * t)).requires_grad_(True) #Return plane waves

class NeuralNetwork(nn.Module):
    #Define neural network
    def __init__(self, no_omega_steps):
        super().__init__()

        #Initialise layers
        self.fc0_1 = nn.Flatten(0, 1)
        self.fc1 = nn.Linear(4, 32)
        self.fc1_1 = nn.Dropout(p=0.00)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, no_omega_steps)
        self.fc4 = nn.Linear(no_omega_steps, no_omega_steps)
        self.fc5 = nn.Linear(no_omega_steps, 2)
        
        #Initialise activation functions
        self.a1 = nn.Tanh() #Tanh activation for first layer
        self.a2 = nn.Sigmoid() #Sigmoid activation function
        self.a3 = complex_exponential(no_omega_steps) #Complex exponential activation function with # trainable parameters as amplitude prefactors for each plane wave 
                                                      #Number of trainable parameters must match the number of neurons in the layer passed to the activation function.

    def forward(self, seed, omega, t):
        seed = self.fc0_1(seed)
        unknown = self.a1(self.fc1(seed))
        unknown = self.a1(self.fc2(unknown))
        F_coefficients = self.a2(self.fc3(unknown)) 
        f_t = self.a3(self.fc4(F_coefficients), omega, t) #Apply complex_exponential activation
        u_t = self.fc5(f_t) #Sum over all the modified plane waves with no activation
        return u_t, F_coefficients

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
    #Creates a tensor of boundary conditions centred around the means in bc_means_tensor
    #with standard deviations given by bc_sigma_tensor.
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

def genBCSpectrum(spectrum_sigma, spectrum_steps):
    #Generate boundary condition spectrum for seeding PINN in order to generate solution spectrum
    x1_a, v1_a, x2_a, v2_a = x1_bc_mean-(spectrum_sigma), dx1dt_bc_mean-(spectrum_sigma), x2_bc_mean-(spectrum_sigma), dx2dt_bc_mean-(spectrum_sigma) #Initial boundary conditions from which to begin the sweep
    x1_b, v1_b, x2_b, v2_b = x1_bc_mean+(spectrum_sigma), dx1dt_bc_mean+(spectrum_sigma), x2_bc_mean+(spectrum_sigma), dx2dt_bc_mean+(spectrum_sigma) #Final boundary conditions at which to end the sweep
    d1, d2, d3, d4 = np.abs(x1_b-x1_a), np.abs(v1_b-v1_a), np.abs(x2_b-x2_a), np.abs(v2_b-v2_a) #Get differences between boundary conditions
    d1_step, d2_step, d3_step, d4_step = d1/spectrum_steps, d2/spectrum_steps, d3/spectrum_steps, d4/spectrum_steps #Get step sizes for each boundary condition 
    x1_bc_iter, v1_bc_iter, x2_bc_iter, v2_bc_iter = x1_a, v1_a, x2_a, v2_a
    bc_spectrum = torch.zeros(spectrum_steps, 2, 2) #List to hold individual solutions for range of boundary conditions
    for spectrum_step in range(spectrum_steps):
        #Increment boundary conditions
        x1_bc_iter += d1_step
        v1_bc_iter += d2_step
        x2_bc_iter += d3_step
        v2_bc_iter += d4_step
        bc_spectrum[spectrum_step] = torch.FloatTensor([[x1_bc_iter, v1_bc_iter], [x2_bc_iter, v2_bc_iter]]).to(device) #Append set of boundary conditions to the spectrum
    bc_spectrum = bc_spectrum.to(device)
    return bc_spectrum

def genBCSoln(bc_spectrum, omega, t_eval):
    #Generate PINN and numerical solution spectra from boundary condition spectrum and return relative errors for plotting and mean errors for histogram (in dB)
    PINN_spectrum = torch.zeros(2, bc_spectrum.shape[0], t_eval.shape[0])
    relative_error = np.zeros((2, bc_spectrum.shape[0], t_eval.shape[0]))
    for soln in range(bc_spectrum.shape[0]): #Solve over boundary condition spectrum
        #PINN solution
        u_test, F_coefficients_test = (pinn(bc_spectrum[soln], omega, t_eval))
        u_1_test, u_2_test = torch.split(u_test, 1, 1)
        u_1_test, u_2_test = u_1_test.detach().to("cpu"), u_2_test.detach().to("cpu")
        PINN_spectrum[0][soln] = u_1_test.squeeze()
        PINN_spectrum[1][soln] = u_2_test.squeeze()
        
        #Numerical solution
        x_1_numerical_test, x_1_numerical_velocity_test, x_2_numerical_test, x_2_numerical_velocity_test = odeint(dSdt, y0=[bc_spectrum[soln][0][0].detach().to("cpu"), bc_spectrum[soln][0][1].detach().to("cpu"), bc_spectrum[soln][1][0].detach().to("cpu"), bc_spectrum[soln][1][1].detach().to("cpu")], t=t_eval.detach().to("cpu")[:,0], tfirst=True).T #Get the numerical solution using odeint() over the evaluation time domain
        
        #Relative error (dB)
        relative_error[0][soln] = 10*np.log10(np.abs(u_1_test.squeeze() - x_1_numerical_test) / np.abs(x_1_numerical_test))
        relative_error[1][soln] = 10*np.log10(np.abs(u_2_test.squeeze() - x_2_numerical_test) / np.abs(x_2_numerical_test))
    
    relative_error[relative_error < -10] = -10 #Cutoff low values
    relative_error[relative_error > 0] = 0 #Cutoff large values
    
    #Get sum of errors (dB) for each boundary condition sample
    mean_error = np.zeros((2, bc_spectrum.shape[0]))
    mean_error[0] = np.mean(relative_error[0], axis=1) 
    mean_error[1] = np.mean(relative_error[1], axis=1) 

    return relative_error, mean_error

#==============
#INITIALIZATION
#==============

device='cuda' #Device on which to train the network. Will automatically switch to cpu if gpu is not available.
eval_device = "cuda" #Device on which to evaluate the trained network. It can be advantageous to evaluate on the CPU, since RAM is cheaper than vRAM for large tensors.
device = IsGPU() #Get status of GPU and switch compute device to it

#Create directory "output"
out_directory = dirOps() #Create out_directory object
out_directory.newDir("output")
dir_state = out_directory.CreateDir() #Get the path to the new output directory along with weather it had to be created

#Create directory "output/training"
out_directory.newDir("output/training")
train_dir = out_directory.CreateDir() #Get the path to the new output directory along with weather it had to be created
out_directory.empty_dir() #Attempt to empty directory. No action will be taken if directory is new.

#Create directory "output/evaluation"
out_directory.newDir("output/evaluation")
eval_dir = out_directory.CreateDir() #Get the path to the new output directory along with weather it had to be created
out_directory.empty_dir() #Attempt to empty directory. No action will be taken if directory is new.

#Create directory "output/error"
out_directory.newDir("output/error")
err_dir = out_directory.CreateDir() #Get the path to the new output directory along with whether it had to be created
out_directory.empty_dir() #Attempt to empty directory. No action will be taken if directory is new.

#Create directory "output/testing"
out_directory.newDir("output/testing")
test_dir = out_directory.CreateDir() #Get the path to the new output directory along with whether it had to be created
out_directory.empty_dir() #Attempt to empty directory. No action will be taken if directory is new.

#=======================================
#DEFINE INDEPENDENT SIMULATION VARIABLES
#=======================================

#Training control variables
target_rolling_mean_loss = 2e-2 # Target rolling mean loss. In order to run the training loop to a specific number of training steps, set this to some unachievable small value, such as 1e-9.
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
#omega_i, omega_step_no, standard_deviation  = 0, 500, 0.013 #Uncomment if using random frequency intervals
omega_i, omega_f, omega_step_no = 0, 5, 100 #Uncomment if using fixed frequency intervals

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

#Frequency spectrum
#For random frequency spectrum intervals, uncomment lines below
'''
omega = torch.zeros(omega_step_no) #Create empty array of omega values
rand_gen_omega = torch.Generator(device='cpu') #Initialise random tensor seed
rand_gen_omega.seed() #Seed normal random tensor generator
omega_steps = torch.abs(torch.empty(omega_step_no).normal_(mean=0, std=standard_deviation, generator=rand_gen_omega)) #Create normal distribution of steps in omega
#Add normal distribution of omega steps to omega such that a monotonically increasing array is formed
for iter in range(len(omega)-1):
    omega[iter+1] = omega[iter] + omega_steps[iter]
omega = omega.to(device)
print("Max omega value: ", torch.max(omega).to("cpu").numpy())
'''
#For fixed frequency spectrum intervals, uncomment line below
omega = torch.linspace(omega_i, omega_f, omega_step_no).to(device) #Set of frequencies to use in activation function for the penultimate layer

#Equations of motion variables
omega_1_squared = (k1+k2)/m1
omega_2_squared = (k2+k3)/m2

#==================
#NUMERICAL SOLUTION
#==================

S_0 = [x1_0, v1_0, x2_0, v2_0] #Vector of boundary conditions for S

x_1_numerical_position, x_1_numerical_velocity, x_2_numerical_position, x_2_numerical_velocity = odeint(dSdt, y0=S_0, t=t_eval.detach().to("cpu")[:,0], tfirst=True).T #Get the numerical solution using odeint()

#==========
#TRAIN PINN
#==========

torch.manual_seed(123) #Fix seed for network

#Initialise tensor of normal distribution and sample size
rand_gen = torch.Generator(device='cpu') #Initialise random tensor seed

#Declare and initialise PINN
pinn = NeuralNetwork(omega_step_no).to(device) #Declare PINN and move it to the compute device
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
#for step in range(1000001): #Run training loop
while rolling_mean_loss >= target_rolling_mean_loss: #Use if network should train to target rolling mean loss
    step += 1
    optimiser.zero_grad()

    rand_gen.seed() #Seed normal random tensor generator

    #Generate positional velocity boundary condition seed by sampling the normal distribution
    x1_seed = torch.empty(1, 1).normal_(mean=x1_bc_mean, std=x1_bc_standard_dev, generator=rand_gen).to(device)
    dx1dt_seed = torch.empty(1, 1).normal_(mean=dx1dt_bc_mean, std=dx1dt_bc_standard_dev, generator=rand_gen).to(device)
    x2_seed = torch.empty(1, 1).normal_(mean=x2_bc_mean, std=x2_bc_standard_dev, generator=rand_gen).to(device)
    dx2dt_seed = torch.empty(1, 1).normal_(mean=dx2dt_bc_mean, std=dx2dt_bc_standard_dev, generator=rand_gen).to(device)
    
    physics_normalised = torch.tensor([[x1_seed, dx1dt_seed], [x2_seed, dx2dt_seed]]).to(device) #Assemble seed into tensor in preparation for passing to the network
    
    u, F_coefficients = pinn(physics_normalised, omega, t_train) #Run the network for the entire training time domain to obtain the positions u(t).
    u_1, u_2 = torch.split(u, 1, 1)

    #First differentials wrt coordinates
    du_1dt = torch.autograd.grad(u_1, t_train, torch.ones_like(u_1), create_graph=True)[0] #First differential of f(t) wrt time
    d2u_1dt2 = torch.autograd.grad(du_1dt, t_train, torch.ones_like(du_1dt), create_graph=True)[0] #Second differential of f(t) wrt time

    #Second coordinate differentials
    du_2dt = torch.autograd.grad(u_2, t_train, torch.ones_like(u_2), create_graph=True)[0] #First differential of f(t) wrt time
    d2u_2dt2 = torch.autograd.grad(du_2dt, t_train, torch.ones_like(du_2dt), create_graph=True)[0] #Second differential of f(t) wrt time
    dF = torch.gradient(F_coefficients)[0] #Get gradient of F for smoothing loss
    
    #Compute field losses
    field_loss = (torch.mean((m1*d2u_1dt2 + m1*omega_1_squared*(u_1-a) - k2*(u_2-b))**2) 
                + torch.mean((m2*d2u_2dt2 + m2*omega_2_squared*(u_2-b) - k2*(u_1-a))**2)).requires_grad_(True)

    #Compute field boundary losses
    boundary_loss = ((u_1[0] - physics_normalised[0][0])**2 + (du_1dt[0] - physics_normalised[0][1])**2 + (u_2[0] - physics_normalised[1][0])**2 + (du_2dt[0] - physics_normalised[1][1])**2).requires_grad_(True) #Positional and velocity losses at t = t_i

    #Other losses
    F_smoothing_loss = torch.mean(dF**2) #First derivative smoothing loss for F coefficients
    high_frequency_loss = torch.mean(torch.abs(omega * F_coefficients)) #Penalize high frequencies such that solution builds from low frequencies first

    loss_sum = (field_weight*field_loss + boundary_weight*boundary_loss + F_smooth_weight*F_smoothing_loss + high_frequency_loss).requires_grad_(True)
    
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
        print("             F Smoothing: ", round(F_smoothing_loss.detach().to("cpu").item(), dp))
        print("          High frequency: ", round(high_frequency_loss.detach().to("cpu").item(), dp))
        #print('Test time ', test_timer_stop-loss_timer_start, 's')

    #Plot the result as the training progresses
    if step % 10000 == 0 or rolling_mean_loss <= target_rolling_mean_loss: #Place a conditional such that a new plot is shown every x steps
        eval_plot_timer_start = time.time()
        no_evals += 1
        pinn.to(eval_device) #Move trained model to evaluation device for evaluation
        
        #Training boundary conditions evaluation
        u_eval_rand_bc, F_coefficients_eval_rand_bc = (pinn(physics_normalised, omega, t_eval))
        u_1_eval_rand_bc, u_2_eval_rand_bc = torch.split(u_eval_rand_bc, 1, 1)
        u_1_eval_rand_bc, u_2_eval_rand_bc = u_1_eval_rand_bc.detach().to("cpu"), u_2_eval_rand_bc.detach().to("cpu")
        
        #User defined boundary conditions evaluation
        evaluate_normalised = torch.FloatTensor([[x1_0, v1_0], [x2_0, v2_0]]).to(device) #Seed network with boundary conditions
        eval_timer_start = time.time()
        u_eval, F_coefficients_eval = pinn(evaluate_normalised, omega, t_eval)
        eval_timer_stop = time.time()
        eval_time += eval_timer_stop-eval_timer_start #Add time spent evaluating to cumulative total evaluation time.
        u_1_eval, u_2_eval = torch.split(u_eval, 1, 1)
        u_1_eval, u_2_eval = u_1_eval.detach().to("cpu"), u_2_eval.detach().to("cpu")
        
        print("Sample of training boundary conditions: ", physics_normalised)

        #Evaluation plotting of time evolution and error between PINN and numerical solutions

        #Training test plots, used to ensure boundary conditions are being varied
        #Numerical solution
        numerical_timer_start = time.time()
        x_1_numerical_position_train, x_1_numerical_velocity_train, x_2_numerical_position_train, x_2_numerical_velocity_train = odeint(dSdt, y0=[physics_normalised[0][0].detach().to("cpu"), physics_normalised[0][1].detach().to("cpu"), physics_normalised[1][0].detach().to("cpu"), physics_normalised[1][1].detach().to("cpu")], t=t_eval.detach().to("cpu")[:,0], tfirst=True).T #Get the numerical solution using odeint() over the evaluation time domain and boundary conditions
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
        error_fig = plt.figure(figsize=(15, 5))
        error_ax1 = error_fig.add_subplot(111)
        error_ax1.plot(t_eval.to("cpu"), x1_abs_err, 'red', linewidth=2, marker=None, label='$x_{1}$')
        error_ax1.plot(t_eval.to("cpu"), x2_abs_err, 'blue', linewidth=2, marker=None, label='$x_{2}$')
        error_ax1.set_xlabel('$t$ (s)', fontsize=40)
        error_ax1.set_ylabel('Absolute error (m)', fontsize=36)
        error_ax1.tick_params(axis='both', which='major', labelsize=32)
        error_ax1.grid(True)
        error_ax1.legend(fontsize=40, bbox_to_anchor=(1, 1.05))
        error_fig.tight_layout()
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

#Test network by evaluating over a range of values and plotting as a 2D surface
print("\nEvaluating network.\n")

# Generate BC spectrum using a larger normal distribution sigma 
# than was used for training for plotting the error and error histogram
bc_spectrum = genBCSpectrum(spectrum_sigma=5, spectrum_steps=1000) # Generate spectrum of boundary conditions
relative_error, mean_error = genBCSoln(bc_spectrum, omega, t_eval) # Generate relative error spectrum and mean errors for histogram

# Generate BC spectrum using the same normal distribution sigma 
# as was used for training in order to generate comparison error histogram
reduced_bc_spectrum = genBCSpectrum(spectrum_sigma=1, spectrum_steps=1000) # Generate spectrum of boundary conditions
reduced_relative_error, reduced_mean_error = genBCSoln(reduced_bc_spectrum, omega, t_eval) # Generate relative error spectrum and mean errors for histogram

#Plotting
mean_variation = (torch.mean(bc_spectrum, (1, 2)) - torch.mean(evaluate_normalised)) # Required to scale the bc axis for plotting

#Heat map plotting

#First coordinate
diff_fig = plt.figure(figsize=(11, 18))
spec = gridspec.GridSpec(ncols=1, nrows=2, width_ratios=[1], wspace=0, hspace=0.0, height_ratios=[1, 1])
spec.update(left=0.16, right=0.82, top=0.99, bottom=0.05)
ax1 = diff_fig.add_subplot(spec[0])
surf_difference_x1 = ax1.imshow(relative_error[0],  extent=(np.amin(t_eval.to("cpu").squeeze().numpy()), np.amax(t_eval.to("cpu").squeeze().numpy()), np.amin(mean_variation.to("cpu").numpy()), np.amax(mean_variation.to("cpu").numpy())))
ax1.set_xlabel('$t$ (s)', fontsize=40, labelpad=15)
ax1.set_ylabel('BC', fontsize=40, labelpad=15)
ax1.tick_params(axis='both', which='major', labelsize=32, pad=15)
cbar_x1 = diff_fig.colorbar(surf_difference_x1, ax=ax1, fraction=0.046, pad=0.02)
cbar_x1.set_label('$x_1$ relative error (dB)', fontsize=40, labelpad=15)
cbar_x1.ax.tick_params(labelsize=32, pad=15)

#Second coordinate
ax2 = diff_fig.add_subplot(spec[1])
surf_difference_x2 = ax2.imshow(relative_error[1], extent=(np.amin(t_eval.to("cpu").squeeze().numpy()), np.amax(t_eval.to("cpu").squeeze().numpy()), np.amin(mean_variation.to("cpu").numpy()), np.amax(mean_variation.to("cpu").numpy())))
ax2.set_xlabel('$t$ (s)', fontsize=40, labelpad=15)
ax2.set_ylabel('BC', fontsize=40, labelpad=15)
ax2.tick_params(axis='both', which='major', labelsize=32, pad=15)
cbar_x2 = diff_fig.colorbar(surf_difference_x2, ax=ax2, fraction=0.046, pad=0.02)
cbar_x2.set_label('$x_2$ relative error (dB)', fontsize=40, labelpad=15)
cbar_x2.ax.tick_params(labelsize=32, pad=15)
plt.savefig(os.path.join(test_dir[0], "error_plot")) #Safe figure in the /output/testing directory

# Plot mean error as a histogram for extended normal distribution sigma
error_fig = plt.figure(figsize=(15, 5))
error_ax1 = error_fig.add_subplot(111)
error_ax1.hist([mean_error[0], mean_error[1]], bins=10, color=['blue', 'red'], alpha=1, label=['$x_1$', '$x_2$'])
error_ax1.set_xlabel('Mean error (dB)', fontsize=40)
error_ax1.set_ylabel('Number', fontsize=40)
error_ax1.tick_params(axis='both', which='major', labelsize=32)
error_ax1.grid(True)
error_ax1.legend(fontsize=40)
error_fig.tight_layout()
plt.savefig(os.path.join(test_dir[0], "extended_error_histogram")) #Safe figure in the /output/testing directory

# Plot mean error as a histogram for the same normal distribution sigma
# as was used for training the PINN
error_fig = plt.figure(figsize=(15, 5))
error_ax1 = error_fig.add_subplot(111)
error_ax1.hist([reduced_mean_error[0], reduced_mean_error[1]], bins=10, color=['blue', 'red'], alpha=1, label=['$x_1$', '$x_2$'])
error_ax1.set_xlabel('Mean error (dB)', fontsize=40)
error_ax1.set_ylabel('Number', fontsize=40)
error_ax1.tick_params(axis='both', which='major', labelsize=32)
error_ax1.grid(True)
error_ax1.legend(fontsize=40)
error_fig.tight_layout()
plt.savefig(os.path.join(test_dir[0], "reduced_error_histogram")) #Safe figure in the /output/testing directory
