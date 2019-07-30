# SCRIPT FOR SOLVING THE MERTON PROBLEM

#%% import needed packages

import DGM
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#%% Parameters 

# Merton problem parameters 
r = 0.05      # risk-free rate
mu = 0.2      # asset drift
sigma = 0.25  # asset volatility
gamma = 1     # exponential utility preference parameter
T = 1         # terminal time (investment horizon)

# Solution parameters (domain on which to solve PDE)
t_low = 0 + 1e-10    # time lower bound
X_low = 0.0 + 1e-10  # wealth lower bound
X_high = 1           # wealth upper bound

# neural network parameters
num_layers = 3
nodes_per_layer = 50
starting_learning_rate = 0.001

# Training parameters
sampling_stages  = 500   # number of times to resample new time-space domain points
steps_per_sample = 10    # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_interior = 1000
nSim_terminal = 100

# multipliers for oversampling i.e. draw X from [X_low - X_oversample, X_high + X_oversample]
X_oversample = 0.5
t_oversample = 0.5

# Plot options
n_plot = 41  # Points on plot grid for each dimension

# Save options
saveOutput = False
saveName   = 'MertonProblem'
saveFigure = False
figureName = 'MertonProblem'

#%% Analytical Solution

# market price of risk
theta = (mu - r)/sigma


def exponential_utility(x): 
    ''' Compute exponential utility for given level of wealth
    
    Args:
        x: wealth
    '''
    
    return -tf.exp(-gamma*x)

def value_function_analytical_solution(t,x):
    ''' Compute the value function for the Merton problem
    
    Args:
        t: time points
        x: space (wealth) points        
    '''
    
    return -np.exp(-x*gamma*np.exp(r*(T - t)) - (T - t)*0.5*theta**2)    

def optimal_control_analytical_solution(t,x):
    ''' Compute the optimal control for the Merton problem
    
    Args:
        t: time points
        x: space (wealth) points        
    '''
    
    return theta/(gamma*sigma) * np.exp(-r*(T - t))


#%% Sampling function - randomly sample time-space pairs

def sampler(nSim_interior, nSim_terminal):
    ''' Sample time-space points from the function's domain; points are sampled
        uniformly on the interior of the domain, at the initial/terminal time points
        and along the spatial boundary at different time points. 
    
    Args:
        nSim_interior: number of space points in the interior of the function's domain to sample 
        nSim_terminal: number of space points at terminal time to sample (terminal condition)
    ''' 
    
    # Sampler #1: domain interior    
    t_interior = np.random.uniform(low=t_low - 0.5*(T-t_low), high=T, size=[nSim_interior, 1])
#    t_interior = np.random.uniform(low=t_low - t_oversample, high=T, size=[nSim_interior, 1])
    X_interior = np.random.uniform(low=X_low - 0.5*(X_high-X_low), high=X_high + 0.5*(X_high-X_low), size=[nSim_interior, 1])
#    X_interior = np.random.uniform(low=X_low - X_oversample, high=X_high + X_oversample, size=[nSim_interior, 1])
#    X_interior = np.random.uniform(low=X_low * X_multiplier_low, high=X_high * X_multiplier_high, size=[nSim_interior, 1])

    # Sampler #2: spatial boundary
        # no spatial boundary condition for this problem
    
    # Sampler #3: initial/terminal condition
    t_terminal = T * np.ones((nSim_terminal, 1))
#    X_terminal = np.random.uniform(low=X_low - X_oversample, high=X_high + X_oversample, size = [nSim_terminal, 1])
    X_terminal = np.random.uniform(low=X_low - 0.5*(X_high-X_low), high=X_high + 0.5*(X_high-X_low), size = [nSim_terminal, 1])
#    X_terminal = np.random.uniform(low=X_low * X_multiplier_low, high=X_high * X_multiplier_high, size = [nSim_terminal, 1])
    
    return t_interior, X_interior, t_terminal, X_terminal

#%% Loss function for Merton Problem PDE

def loss(model, t_interior, X_interior, t_terminal, X_terminal):
    ''' Compute total loss for training.
    
    Args:
        model:      DGM model object
        t_interior: sampled time points in the interior of the function's domain
        X_interior: sampled space points in the interior of the function's domain
        t_terminal: sampled time points at terminal point (vector of terminal times)
        X_terminal: sampled space points at terminal time
    ''' 
    
    # Loss term #1: PDE
    # compute function value and derivatives at current sampled points
    V = model(t_interior, X_interior)
    V_t = tf.gradients(V, t_interior)[0]
    V_x = tf.gradients(V, X_interior)[0]
    V_xx = tf.gradients(V_x, X_interior)[0]
    diff_V = -0.5 * theta**2 * V_x**2 + (V_t + r*X_interior*V_x)*V_xx    

    # compute average L2-norm of differential operator
    L1 = tf.reduce_mean(tf.square(diff_V)) 
    
    # Loss term #2: boundary condition
        # no boundary condition for this problem
    
    # Loss term #3: initial/terminal condition
    target_terminal = exponential_utility(X_terminal)
    fitted_terminal = model(t_terminal, X_terminal)
    
    L3 = tf.reduce_mean( tf.square(fitted_terminal - target_terminal) )

    return L1, L3
    

#%% Set up network

# initialize DGM model (last input: space dimension = 1)
model = DGM.DGMNet(nodes_per_layer, num_layers, 1)

# tensor placeholders (_tnsr suffix indicates tensors)
# inputs (time, space domain interior, space domain at initial time)
t_interior_tnsr = tf.placeholder(tf.float32, [None,1])
X_interior_tnsr = tf.placeholder(tf.float32, [None,1])
t_terminal_tnsr = tf.placeholder(tf.float32, [None,1])
X_terminal_tnsr = tf.placeholder(tf.float32, [None,1])

# loss 
L1_tnsr, L3_tnsr = loss(model, t_interior_tnsr, X_interior_tnsr, t_terminal_tnsr, X_terminal_tnsr)
loss_tnsr = L1_tnsr + L3_tnsr

# value function
V = model(t_interior_tnsr, X_interior_tnsr)

# optimal control computed numerically from fitted value function 
def compute_fitted_optimal_control(V):
    V_x = tf.gradients(V, X_interior_tnsr)[0]
    V_xx = tf.gradients(V_x, X_interior_tnsr)[0]
    return -(theta/(gamma*sigma))*V_x/(V_xx)

numerical_optimal_control = compute_fitted_optimal_control(V)

# set optimizer - NOTE THIS IS DIFFERENT FROM OTHER APPLICATIONS!
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step,100000, 0.96, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)

# initialize variables
init_op = tf.global_variables_initializer()

# open session
sess = tf.Session()
sess.run(init_op)

#%% Train network
# initialize loss per training
loss_list = []

# for each sampling stage
for i in range(sampling_stages):
    
    # sample uniformly from the required regions
    t_interior, X_interior, t_terminal, X_terminal = sampler(nSim_interior, nSim_terminal)
    
    # for a given sample, take the required number of SGD steps
    for _ in range(steps_per_sample):
        loss,L1,L3,_ = sess.run([loss_tnsr, L1_tnsr, L3_tnsr, optimizer],
                                feed_dict = {t_interior_tnsr:t_interior, X_interior_tnsr:X_interior, t_terminal_tnsr:t_terminal, X_terminal_tnsr:X_terminal})
        loss_list.append(loss)
    
    print(loss, L1, L3, i)

# save outout
if saveOutput:
    saver = tf.train.Saver()
    saver.save(sess, './SavedNets/' + saveName)

#%% Plot value function results

# LaTeX rendering for text in plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# figure options
plt.figure()
plt.figure(figsize = (12,10))

# time values at which to examine density
valueTimes = [t_low, T/3, 2*T/3, T]

# vector of t and S values for plotting
X_plot = np.linspace(X_low, X_high, n_plot)

for i, curr_t in enumerate(valueTimes):
    
    # specify subplot
    plt.subplot(2,2,i+1)
    
    # simulate process at current t 
    optionValue = value_function_analytical_solution(curr_t, X_plot)
    
    # compute normalized density at all x values to plot and current t value
    t_plot = curr_t * np.ones_like(X_plot.reshape(-1,1))
    fitted_optionValue = sess.run([V], feed_dict= {t_interior_tnsr:t_plot, X_interior_tnsr:X_plot.reshape(-1,1)})
    
    # plot histogram of simulated process values and overlay estimated density
    plt.plot(X_plot, optionValue, color = 'b', label='Analytical Solution', linewidth = 3, linestyle=':')
    plt.plot(X_plot, fitted_optionValue[0], color = 'r', label='DGM estimate')    
    
    # subplot options
    plt.xlim(xmin=0.0, xmax=X_high)
    plt.xlabel(r"Wealth", fontsize=15, labelpad=10)
    plt.ylabel(r"Value Function", fontsize=15, labelpad=20)
    plt.title(r"\boldmath{$t$}\textbf{ = %.2f}"%(curr_t), fontsize=18, y=1.03)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(linestyle=':')
    
    if i == 0:
        plt.legend(loc='upper left', prop={'size': 16})
    
# adjust space between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.4)

if saveFigure:
    plt.savefig(figureName + '_valueFunction.png')


#%% Plot optimal control results

# figure options
plt.figure()
plt.figure(figsize = (12,10))

# time values at which to examine density
valueTimes = [t_low, T/3, 2*T/3, T]

# vector of t and S values for plotting
X_plot = np.linspace(X_low, X_high, n_plot)

for i, curr_t in enumerate(valueTimes):
    
    # specify subplot
    plt.subplot(2,2,i+1)
    
    # simulate process at current t 
    optimal_control = optimal_control_analytical_solution(t_plot,X_plot)
    
    # compute normalized density at all x values to plot and current t value
    t_plot = curr_t * np.ones_like(X_plot.reshape(-1,1))
    fitted_optimal_control = sess.run([numerical_optimal_control],feed_dict={t_interior_tnsr:t_plot, X_interior_tnsr:X_plot.reshape(-1,1)})
    
    # plot histogram of simulated process values and overlay estimated density
    plt.plot(X_plot, optimal_control, color = 'b', label='Analytical Solution', linewidth = 3, linestyle=':')
    plt.plot(X_plot, fitted_optimal_control[0], color = 'r', label='DGM estimate')    
    
    # subplot options
    plt.xlim(xmin=0.0, xmax=X_high)
    plt.ylim(ymin=2.0, ymax=3.5)
    plt.xlabel(r"Wealth", fontsize=15, labelpad=10)
    plt.ylabel(r"Value Function", fontsize=15, labelpad=20)
    plt.title(r"\boldmath{$t$}\textbf{ = %.2f}"%(curr_t), fontsize=18, y=1.03)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(linestyle=':')
    
    if i == 0:
        plt.legend(loc='upper left', prop={'size': 16})
    
# adjust space between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.4)

if saveFigure:
    plt.savefig(figureName + '_optimalControl.png')

#%% Error heatmaps - value function
# vector of t and X values for plotting
X_plot = np.linspace(X_low, X_high, n_plot)
t_plot = np.linspace(t_low, T, n_plot)

# compute value function for each (t,X) pair
value_function_mesh = np.zeros([n_plot, n_plot])

for i in range(n_plot):
    for j in range(n_plot):
    
        value_function_mesh[j,i] = value_function_analytical_solution(t_plot[i], X_plot[j])
    
# compute model-implied value function for each (t,X) pair
t_mesh, X_mesh = np.meshgrid(t_plot, X_plot)

t_plot = np.reshape(t_mesh, [n_plot**2,1])
X_plot = np.reshape(X_mesh, [n_plot**2,1])

fitted_value_function = sess.run([V], feed_dict= {t_interior_tnsr:t_plot, X_interior_tnsr:X_plot})
fitted_value_function_mesh = np.reshape(fitted_value_function, [n_plot, n_plot])

# PLOT ABSOLUTE ERROR
plt.figure()
plt.figure(figsize = (8,6))

plt.pcolormesh(t_mesh, X_mesh, np.abs(value_function_mesh - fitted_value_function_mesh), cmap = "rainbow")

# plot options
plt.colorbar()
plt.title("Absolute Error", fontsize=20)
plt.ylabel("Wealth", fontsize=15, labelpad=10)
plt.xlabel("Time", fontsize=15, labelpad=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

if saveFigure:
    plt.savefig(figureName + '_valueFunction_absErr.png')

# PLOT RELATIVE ERROR
plt.figure()
plt.figure(figsize = (8,6))

plt.pcolormesh(t_mesh, X_mesh, np.abs(1 - np.divide(fitted_value_function_mesh, value_function_mesh)), cmap = "rainbow")

# plot options
plt.colorbar()
plt.title("Relative Error", fontsize=20)
plt.ylabel("Wealth", fontsize=15, labelpad=10)
plt.xlabel("Time", fontsize=15, labelpad=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

if saveFigure:
    plt.savefig(figureName + '_valueFunction_relErr.png')
    
#%% Error heatmaps - optimal control
# vector of t and X values for plotting
X_plot = np.linspace(X_low, X_high, n_plot)
t_plot = np.linspace(t_low, T, n_plot)

# compute optimal control for each (t,X) pair
optimal_control_mesh = np.zeros([n_plot, n_plot])

for i in range(n_plot):
    for j in range(n_plot):
    
        optimal_control_mesh[j,i] = optimal_control_analytical_solution(t_plot[i], X_plot[j])
    
# compute model-implied optimal control for each (t,X) pair
t_mesh, X_mesh = np.meshgrid(t_plot, X_plot)

t_plot = np.reshape(t_mesh, [n_plot**2,1])
X_plot = np.reshape(X_mesh, [n_plot**2,1])

fitted_optimal_control = sess.run([numerical_optimal_control], feed_dict= {t_interior_tnsr:t_plot, X_interior_tnsr:X_plot})
fitted_optimal_control_mesh = np.reshape(fitted_optimal_control, [n_plot, n_plot])

# PLOT ABSOLUTE ERROR
plt.figure()
plt.figure(figsize = (8,6))

plt.pcolormesh(t_mesh, X_mesh, np.abs(optimal_control_mesh - fitted_optimal_control_mesh), cmap = "rainbow")

# plot options
plt.colorbar()
plt.title("Absolute Error", fontsize=20)
plt.ylabel("Wealth", fontsize=15, labelpad=10)
plt.xlabel("Time", fontsize=15, labelpad=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

if saveFigure:
    plt.savefig(figureName + '_optimalControl_absErr.png')

# PLOT RELATIVE ERROR
plt.figure()
plt.figure(figsize = (8,6))

plt.pcolormesh(t_mesh, X_mesh, np.abs(1 - np.divide(fitted_optimal_control_mesh, optimal_control_mesh)), cmap = "rainbow")

# plot options
plt.colorbar()
plt.title("Relative Error", fontsize=20)
plt.ylabel("Wealth", fontsize=15, labelpad=10)
plt.xlabel("Time", fontsize=15, labelpad=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

if saveFigure:
    plt.savefig(figureName + '_optimalControl_relErr.png')