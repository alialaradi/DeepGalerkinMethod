# SCRIPT FOR SOLVING THE FOKKER-PLANCK EQUATION FOR ORNSTEIN-UHLENBECK PROCESS 
# WITH RANDOM (NORMALLY DISTRIBUTED) STARTING VALUE (see p.54)

#%% import needed packages

import DGM
import tensorflow as tf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#%% Parameters 

# OU process parameters 
kappa = 0.5  # mean reversion rate
theta = 0.0  # mean reversion level
sigma = 2    # volatility

# mean and standard deviation for (normally distributed) process starting value
alpha = 0.0
beta = 1

# terminal time 
T = 1.0

# bounds of sampling region for space dimension, i.e. sampling will be done on
# [multiplier*Xlow, multiplier*Xhigh]
Xlow = -4.0
Xhigh = 4.0
x_multiplier = 2.0
t_multiplier = 1.5

# neural network parameters
num_layers = 3
nodes_per_layer = 50
learning_rate = 0.001

# Training parameters
sampling_stages = 500    # number of times to resample new time-space domain points
steps_per_sample = 10    # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_t = 5
nSim_x_interior = 50
nSim_x_initial = 50

# Save options
saveOutput = False
saveName   = 'FokkerPlanck'
saveFigure = False
figureName = 'fokkerPlanck_density.png'

#%% OU Simulation function

def simulateOU_GaussianStart(alpha, beta, theta, kappa, sigma, nSim, T):
    ''' Simulate end point of Ornstein-Uhlenbeck process with normally 
        distributed random starting value.
    
    Args:
        alpha: mean of random starting value
        beta:  standard deviation of random starting value
        theta: mean reversion level
        kappa: mean reversion rate
        sigma: volatility 
        nSim:  number of simulations
        T:     terminal time        
    '''  
        
    # simulate initial point based on normal distribution
    X0 = np.random.normal(loc = alpha, scale = beta, size = nSim)
    
    # mean and variance of OU endpoint
    m = theta + (X0 - theta) * np.exp(-kappa * T)
    v = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2*kappa*T)))
    
    # simulate endpoint
    Xt = np.random.normal(m,v)    
    
    return Xt

#%% Sampling function - randomly sample time-space pairs 

def sampler(nSim_t, nSim_x_interior, nSim_x_initial):
    ''' Sample time-space points from the function's domain; points are sampled
        uniformly on the interior of the domain, at the initial/terminal time points
        and along the spatial boundary at different time points. 
    
    Args:
        nSim_t:          number of (interior) time points to sample
        nSim_x_interior: number of space points in the interior of the function's domain to sample 
        nSim_x_initial:  number of space points at initial time to sample (initial condition)
    ''' 
    
    # Sampler #1: domain interior
    t = np.random.uniform(low=0, high=T*t_multiplier, size=[nSim_t, 1])
    x_interior = np.random.uniform(low=Xlow*x_multiplier, high=Xhigh*x_multiplier, size=[nSim_x_interior, 1])
    
    # Sampler #2: spatial boundary
        # no spatial boundary condition for this problem 
    
    # Sampler #3: initial/terminal condition
    x_initial = np.random.uniform(low=Xlow*1.5, high=Xhigh*1.5, size = [nSim_x_initial, 1])
    
    return t, x_interior, x_initial

#%% Loss function for Fokker-Planck equation

def loss(model, t, x_interior, x_initial, nSim_t, alpha, beta):
    ''' Compute total loss for training.
        NOTE: the loss is based on the PDE satisfied by the negative-exponential
              of the density and NOT the density itself, i.e. the u(t,x) in 
              p(t,x) = exp(-u(t,x)) / c(t)
              where p is the density and c is the normalization constant
    
    Args:
        model:      DGM model object
        t:          sampled (interior) time points
        x_interior: sampled space points in the interior of the function's domain
        x_initial:  sampled space points at initial time
        nSim_t:     number of (interior) time points sampled (size of t)
        alpha:      mean of normal distribution for process starting value
        beta:       standard deviation of normal distribution for process starting value
    ''' 
    
    # Loss term #1: PDE
    
    # initialize vector of losses
    losses_u = []
    
    # for each simulated interior time point
    for tIndex in range(nSim_t):
        
        # make vector of current time point to align with simulated interior space points   
        curr_t = t[tIndex]
        t_vector = curr_t * tf.ones_like(x_interior)
        
        # compute function value and derivatives at current sampled points
        u    = model(t_vector, x_interior)
        u_t  = tf.gradients(u, t_vector)[0]
        u_x  = tf.gradients(u, x_interior)[0]
        u_xx = tf.gradients(u_x, x_interior)[0]
        
        # psi function: normalized and exponentiated neural network
        # note: sums are used to approximate integrals (importance sampling)
        psi_denominator = tf.reduce_sum(tf.exp(-u))
        psi = tf.reduce_sum( u_t*tf.exp(-u) ) / psi_denominator

        # PDE differential operator
        # NOTE: EQUATION IN DOCUMENT IS INCORRECT - EQUATION HERE IS CORRECT
        diff_f = -u_t - kappa + kappa*(x_interior- theta)*u_x - 0.5*sigma**2*(-u_xx + u_x**2) + psi
        
        # compute L2-norm of differential operator and attach to vector of losses
        currLoss = tf.reduce_mean(tf.square(diff_f)) 
        losses_u.append(currLoss)
    
    # average losses across sample time points 
    L1 = tf.add_n(losses_u) / nSim_t
    
    # Loss term #2: boundary condition
        # no boundary condition for this problem
    
    # Loss term #3: initial condition
    
    # compute negative-exponential of neural network-implied pdf at t = 0
    # i.e. the u in p = e^[-u(t,x)] / c(t)
    fitted_pdf = model(0*tf.ones_like(x_initial), x_initial)
    
    # target pdf - normally distributed starting value
    # NOTE: we are only comparing the exponential terms 
    target_pdf  = 0.5*(x_initial - alpha)**2 / (beta**2)
    
    # average L2 error for initial distribution
    L3 = tf.reduce_mean(tf.square(fitted_pdf - target_pdf))

    return L1, L3    

#%% Set up network

# initialize DGM model (last input: space dimension = 1)
model = DGM.DGMNet(nodes_per_layer, num_layers, 1)

# tensor placeholders (_tnsr suffix indicates tensors)
# inputs (time, space domain interior, space domain at initial time)
t_tnsr = tf.placeholder(tf.float32, [None,1])
x_interior_tnsr = tf.placeholder(tf.float32, [None,1])
x_initial_tnsr = tf.placeholder(tf.float32, [None,1])

# loss 
L1_tnsr, L3_tnsr = loss(model, t_tnsr, x_interior_tnsr, x_initial_tnsr, nSim_t, alpha, beta)
loss_tnsr = L1_tnsr + L3_tnsr

# UNNORMALIZED density 
u = model(t_tnsr, x_interior_tnsr)
p_unnorm = tf.exp(-u)

# set optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)

# initialize variables
init_op = tf.global_variables_initializer()

# open session
sess = tf.Session()
sess.run(init_op)

#%% Train network
# for each sampling stage
for i in range(sampling_stages):
    
    # sample uniformly from the required regions
    t, x_interior, x_initial = sampler(nSim_t, nSim_x_interior, nSim_x_initial)
    
    # for a given sample, take the required number of SGD steps
    for j in range(steps_per_sample):
        loss,L1,L3,_ = sess.run([loss_tnsr, L1_tnsr, L3_tnsr, optimizer],
                                feed_dict = {t_tnsr:t, x_interior_tnsr:x_interior, x_initial_tnsr:x_initial})
        
    print(loss, L1, L3, i)

# save outout
if saveOutput:
    saver = tf.train.Saver()
    saver.save(sess, './SavedNets/' + saveName)

#%% Plot results

# LaTeX rendering for text in plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# figure options
plt.figure()
plt.figure(figsize = (12,10))

# time values at which to examine density
densityTimes = [0, 0.33*T, 0.66*T, T]

# vector of x values for plotting 
x_plot = np.linspace(Xlow*x_multiplier, Xhigh*x_multiplier, 1000)

for i, curr_t in enumerate(densityTimes):
    
    # specify subplot
    plt.subplot(2,2,i+1)
    
    # simulate process at current t 
    sim_x = simulateOU_GaussianStart(alpha, beta, theta, kappa, sigma, 10000, curr_t)
    
    # compute normalized density at all x values to plot and current t value
    t_plot = curr_t * np.ones_like(x_plot.reshape(-1,1))
    unnorm_dens = sess.run([p_unnorm], feed_dict= {t_tnsr:t_plot, x_interior_tnsr:x_plot.reshape(-1,1)})
    density = unnorm_dens[0] / sp.integrate.simps(unnorm_dens[0].reshape(x_plot.shape), x_plot)
    
    # plot histogram of simulated process values and overlay estimated density
    plt.hist(sim_x, bins=40, normed=True, color = 'b')
    plt.plot(x_plot, density, 'r', linewidth=2.5)
    
    # subplot options
    plt.ylim(ymin=0.0, ymax=0.45)
    plt.xlabel(r"$x$", fontsize=15, labelpad=10)
    plt.ylabel(r"$p(t,x)$", fontsize=15, labelpad=20)
    plt.title(r"\boldmath{$t$}\textbf{ = %.2f}"%(curr_t), fontsize=18, y=1.03)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
# adjust space between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.4)

if saveFigure:
    plt.savefig(figureName)