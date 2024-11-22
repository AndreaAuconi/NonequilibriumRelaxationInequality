import os
#os.chdir('/home/andrea/Desktop/StochThermo_InfoGeom/Revision_FokkerPlanck/Method/')
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import datetime
from functions import *
from plots import *
from Fields import Fields
from tqdm import tqdm

# %% Settings

# which plot
Fig = 'A1'

# model parameters
T = 5.
approx_eq = False

# spatial discretization parameters
l = 12.
discretization = 400

#steady-state estimation
if Fig in ['2', 'A1', 'A4']:
    t_init = 35.
else:
    t_init = 50.
    
# time discretization parameters    
sampling_interval_init = 100
dt_init = 2e-3
t_experiment = 1e-2
dt_experiment = 2e-5
sampling_interval_experiment = 25

# smoothing fields parameters
border_exponent = 4
precision_smoothing = 500
coeff_smoothing = 0.05

# random fields parameters
nonlinearity = 1.2
n_std = 5

# experiments parameters
k = 0.02 # 'Periodic' case
if Fig == 'A4':
    perturbation_type = 'Displacement'
    epsilon = 0.1   
else:
    perturbation_type = 'Periodic'
    epsilon = 1.
    
# replicas for computing averages
n_replicas = 1000

# multiprocessing parameter
fraction_cores = 0.45

n_cores = int(multiprocessing.cpu_count()*fraction_cores)
print('n_cores = ' + str(n_cores))

dl = l/discretization
class_Fields = Fields(l, dl, nonlinearity, n_std, coeff_smoothing, precision_smoothing, border_exponent)

# Check if steady-state is reached by F_eq
Check_F_eq = True
load_steady_state = False

# %% Force field

gamma = 3.

if Fig == '2':
    F = class_Fields.getOUField(gamma)
elif Fig == 'A1' or Fig == 'A4':
    mu1, mu2 = 2/15, 2.
    F = class_Fields.getQuadraticField(gamma, mu1, mu2)
else:
    force, conf_par1, conf_par2 = 50., 0.2, 1.2
    parameters_x, parameters_y = np.random.normal(0,1,10), np.random.normal(0,1,10)
    parameters_x[0], parameters_y[0] = 0., 0.
    F = class_Fields.getRandomField(force, parameters_x, parameters_y, conf_par1, conf_par2)          
    print_params(parameters_x, parameters_y)

# %% Get steady-state distribution

print(datetime.datetime.now())
print('steady-state estimation ...')

Rx, Mx, Ry, My = evolution_Matrices (F, T, dl, dt_init)

noneq_dyn_step = lambda p: dynamics_step(p, Rx, Mx, Ry, My)

if load_steady_state:
    final_p = np.load('p_star.npy')
else:
    parameters_init = np.random.normal(0,1,10)
    p0 = class_Fields.getRandomDistribution(parameters_init) 
    p = p0.copy()
    p_t = []; t = 0; i = 0
    while t < t_init:
        t += dt_init; i += 1
        p = noneq_dyn_step (p)
        if i == sampling_interval_init:
            i = 0; print('Noneq_' + str(100*t/t_init)[:5])
    final_p = p.copy()
    np.save('new_steady_state.npy', final_p)

plt.clf()
plt.imshow(np.log(final_p))
plt.savefig('Log_Neq_density.pdf')

# %%

F_eq = get_equilibrium_field (final_p, dl, T)

Eq_Rx, Eq_Mx, Eq_Ry, Eq_My = evolution_Matrices (F_eq, T, dl, dt_init)


if Check_F_eq:  
    print(datetime.datetime.now())
    print('Testing F_Eq ...') 
    eq_dyn_step = lambda p: dynamics_step (p, Eq_Rx, Eq_Mx, Eq_Ry, Eq_My) 
    p = p0.copy()
    p_t = []; t = 0; i = 0
    while t < t_init:
        t += dt_init; i += 1
        p = eq_dyn_step (p)
        if i == sampling_interval_init:
            i = 0; print('Eq_' + str(100*t/t_init)[:5])
    final_p_eq = p.copy()      
    print('distance: ' + str(D_KL(final_p, final_p_eq)))
    final = 0.5*(final_p+final_p_eq)
    plt.clf()
    plt.imshow(final_p_eq-final_p)
    plt.savefig('Diff_Density_Eq.pdf')
else:
    final = final_p
    
ep = entropy_production (final, dl, F, T)
print('entropy production: ' + str(ep))

plot_currents(final, dl, F, T, discretization, l)

print(datetime.datetime.now())

n_tests = 10
if not os.path.exists('Debug'):
    os.mkdir('Debug')
os.chdir('Debug')
Debug_integrals_ensemble(final, dl, F, T)
if perturbation_type == 'Displacement':
    Debug_integrals_single_Displacement(n_tests, final, class_Fields, dl, F, T, epsilon)
else:
    Debug_integrals_single_Periodic(n_tests, final, class_Fields, dl, F, T, epsilon, k)

os.chdir("..")

# %% Relaxation experiments

print('Experiments ...')
print(datetime.datetime.now())

Rx, Mx, Ry, My = evolution_Matrices (F, T, dl, dt_experiment)
Eq_Rx, Eq_Mx, Eq_Ry, Eq_My = evolution_Matrices (F_eq, T, dl, dt_experiment)

noneq_dyn_step = lambda p: dynamics_step(p, Rx, Mx, Ry, My)
eq_dyn_step = lambda p: dynamics_step (p, Eq_Rx, Eq_Mx, Eq_Ry, Eq_My)    

def single_experiment ():
     
    if perturbation_type == 'Displacement':
        theta = 2*np.pi*np.random.random()
        phi = class_Fields.getDisplacementPerturbation(final, epsilon, theta, dl)
    else:
        theta = 2*np.pi*np.random.random()
        phase = 2*np.pi*np.random.random()
        phi = class_Fields.getPeriodicPerturbation(final, epsilon, k, theta, phase)
            
    p_0 = final*(1+phi)
    
    d_t_Deq = first_order_Eq(final, phi, dl, T) 
    d2_t2_Deq = second_order_Eq(final, phi, dl, T)        
    D_init = D_KL(p_0, final)
          
    Dneq = []  
    p, p_eq = p_0.copy(), p_0.copy()
   
    t, i = 0, 0
    while t < t_experiment:
        t += dt_experiment; i += 1
        p = noneq_dyn_step (p)
        if not(approx_eq):
            p_eq = eq_dyn_step (p_eq)
        if i == sampling_interval_experiment:
            i = 0
            if not(approx_eq):
                D_KL_eq = D_KL(p_eq,final)
            else:
                D_KL_eq = D_init + d_t_Deq*t + 0.5*d2_t2_Deq*np.power(t,2)
            Dneq.append(D_KL(p,final)-D_KL_eq)

    bound_factor = bound_single_factor (final, phi, dl, T)
    analytic_Xi = analytic_xi (final, phi, dl, F, T)    
    analytic_factor = bound_factor*np.power(analytic_Xi,2) / 4
    
    return Dneq, bound_factor, analytic_factor

    
replicas = Parallel(n_jobs=n_cores)(delayed(single_experiment)() for _ in tqdm(range(n_replicas)))

print(datetime.datetime.now())  
 
t_list = [i*dt_experiment*sampling_interval_experiment for i in range(1,len(replicas[0][0])+1)]
refined_t_list = [i*dt_experiment for i in range(1,int(1.04*len(replicas[0][0])*sampling_interval_experiment))]

plot_single_trajectories (n_tests, replicas, t_experiment, t_list, refined_t_list, ep)

# %% Ensemble plots

if perturbation_type == 'Periodic':    
    ensemble_plots_Periodic (replicas, t_list, refined_t_list, ep, final, dl, F, T, epsilon, k)           
elif perturbation_type == 'Displacement':    
    ensemble_plots_Displacement (replicas, t_list, refined_t_list, ep, final, dl, F, T, epsilon)
      
