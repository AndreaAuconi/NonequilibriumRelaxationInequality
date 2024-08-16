#import os
#os.chdir('/home/andrea/Desktop/Revision_FokkerPlanck/Python/')
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime
from functions import *

# %% Settings

# model parameters
T = 5.

# spatial discretization parameters
l = 12
discretization = 120#160

# time discretization parameters
t_init = 35. #steady state estimation
sampling_interval_init = 1000
dt_init = 5e-5
t_experiment = 0.01
dt_experiment = 2e-6
sampling_interval_experiment = 500

# smoothing fields parameters
coeff_smoothing = 0.1
phi_smoothing = 0.1
precision_smoothing = 10000
domain_fraction = 0.85

# random fields parameters
nonlinearity = 1.
n_std = 5
conf_par = 0.18
conf_par2 = 3.5

# experiments parameters
perturbation_type = 'Displacement'#'Periodic'
epsilon = 0.005
k = 0.01

# replicas for computing averages
n_replicas = 200#1000
n_plotted = 8 # <= n_replicas

# multiprocessing parameter
fraction_cores = 0.85

n_cores = int(multiprocessing.cpu_count()*fraction_cores)
print('n_cores = ' + str(n_cores))

dl = l/discretization
class_Fields = Fields(l, dl, nonlinearity, n_std, coeff_smoothing, precision_smoothing)


# %% Force field

gamma = 3.
#F = class_Fields.getOUField(gamma)

#mu1, mu2 = 2/15, 2
mu1, mu2 = 2/l, 2
F = class_Fields.getQuadraticField(gamma, mu1, mu2)

#plt.imshow(Laplacian(F[1],dl))
#plt.imshow(np.transpose(F[1]))
   
# %% Get steady state distribution

print(datetime.datetime.now())

noneq_dyn_step = lambda p: dynamics_step(p, dl, F, T, dt_init)

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
 

F_eq = get_equilibrium_field (final_p, dl, T)


# Debug: check if numerical density oscillations are present
plt.clf(); plt.imshow(F_eq[1]); plt.savefig('F_eq.pdf')

eq_dyn_step = lambda p: dynamics_step (p, dl, F_eq, T, dt_init)

p = p0.copy()
p_t = []; t = 0; i = 0
while t < t_init:
    t += dt_init; i += 1
    p = eq_dyn_step (p)
    if i == sampling_interval_init:
        i = 0; print('Eq_' + str(100*t/t_init)[:5])
final_p_eq = p.copy()
  

print('distance: ' + str(D_KL(final_p,final_p_eq)))
final = 0.5*(final_p+final_p_eq)

ep = entropy_production (final, dl, F, T)
print('entropy production: ' + str(ep))

print(datetime.datetime.now())

# %% Relaxation experiments

noneq_dyn_step = lambda p: dynamics_step (p, dl, F, T, dt_experiment)
eq_dyn_step = lambda p: dynamics_step (p, dl, F_eq, T, dt_experiment)

def single_experiment ():
     
    if perturbation_type == 'Random':
        parameters = np.random.normal(0,1,10)
        phi = class_Fields.getRandomPerturbation(final, epsilon, parameters)
    elif perturbation_type == 'Displacement':
        theta = 2*np.pi*np.random.random()
        phi = class_Fields.getDisplacementPerturbation(final, epsilon, theta, phi_smoothing, domain_fraction)
    else:
        theta = 2*np.pi*np.random.random()
        phase = 2*np.pi*np.random.random()
        phi = class_Fields.getPeriodicPerturbation(final, epsilon, k, theta, phase)
        
    p_0 = final*(1+phi)
          
    Dneq = []  
    p = p_0.copy()
    p_eq = p_0.copy()
   
    t, i = 0, 0
    while t < t_experiment:
        t += dt_experiment; i += 1
        p = noneq_dyn_step (p)
        p_eq = eq_dyn_step (p_eq)
        if i == sampling_interval_experiment and t>0:
            i = 0
            Dneq.append(D_KL(p,final)-D_KL(p_eq,final))

    bound_factor = bound_single_factor (final, phi, dl, F, T)
    analytic_Xi = analytic_xi (final, phi, dl, F, T)    
    analytic_factor = bound_factor*np.power(analytic_Xi,2) / 4
    
    return Dneq, bound_factor, analytic_factor

    
replicas = Parallel(n_jobs=n_cores)(delayed(single_experiment)() for _ in range(n_replicas))

print(datetime.datetime.now())  
 
t_list = [i*dt_experiment*sampling_interval_experiment for i in range(1,len(replicas[0][0])+1)]
refined_t_list = [i*dt_experiment for i in range(1,int(1.02*len(replicas[0][0])*sampling_interval_experiment))]


# %% 

max_ratio = 0
for this in range(n_replicas):     
    ratio = replicas[this][2]/ep
    if ratio > max_ratio:
        max_ratio = ratio       
print('max ratio = ' + str(max_ratio))
 

plt.clf()

for this in range(n_plotted):

    plt.scatter(t_list, replicas[this][1]*np.power(replicas[this][0],2), s = 1)#, color = 'gray')
    analytic = np.array([replicas[this][2]*np.power(t,4) for t in refined_t_list])    
    plt.plot(refined_t_list, analytic, linestyle = 'dashed', linewidth = 0.2)#, color = 'gray')

plt.plot(refined_t_list, [ep*np.power(t,4) for t in refined_t_list], color = 'black', linewidth = 1.5)
  
plt.xlim(0, np.max(refined_t_list))
plt.ylim(0, 1.7*max_ratio*ep*np.power(t_experiment,4))
plt.xlabel(r'$t$', size = 14)
filename = 'epsilon_' + str(epsilon)[:6] + '_k_' + str(k)[:6] + '_expansion.pdf'
plt.savefig(filename)


# %% Randomized perturbations

if perturbation_type == 'Displacement':
    
    H = Hessian (np.log(final), dl)
    H_grad_zeta = matrix_Vector_prod(H, grad(zeta (final, dl), dl))
    local = final * dot_prod(H_grad_zeta, nu(final, dl, F, T))
    plt.clf()
    plt.imshow(local)
    plt.savefig('local_analytical.pdf')
    
    E_xi_disp = analytical_displacement (final, dl, F, T, epsilon)#, class_Fields.decay_numerical)
    
    mean_Dneq = np.mean([i[0] for i in replicas], axis = 0)
    
    plt.clf()
    plt.scatter(t_list, mean_Dneq, color = 'black')
    plt.plot(refined_t_list, [0.5*E_xi_disp*np.power(t,2) for t in refined_t_list], color = 'black', linewidth = 1)
    plt.savefig('analytical_prediction.pdf')
    
    std_Dneq = np.std([i[0] for i in replicas], axis = 0)
    
    plt.clf()
    plt.scatter(t_list, mean_Dneq, color = 'black')
    plt.scatter(t_list, std_Dneq, color = 'red') 
    filename = 'DEBUG.pdf'
    plt.savefig(filename)

      
elif perturbation_type == 'Periodic':
    
    mean_Dneq = np.mean([i[0] for i in replicas], axis = 0)  
    std_Dneq = np.std([i[0] for i in replicas], axis = 0)
    
    plt.clf()
    plt.scatter(t_list, mean_Dneq, color = 'black')
    plt.scatter(t_list, std_Dneq, color = 'red') 
    filename = 'DEBUG.pdf'
    plt.savefig(filename)

        
    Var_Dneq = np.var([i[0] for i in replicas], axis = 0)
    
    J = spatial_Fisher (final, dl, T)
    g = (8/3)/(T*J*np.power(epsilon*k,4))
    Var_xi = (3*np.power(T,2)*np.power(epsilon*k/2,4)) * (nonequilibrium_term_1 (final, dl, F, T) + nonequilibrium_term_2 (final, dl, F, T) )
        
    plt.clf()
    plt.plot(refined_t_list, [ep*np.power(t,4) for t in refined_t_list], color = 'black', linewidth = 1)
    plt.scatter(t_list, 4*g*Var_Dneq, color = 'black', s = 5.)     
    plt.plot(refined_t_list, [g*Var_xi*np.power(t,4) for t in refined_t_list], color = 'gray', linestyle = 'dashed', linewidth = 1)
     
    plt.legend([r'$\sigma^* t^4$', r'$4g\,\mathrm{Var}\left[ D-D^{eq} \right]$', r'$g t^4 \,\mathrm{Var}\left[\xi\right]$'], fontsize="13")
    
    plt.xlim(0, np.max(refined_t_list))
    plt.ylim(0, 1.7*np.max(4*g*Var_Dneq))
    plt.xlabel(r'$t$', size = 14)
    
    filename = 'epsilon_' + str(epsilon)[:6] + '_k_' + str(k)[:6] + '_ensemble_bound.pdf'
    plt.savefig(filename)
         
    
    plt.clf()
    fig, ax = plt.subplots()#figsize=(8, 8)
    
    ax.plot(refined_t_list, [ep*np.power(t,4) for t in refined_t_list], color = 'black', linewidth = 1)
    ax.plot(refined_t_list, [g*Var_xi*np.power(t,4) for t in refined_t_list], color = 'dimgray', linestyle = 'dashed', linewidth = 1.5)
    ax.scatter(t_list, 4*g*Var_Dneq, color = 'black', marker = 'o', s = 12.)     
    
    ax.legend([r'$\sigma^* t^4$', r'$g t^4 \,\mathrm{Var}\left[\xi\right]$', r'$4g\,\mathrm{Var}\left[ D-D^{eq} \right]$'], fontsize="13")
    
    plt.xlim(0, np.max(refined_t_list))
    plt.ylim(0, 1.5*np.max(4*g*Var_Dneq))
    plt.xlabel(r'$t$', size = 14)
    
    sub_ax = inset_axes(
        parent_axes=ax,
        width=1.85,#"30%",
        height=1.1,#"22%",
        bbox_to_anchor=(250,200),#'center left',
        borderpad=3,
    )
    
    
    sub_ax.hist(
        [i[2] for i in replicas],
        bins = 30,
        color='silver'
    )
    
    sub_ax.axvline(x = ep, color = 'black', linewidth = 1)
    sub_ax.axvline(x = g*Var_xi, color = 'dimgray', linestyle = 'dashed', linewidth = 1.5)
    
    
    sub_ax.set_title('Bounds of Eq. (13)', size = 10)
    
    filename = 'epsilon_' + str(epsilon)[:6] + '_k_' + str(k)[:6] + '_Fig3.pdf'
    plt.savefig(filename)
     
    
    plt.clf()
    plt.axvline(x = ep, color = 'black')
    plt.axvline(x = g*Var_xi, color = 'gray', linestyle = 'dashed')
    plt.hist([i[2] for i in replicas], bins = 30, color = 'gray')
    plt.legend(['ep', 'Eq. (20) bounds', 'Eq. (12) bounds'])
    plt.savefig('Hist_comparison.pdf')
       

# %%  plot currents

fraction = 0.36

N = int(2*discretization)
extent = [-l,l,l,-l]

this_n = int(0.5*(1-fraction)*N)

limits = [this_n, N-this_n]

#x = np.array([i for i in range(0,N)])[limits[0]:limits[1]]
#y = np.array([i for i in range(0,N)])[limits[0]:limits[1]]

x = np.array([(i-(N-1)/2)*dl for i in range(0,N)])[limits[0]:limits[1]]
y = np.array([(i-(N-1)/2)*dl for i in range(0,N)])[limits[0]:limits[1]]


currents = 40 * (final * nu (final, dl, F, T))[:,limits[0]:limits[1],limits[0]:limits[1]]
width = 8*np.sqrt(np.power(currents[0],2) + np.power(currents[1],2))

plt.clf()
plt.streamplot(x, y, np.transpose(currents[0]), np.transpose(currents[1]), density=1., linewidth=np.transpose(width), arrowsize= 0.8, color = 'white') 
plt.imshow(np.transpose(final), cmap='Greys', extent=extent)
plt.xlabel('$x$', size = 12)
plt.ylabel('$y$', size = 12,rotation = 0)

plt.xlim(x[0],x[-1])
plt.ylim(y[0],y[-1])

plt.savefig('currents.pdf') 


plt.clf()
plt.streamplot(x, y, np.transpose(currents[0]), np.transpose(currents[1]), density=1., linewidth=np.transpose(width), arrowsize= 0.8, color = 'black') 
plt.imshow(np.transpose(final), cmap='Greys_r', extent=extent)
plt.xlabel('$x$', size = 12)
plt.ylabel('$y$', size = 12,rotation = 0)

plt.xlim(x[0],x[-1])
plt.ylim(y[0],y[-1])

plt.savefig('currents_r.pdf') 
