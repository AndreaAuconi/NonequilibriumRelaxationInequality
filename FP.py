import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime
from functions import *

# %% Settings

# which plot
Fig = 'R'

# model parameters
T = 5.
approx_eq = False

# spatial discretization parameters
l = 12.
discretization = 220

#steady-state estimation
if Fig in ['2', 'A1', 'A4']:
    t_init = 35.
else:
    t_init = 50.
    
# time discretization parameters    
sampling_interval_init = 100
dt_init = 2e-3
t_experiment = 1e-2
dt_experiment = 5e-6
sampling_interval_experiment = 100

# smoothing fields parameters
coeff_smoothing = 0.1
precision_smoothing = 20000
exclude_border = 0.02

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
n_replicas = 300#1000
n_plotted = 8 # <= n_replicas

# multiprocessing parameter
fraction_cores = 0.4

n_cores = int(multiprocessing.cpu_count()*fraction_cores)
print('n_cores = ' + str(n_cores))

dl = l/discretization
class_Fields = Fields(l, dl, nonlinearity, n_std, coeff_smoothing, precision_smoothing, exclude_border)
factor_border = class_Fields.factor_border

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
    conf_par1 = 0.2#0.25
    conf_par2 = 1.2
    force = 50.#35.
    parameters_x, parameters_y = np.random.normal(0,1,10), np.random.normal(0,1,10)
    parameters_x[0], parameters_y[0] = 0., 0.
    F = class_Fields.getRandomField(force, parameters_x, parameters_y, conf_par1, conf_par2)          
    with open('parameters.txt', 'w') as f:
        for j in parameters_x:
            f.write(str(j))
            f.write('\n')
        for j in parameters_y:
            f.write(str(j))
            f.write('\n')

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

F_eq = symm_get_equilibrium_field (final_p, dl, T)

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

print(datetime.datetime.now())

plt.clf()
plt.imshow(final_p)
plt.savefig('Density_Neq.pdf')


# %% Debug integrals

for i in range(5):
    
    if perturbation_type == 'Random':
        parameters = np.random.normal(0,1,10)
        phi = class_Fields.getRandomPerturbation(final, epsilon, parameters)
    elif perturbation_type == 'Displacement':
        theta = 2*np.pi*np.random.random()
        phi = class_Fields.getDisplacementPerturbation(final, epsilon, theta)
    else:
        theta = 2*np.pi*np.random.random()
        phase = 2*np.pi*np.random.random()
        phi = class_Fields.getPeriodicPerturbation(final, epsilon, k, theta, phase)

    print('max phi = ', np.max(phi))

    this_alpha = alpha (final, phi, dl, T)
    integrand = factor_border * final * np.power(this_alpha, 2) * squared_modulus(grad(phi, dl))
    plt.clf()
    plt.imshow(integrand)
    plt.savefig('bound_integrand_' + str(i))
    
    nu_star = nu (final, dl, F, T)
    grad_phi = grad(phi, dl)
    integrand_Xi = factor_border * final * alpha (final, phi, dl, T) * dot_prod(nu_star, grad_phi)    
    plt.clf()
    plt.imshow(integrand_Xi)
    plt.savefig('Xi_integrand_' + str(i))
    
    grad_phi = grad(phi, dl)
    integrand = squared_modulus(final * phi * grad_phi)   
    plt.clf()
    plt.imshow(integrand)
    plt.savefig('Decay_1_' + str(i))
    
    nu_star = nu (final, dl, F, T)
    integrand = squared_modulus(final * np.power(phi, 2) * nu_star)   
    plt.clf()
    plt.imshow(integrand)
    plt.savefig('Decay_2_' + str(i))
    
    grad_phi = grad(phi, dl)
    integrand = squared_modulus(final * Laplacian(phi, dl) * grad_phi)   
    plt.clf()
    plt.imshow(integrand)
    plt.savefig('Decay_3_' + str(i))
    
    grad_phi = grad(phi, dl)
    integrand = squared_modulus(final * dot_prod(grad_phi, nu_star) * grad_phi)   
    plt.clf()
    plt.imshow(integrand)
    plt.savefig('Decay_4_' + str(i))
    
    nu_star = nu (final, dl, F, T)
    integrand = squared_modulus(final * np.log(final) * nu_star)   
    plt.clf()
    plt.imshow(integrand)
    plt.savefig('Decay_5_' + str(i))

    plt.clf()
    plt.imshow(phi)
    plt.savefig('phi_example_' + str(i))

    plt.clf()
    plt.imshow(final*phi)
    plt.savefig('d_phi_example_' + str(i))
    
    plt.clf()
    plt.imshow(final*(1+phi)*np.log(1+phi))
    plt.savefig('D0_' + str(i))

# %% Relaxation experiments

print('Experiments ...')
print(datetime.datetime.now())


Rx, Mx, Ry, My = evolution_Matrices (F, T, dl, dt_experiment)
Eq_Rx, Eq_Mx, Eq_Ry, Eq_My = evolution_Matrices (F_eq, T, dl, dt_experiment)

noneq_dyn_step = lambda p: dynamics_step(p, Rx, Mx, Ry, My)
eq_dyn_step = lambda p: dynamics_step (p, Eq_Rx, Eq_Mx, Eq_Ry, Eq_My)    

def single_experiment ():
     
    if perturbation_type == 'Random':
        parameters = np.random.normal(0,1,10)
        phi = class_Fields.getRandomPerturbation(final, epsilon, parameters)
    elif perturbation_type == 'Displacement':
        theta = 2*np.pi*np.random.random()
        phi = class_Fields.getDisplacementPerturbation(final, epsilon, theta)
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

    bound_factor = bound_single_factor (final, phi, dl, T, factor_border)
    analytic_Xi = analytic_xi (final, phi, dl, F, T, factor_border)    
    analytic_factor = bound_factor*np.power(analytic_Xi,2) / 4
    
    return Dneq, bound_factor, analytic_factor

    
replicas = Parallel(n_jobs=n_cores)(delayed(single_experiment)() for _ in range(n_replicas))

print(datetime.datetime.now())  
 
t_list = [i*dt_experiment*sampling_interval_experiment for i in range(1,len(replicas[0][0])+1)]
refined_t_list = [i*dt_experiment for i in range(1,int(1.04*len(replicas[0][0])*sampling_interval_experiment))]

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
    local = factor_border * final * dot_prod(H_grad_zeta, nu(final, dl, F, T))
    plt.clf()
    plt.imshow(local)
    plt.savefig('local_analytical.pdf')
        
    local = factor_border * final * squared_modulus(H_grad_zeta)
    plt.clf()
    plt.imshow(local)
    plt.savefig('local_displacement_bound.pdf')

    
    E_xi_disp = analytical_displacement (final, dl, F, T, epsilon, factor_border)#, class_Fields.decay_numerical)
    
    mean_Dneq = np.mean([i[0] for i in replicas], axis = 0)
    
    R_T2_eps4 = displacement_bound_factor (final, dl, F, T, epsilon, factor_border)
    
    plt.clf()
    plt.scatter(t_list, mean_Dneq, color = 'black')
    plt.plot(refined_t_list, [0.5*E_xi_disp*np.power(t,2) for t in refined_t_list], color = 'black', linewidth = 1)
    plt.savefig('DEBUG_analytical_prediction.pdf')
    
    plt.clf()
    plt.plot(refined_t_list, [ep*np.power(t,4) for t in refined_t_list], color = 'black', linewidth = 1)
    plt.plot(refined_t_list, [np.power(E_xi_disp,2)*np.power(t,4) / R_T2_eps4 for t in refined_t_list], color = 'dimgray', linestyle = 'dashed', linewidth = 1.5)
    plt.scatter(t_list, 4*np.power(mean_Dneq, 2) / R_T2_eps4, color = 'black', marker = 'o', s = 12.)
    plt.xlim(0, np.max(refined_t_list))
    plt.ylim(0, 1.5*np.max(4*np.power(mean_Dneq, 2) / R_T2_eps4))
    plt.legend([r'$\sigma^* t^4$', r'$h t^4 \,\left(\mathrm{E}\left[\xi\right]\right)^2$', r'$4h\,\left(\mathrm{E}\left[ D-D^{eq} \right]\right)^2$'], fontsize="13")
    plt.savefig('Fig_displacement.pdf')

        
    n_bins = 100
    hist, bins, _ = plt.hist([i[2] for i in replicas], bins=n_bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    
    plt.clf()
    fig, ax = plt.subplots()#figsize=(8, 8)
    
    ax.plot(refined_t_list, [ep*np.power(t,4) for t in refined_t_list], color = 'black', linewidth = 1)
    ax.plot(refined_t_list, [np.power(E_xi_disp,2)*np.power(t,4) / R_T2_eps4 for t in refined_t_list], color = 'dimgray', linestyle = 'dashed', linewidth = 1.5)
    ax.scatter(t_list, 4*np.power(mean_Dneq, 2) / R_T2_eps4, color = 'black', marker = 'o', s = 12.)     
    
    ax.legend([r'$\sigma^* t^4$', r'$h t^4 \,\left(\mathrm{E}\left[\xi\right]\right)^2$', r'$4h\,\left(\mathrm{E}\left[ D-D^{eq} \right]\right)^2$'], fontsize="13")

    plt.xlim(0, np.max(refined_t_list))
    plt.ylim(0, 1.5*np.max(4*np.power(mean_Dneq, 2) / R_T2_eps4))
    plt.xlabel(r'$t$', size = 14)
    
    sub_ax = inset_axes(
        parent_axes=ax,
        width=1.85,#"30%",
        height=1.15,#"22%",
        bbox_to_anchor=(250,200),#'center left',
        borderpad=3,
    )
    

    sub_ax.hist(
        [i[2] for i in replicas],
        bins = logbins,
        color='silver'
    )
    
    sub_ax.axvline(x = ep, color = 'black', linewidth = 1)
    sub_ax.axvline(x = np.power(E_xi_disp,2) / R_T2_eps4, color = 'dimgray', linestyle = 'dashed', linewidth = 1.5)

    sub_ax.set_xlim(ep*0.002,ep*1.5)
    sub_ax.set_xscale('log')       
    sub_ax.set_title('Bounds of Eqs. (13), (21)', size = 10)
    
    
    
    filename = 'epsilon_' + str(epsilon)[:6] + '_k_' + str(k)[:6] + '_FigPaperDisplacement.pdf'
    plt.savefig(filename)
     
    print('avg-based bound ratio = ' + str(np.power(E_xi_disp,2) / (R_T2_eps4*ep)))


      
elif perturbation_type == 'Periodic':
    
    inverted_nu_star = inverted_components(nu (final, dl, F, T)) 
    local = dot_prod(inverted_nu_star, grad(final, dl))
    plt.clf()
    plt.imshow(local)
    plt.savefig('local_analytical_1.pdf')

    nu_star = nu (final, dl, F, T)
    local = grad(final, dl)[0]*nu_star[0]
    plt.clf()
    plt.imshow(local)
    plt.savefig('local_analytical_2.pdf')

    
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
    plt.scatter(t_list, 4*g*Var_Dneq, color = 'black', s = 6.)     
    plt.plot(refined_t_list, [g*Var_xi*np.power(t,4) for t in refined_t_list], color = 'gray', linestyle = 'dashed', linewidth = 1)
     
    plt.legend([r'$\sigma^* t^4$', r'$4g\,\mathrm{Var}\left[ D-D^{eq} \right]$', r'$g t^4 \,\mathrm{Var}\left[\xi\right]$'], fontsize="13")
    
    plt.xlim(0, np.max(refined_t_list))
    plt.ylim(0, 1.7*np.max(4*g*Var_Dneq))
    plt.xlabel(r'$t$', size = 14)
    
    filename = 'epsilon_' + str(epsilon)[:6] + '_k_' + str(k)[:6] + '_ensemble_bound.pdf'
    plt.savefig(filename)
      
    
    n_bins = 100
    hist, bins, _ = plt.hist([i[2] for i in replicas], bins=n_bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    
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
        height=1.15,#"22%",
        bbox_to_anchor=(250,200),#'center left',
        borderpad=3,
    )
    

    sub_ax.hist(
        [i[2] for i in replicas],
        bins = logbins,
        color='silver'
    )
    
    sub_ax.axvline(x = ep, color = 'black', linewidth = 1)
    sub_ax.axvline(x = g*Var_xi, color = 'dimgray', linestyle = 'dashed', linewidth = 1.5)

    sub_ax.set_xlim(ep*0.002,ep*1.5)
    sub_ax.set_xscale('log')       
    sub_ax.set_title('Bounds of Eqs. (13), (19)', size = 10)
    
    
    
    filename = 'epsilon_' + str(epsilon)[:6] + '_k_' + str(k)[:6] + '_FigPaper.pdf'
    plt.savefig(filename)
     
    print('avg-based bound ratio = ' + str(g*Var_xi/ep))
           

# %%  plot currents

fraction = 0.4

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

plt.clf()
plt.streamplot(x, y, np.transpose(currents[0]), np.transpose(currents[1]), density=1., linewidth=np.transpose(width), arrowsize= 0.8, color = 'white') 
plt.imshow(np.log(np.transpose(final)), cmap='Greys', extent=extent)
plt.xlabel('$x$', size = 12)
plt.ylabel('$y$', size = 12,rotation = 0)

plt.xlim(x[0],x[-1])
plt.ylim(y[0],y[-1])

plt.savefig('Log_currents.pdf') 


plt.clf()
plt.streamplot(x, y, np.transpose(currents[0]), np.transpose(currents[1]), density=1., linewidth=np.transpose(width), arrowsize= 0.8, color = 'black') 
plt.imshow(np.log(np.transpose(final)), cmap='Greys_r', extent=extent)
plt.xlabel('$x$', size = 12)
plt.ylabel('$y$', size = 12,rotation = 0)

plt.xlim(x[0],x[-1])
plt.ylim(y[0],y[-1])

plt.savefig('Log_currents_r.pdf')


this = [i[0][-1] for i in replicas]
np.save('replicas_last.npy', this)
plt.clf()
plt.hist(this, bins=30)
plt.savefig('hist_Debug.pdf')

