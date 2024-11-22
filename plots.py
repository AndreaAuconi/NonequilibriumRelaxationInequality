import numpy as np
from functions import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def print_params(parameters_x, parameters_y):
    with open('parameters.txt', 'w') as f:
        for j in parameters_x:
            f.write(str(j))
            f.write('\n')
        for j in parameters_y:
            f.write(str(j))
            f.write('\n')        

def Debug_integrals_ensemble(final, dl, F, T):
    
    plt.clf()
    plt.imshow(final)
    plt.savefig('Density_Neq.pdf')
    
    local = final *  squared_modulus(grad_ln(final, dl))
    plt.clf()
    plt.imshow(local)
    plt.savefig('Fisher_information_local.pdf')
    
    inverted_nu_star = inverted_components(nu (final, dl, F, T)) 
    local = dot_prod(inverted_nu_star, grad(final, dl))
    plt.clf()
    plt.imshow(local)
    plt.savefig('Periodic_local_analytical_1.pdf')
    
    nu_star = nu (final, dl, F, T)
    local = grad(final, dl)[0]*nu_star[0]
    plt.clf()
    plt.imshow(local)
    plt.savefig('Periodic_local_analytical_2.pdf')
    
    H = Hessian (np.log(final), dl)
    H_grad_zeta = matrix_Vector_prod(H, grad(zeta (final, dl), dl))
    local = final * dot_prod(H_grad_zeta, nu(final, dl, F, T))
    plt.clf()
    plt.imshow(local)
    plt.savefig('Displacement_local_analytical.pdf')
        
    local = final * squared_modulus(H_grad_zeta)
    plt.clf()
    plt.imshow(local)
    plt.savefig('Displacement_local_bound.pdf')


def Debug_integrals_single_Periodic(n_tests, final, class_Fields, dl, F, T, epsilon, k):
    
    for i in range(n_tests):
        
        theta = 2*np.pi*np.random.random()
        phase = 2*np.pi*np.random.random()
        phi = class_Fields.getPeriodicPerturbation(final, epsilon, k, theta, phase)

        print('max phi = ', np.max(phi))
    
        this_alpha = alpha (final, phi, dl, T)
        integrand = final * np.power(this_alpha, 2) * squared_modulus(grad(phi, dl))
        plt.clf()
        plt.imshow(integrand)
        plt.savefig('bound_integrand_' + str(i))
        
        nu_star = nu (final, dl, F, T)
        grad_phi = grad(phi, dl)
        integrand_Xi = final * alpha (final, phi, dl, T) * dot_prod(nu_star, grad_phi)    
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


def Debug_integrals_single_Displacement(n_tests, final, class_Fields, dl, F, T, epsilon):
    
    for i in range(n_tests):
    
        theta = 2*np.pi*np.random.random()
        phi = class_Fields.getDisplacementPerturbation(final, epsilon, theta, dl)

        print('max phi = ', np.max(phi))
    
        this_alpha = alpha (final, phi, dl, T)
        integrand = final * np.power(this_alpha, 2) * squared_modulus(grad(phi, dl))
        plt.clf()
        plt.imshow(integrand)
        plt.savefig('bound_integrand_' + str(i))
        
        nu_star = nu (final, dl, F, T)
        grad_phi = grad(phi, dl)
        integrand_Xi = final * alpha (final, phi, dl, T) * dot_prod(nu_star, grad_phi)    
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


def plot_single_trajectories (n_tests, replicas, t_experiment, t_list, refined_t_list, ep):

    n_replicas = len(replicas)
    max_ratio = 0
    for this in range(n_replicas):     
        ratio = replicas[this][2]/ep
        if ratio > max_ratio:
            max_ratio = ratio       
    print('max ratio = ' + str(max_ratio))
    
    plt.clf()
    
    for this in range(n_tests):
    
        plt.scatter(t_list, replicas[this][1]*np.power(replicas[this][0],2), s = 1)#, color = 'gray')
        analytic = np.array([replicas[this][2]*np.power(t,4) for t in refined_t_list])    
        plt.plot(refined_t_list, analytic, linestyle = 'dashed', linewidth = 0.2)#, color = 'gray')
    
    plt.plot(refined_t_list, [ep*np.power(t,4) for t in refined_t_list], color = 'black', linewidth = 1.5)
      
    plt.xlim(0, np.max(refined_t_list))
    plt.ylim(0, 1.7*max_ratio*ep*np.power(t_experiment,4))
    plt.xlabel(r'$t$', size = 14)
    plt.savefig('Debug_expansion.pdf')


def ensemble_plots_Periodic (replicas, t_list, refined_t_list, ep, final, dl, F, T, epsilon, k):    
   
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
           


def ensemble_plots_Displacement (replicas, t_list, refined_t_list, ep, final, dl, F, T, epsilon):    

    E_xi_disp = analytical_displacement (final, dl, F, T, epsilon)
    
    mean_Dneq = np.mean([i[0] for i in replicas], axis = 0)
    
    R_T2_eps4 = displacement_bound_factor (final, dl, F, T, epsilon)
    
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
    plt.savefig('Fig_displacement_noInset.pdf')

        
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
       
    
    filename = 'epsilon_' + str(epsilon)[:6] + '_FigPaperDisplacement.pdf'
    plt.savefig(filename)
     
    print('avg-based bound ratio = ' + str(np.power(E_xi_disp,2) / (R_T2_eps4*ep)))
    

def plot_currents(final, dl, F, T, discretization, l):
    
    fraction = 0.4
    
    N = int(2*discretization)
    extent = [-l,l,l,-l]
    
    this_n = int(0.5*(1-fraction)*N)
    
    limits = [this_n, N-this_n]
    
    x = np.array([(i-(N-1)/2)*dl for i in range(0,N)])[limits[0]:limits[1]]
    y = np.array([(i-(N-1)/2)*dl for i in range(0,N)])[limits[0]:limits[1]]
    
    
    currents = 40 * (final * nu (final, dl, F, T))[:,limits[0]:limits[1],limits[0]:limits[1]]
    width = 20*np.sqrt(np.power(currents[0],2) + np.power(currents[1],2))
    
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
    
