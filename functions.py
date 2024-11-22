import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve

def E (p, f):
    return np.sum(p*f)

def D_KL (p, q):
    return E(p, np.log(p/q))

def dot_prod (vec_a, vec_b):
    return vec_a[0]*vec_b[0] + vec_a[1]*vec_b[1]

def squared_modulus (vec):
    return dot_prod (vec, vec)

def direction_vec (theta):
    return np.array([np.cos(theta), np.sin(theta)]) 

def grad (f, dl):
    dim = np.shape(f)[0]
    x_tr_flat = (np.transpose(f)).flatten()
    grad_x_tr_flat = (np.roll(x_tr_flat, -1) - np.roll(x_tr_flat, 1))/(2*dl)
    grad_x = np.transpose(grad_x_tr_flat.reshape((dim, dim)))
    y_flat = f.flatten()
    grad_y_flat = (np.roll(y_flat, -1) - np.roll(y_flat, 1))/(2*dl)
    grad_y = grad_y_flat.reshape((dim, dim))
    return np.array([grad_x, grad_y])

def div (vec, dl):
    return grad(vec[0], dl)[0] + grad(vec[1], dl)[1]

def Laplacian (f, dl):
    dim = np.shape(f)[0]
    x_tr_flat = (np.transpose(f)).flatten()
    d2_x_tr_flat = (np.roll(x_tr_flat, -1) + np.roll(x_tr_flat, 1) -2*x_tr_flat) / np.power(dl, 2)
    d2_x = np.transpose(d2_x_tr_flat.reshape((dim, dim)))
    y_flat = f.flatten()
    d2_y_flat = (np.roll(y_flat, -1) + np.roll(y_flat, 1) -2*y_flat) / np.power(dl, 2)
    d2_y = d2_y_flat.reshape((dim, dim))
    return d2_x + d2_y    

def grad_ln (f, dl):
    return grad (np.log(f), dl)

def nu (p, dl, F, T):
    return F - T*grad_ln(p, dl)

def evolution_Matrices (F, T, dl, dt):    
    n_bins = np.shape(F[0])[0]
    n_bins_2 = n_bins**2
    ones_vec = np.ones(shape = n_bins_2)
    Fx, Fy = F
    Fy_flat = Fy.flatten()
    Fx_flat_tr = np.transpose(Fx).flatten()
    m_0 = (2/dt +2*T/np.power(dl, 2))*ones_vec
    m_up = Fx_flat_tr/(2*dl) -T/np.power(dl, 2)
    m_down = -Fx_flat_tr/(2*dl) -T/np.power(dl, 2)
    r_0 = (2/dt -2*T/np.power(dl, 2))*ones_vec
    r_up = -Fy_flat/(2*dl) +T/np.power(dl, 2)
    r_down = Fy_flat/(2*dl) +T/np.power(dl, 2)            
    M_hs_x = csc_matrix(diags([m_0, m_up[1:], m_down[:-1], m_up[0], m_down[-1]], [0, 1, -1, 1-n_bins_2, n_bins_2-1]))
    R_hs_x = csc_matrix(diags([r_0, r_up[1:], r_down[:-1], r_up[0], r_down[-1]], [0, 1, -1, 1-n_bins_2, n_bins_2-1]))
    M_hs_y = csc_matrix(diags([m_0, -r_up[1:], -r_down[:-1], -r_up[0], -r_down[-1]], [0, 1, -1, 1-n_bins_2, n_bins_2-1]))
    R_hs_y = csc_matrix(diags([r_0, -m_up[1:], -m_down[:-1], -m_up[0], -m_down[-1]], [0, 1, -1, 1-n_bins_2, n_bins_2-1]))    
    return R_hs_x, M_hs_x, R_hs_y, M_hs_y

def half_dynamics_x (p, R, M, n_bins):
    flat_p = p.flatten()
    vector = R @ flat_p
    vector_tr = (np.transpose(vector.reshape((n_bins, n_bins)))).flatten()
    this_M = M.copy()
    sol_tr_vec = spsolve(this_M, vector_tr)
    sol_tr = sol_tr_vec.reshape((n_bins, n_bins))
    return np.transpose(sol_tr)

def half_dynamics_y (p, R, M, n_bins):
    flat_p_tr = (np.transpose(p)).flatten()    
    matrix_tr = (R @ flat_p_tr).reshape((n_bins, n_bins)) 
    vector = (np.transpose(matrix_tr)).flatten()
    this_M = M.copy()
    return spsolve(this_M, vector).reshape((n_bins, n_bins))

def dynamics_step (p, Rx, Mx, Ry, My):
    n_bins = np.shape(p)[0]
    p1 = half_dynamics_x (p, Rx, Mx, n_bins)
    p1 /= np.sum(p1)
    p2 = half_dynamics_y (p1, Ry, My, n_bins)
    return p2 / np.sum(p2)

def smoothing_step (f, Rx, Mx, Ry, My):
    n_bins = np.shape(f)[0]
    f1 = half_dynamics_x (f, Rx, Mx, n_bins)
    f2 = half_dynamics_y (f1, Ry, My, n_bins)
    return f2

def smoothing (p, dl, coeff, precision):
        f = np.array([np.zeros_like(p), np.zeros_like(p)])
        Rx, Mx, Ry, My = evolution_Matrices (f, coeff, dl, 1/precision)
        smooth_step = lambda p: smoothing_step(p, Rx, Mx, Ry, My)
        q = p.copy()
        for t in range(precision):
            q = smooth_step (q)
        return q
           
def get_equilibrium_field (p_star, dl, T):
    return T*grad_ln (p_star, dl)

def alpha (p_star, phi, dl, T):
    grad_phi = grad(phi, dl)   
    grad_ln_p_star = grad_ln (p_star, dl)
    return dot_prod(grad_ln_p_star, grad_phi) + Laplacian(phi, dl)

def first_order_Eq (p_star, phi, dl, T):
    integral = E(p_star, squared_modulus(grad(phi, dl)))
    return -T*integral

def second_order_Eq (p_star, phi, dl, T):
    integral = E(p_star,np.power(alpha (p_star, phi, dl, T), 2))
    return 2*np.power(T,2)*integral
    
def analytic_xi (p_star, phi, dl, F, T):
    nu_star = nu (p_star, dl, F, T)
    grad_phi = grad(phi, dl)
    integral = E(p_star, alpha (p_star, phi, dl, T) * dot_prod(nu_star, grad_phi))
    return -2*T*integral
  
def bound_single_factor (p_star, phi, dl, T):
    this_alpha = alpha (p_star, phi, dl, T)
    integral = E(p_star, np.power(this_alpha, 2) * squared_modulus(grad(phi, dl)))
    return 1/(np.power(T,3)*integral)
  
def entropy_production (p_star, dl, F, T):
    nu_star = nu (p_star, dl, F, T)
    ep = (1/T)*E(p_star, squared_modulus(nu_star))
    return ep
  
def spatial_Fisher (p_star, dl, T):
    return np.power(T,2)*E(p_star, squared_modulus(grad_ln(p_star, dl)))

def inverted_components (vec):
    return np.array([vec[1], vec[0]])
    
def nonequilibrium_term_1 (p_star, dl, F, T):
    inverted_nu_star = inverted_components(nu (p_star, dl, F, T)) 
    local = dot_prod(inverted_nu_star, grad(p_star, dl))
    return np.power(np.sum(local),2)

def nonequilibrium_term_2 (p_star, dl, F, T):
    nu_star = nu (p_star, dl, F, T)
    local = grad(p_star, dl)[0]*nu_star[0]
    return 4*np.power(np.sum(local),2)

def zeta (p_star, dl): 
    return 0.5*squared_modulus(grad_ln(p_star, dl)) + Laplacian(np.log(p_star), dl)

def matrix_Vector_prod (M, vec):
    x_comp = M[0][0]*vec[0] + M[0][1]*vec[1]
    y_comp = M[1][0]*vec[0] + M[1][1]*vec[1]
    return np.array([x_comp, y_comp])

def Hessian (f, dl):
    gradient = grad(f, dl)
    row1 = grad(gradient[0], dl)
    row2 = grad(gradient[1], dl)
    return np.array([[row1[0], row1[1]], [row2[0], row2[1]]])

def analytical_displacement (p_star, dl, F, T, epsilon):
    H = Hessian (np.log(p_star), dl)
    H_grad_zeta = matrix_Vector_prod(H, grad(zeta (p_star, dl), dl))
    integral = E(p_star, dot_prod(H_grad_zeta, nu(p_star, dl, F, T)))
    return -T*np.power(epsilon,2)*integral
 
def displacement_bound_factor (p_star, dl, F, T, epsilon):
    H = Hessian (np.log(p_star), dl)
    H_grad_zeta = matrix_Vector_prod(H, grad(zeta (p_star, dl), dl))
    R = E(p_star, squared_modulus(H_grad_zeta))
    return np.power(T,2)*np.power(epsilon,4)*R
 
 