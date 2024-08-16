import numpy as np

def D_KL (p, q):
    return np.sum(p*(np.log(p/q)))

def dot_prod (vec_a, vec_b):
    return vec_a[0]*vec_b[0] + vec_a[1]*vec_b[1]

def squared_modulus (vec):
    return dot_prod (vec, vec)

def direction_vec (theta):
    return np.array([np.cos(theta), np.sin(theta)]) 
 
def grad (f, dl):# not symmetric
    grad_x = (f - np.roll(f, 1, 0))/dl
    grad_y = (f - np.roll(f, 1, 1))/dl
    return np.array([grad_x, grad_y])

def left_grad (f, dl):
    grad_x = (np.roll(f, -1, 0) - f)/dl
    grad_y = (np.roll(f, -1, 1) - f)/dl
    return np.array([grad_x, grad_y])

def div (vec, dl):
    return grad(vec[0], dl)[0] + grad(vec[1], dl)[1]

def left_div (vec, dl):
    return left_grad(vec[0], dl)[0] + left_grad(vec[1], dl)[1]

def Laplacian (f, dl):# symmetric # avoids skip
    fxx = np.roll(f, 1, 0) + np.roll(f, -1, 0) -2*f
    fyy = np.roll(f, 1, 1) + np.roll(f, -1, 1) -2*f 
    return (fxx + fyy) / np.power(dl,2)

def grad_ln (f, dl):
    return grad (np.log(f), dl)

def nu (p, dl, F, T):
    return F - T*grad_ln(p, dl)
   
def dynamics_step (p, dl, F, T, dt):
    f = p * nu (p, dl, F, T)
    dp = - dt * left_div(f, dl)
    p_tilde = p + dp
    return p_tilde/np.sum(p_tilde)

def smoothing (p, dl, coeff, precision):
    if coeff > 0:
        p_smooth = p.copy()
        factor = coeff/precision
        for _ in range(precision):
            p_smooth += factor * Laplacian (p_smooth, dl)
        return p_smooth
    else:
        return p
           
def get_equilibrium_field (p_star, dl, T):   
    return T*grad_ln (p_star, dl)

def alpha (p_star, phi, dl, F, T):
    grad_phi = grad(phi, dl)   
    grad_ln_p_star = grad_ln (p_star, dl)
    return dot_prod(grad_ln_p_star, grad_phi) + Laplacian(phi, dl)
    
def analytic_xi (p_star, phi, dl, F, T):
    nu_star = nu (p_star, dl, F, T)
    grad_phi = grad(phi, dl)
    local = p_star * alpha (p_star, phi, dl, F, T) * dot_prod(nu_star, grad_phi)
    return -2*T*np.sum(local)
  
def bound_single_factor (p_star, phi, dl, F, T):
    this_alpha = alpha (p_star, phi, dl, F, T)
    local = p_star * np.power(this_alpha,2) * squared_modulus(grad(phi, dl))
    return 1/(np.power(T,3)*np.sum(local))
  
def entropy_production (p_star, dl, F, T):
    nu_star = nu (p_star, dl, F, T)
    ep = (1/T)*np.sum(p_star*squared_modulus(nu_star))
    return ep
  
def spatial_Fisher (p_star, dl, T):
    return np.power(T,2)*np.sum(p_star*squared_modulus(grad_ln(p_star, dl)))

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

def analytical_displacement (p_star, dl, F, T, epsilon):#, decay):
    H = Hessian (np.log(p_star), dl)
    H_grad_zeta = matrix_Vector_prod(H, grad(zeta (p_star, dl), dl))
    local = p_star * dot_prod(H_grad_zeta, nu(p_star, dl, F, T))# * decay
    return -T*np.power(epsilon,2)*np.sum(local)
  
      
class Fields:
    
    def __init__(self, l, dl, nonlinearity, n_std, coeff_smoothing, precision_smoothing):
        norm = 1/nonlinearity
        self.l = l
        self.dl = dl
        self.x = np.array([-l + i*dl for i in range(0,int(2*l/dl))])
        self.y = np.array([-l + i*dl for i in range(0,int(2*l/dl))])
        self.unitMatrix = np.outer(np.ones_like(self.x),np.ones_like(self.y))
        self.xMatrix = np.outer(self.x,np.ones_like(self.y))
        self.yMatrix = np.outer(np.ones_like(self.x),self.y)
        self.xx = np.outer(np.power(self.x,2),np.ones_like(self.y))
        self.yy = np.outer(np.ones_like(self.x),np.power(self.y,2))
        self.xy = np.outer(self.x,self.y)
        self.xxx = np.outer(np.power(self.x,3),np.ones_like(self.y))
        self.yyy = np.outer(np.ones_like(self.x),np.power(self.y,3))
        self.xxy = np.outer(np.power(self.x,2),self.y)
        self.xyy = np.outer(self.x,np.power(self.y,2))
        self.decay = np.exp(-(self.xx + self.yy)/(2*np.power(l/n_std,2)))
        self.xMatrix_n = self.xMatrix/norm
        self.yMatrix_n = self.yMatrix/norm
        self.xx_n = self.xx/np.power(norm,2)
        self.yy_n = self.yy/np.power(norm,2)
        self.xy_n = self.xy/np.power(norm,2)
        self.xxx_n = self.xxx/np.power(norm,3)
        self.yyy_n = self.yyy/np.power(norm,3)
        self.xxy_n = self.xxy/np.power(norm,3)
        self.xyy_n = self.xyy/np.power(norm,3)
        self.coeff_smoothing = coeff_smoothing
        self.precision_smoothing = precision_smoothing
             
    def getRandomDistribution(self, parameters):      
        f00, fx, fy, fxx, fyy, fxy, fxxx, fyyy, fxxy, fxyy = parameters        
        f =  f00*self.unitMatrix + fx*self.xMatrix_n + fy*self.yMatrix_n + (1/2)*fxx*self.xx_n + (1/2)*fyy*self.yy_n + fxy*self.xy_n + \
                (1/6)*fxxx*self.xxx_n + (1/6)*fyyy*self.yyy_n + (1/2)*fxxy*self.xxy_n + (1/2)*fxyy*self.xyy_n        
        p = self.decay * np.power(f,2)
        p /= np.sum(p)
        p = smoothing (p, self.dl, self.coeff_smoothing, self.precision_smoothing)
        return p
    
    def getRandomPerturbation(self, p, strength, parameters):
        f00, fx, fy, fxx, fyy, fxy, fxxx, fyyy, fxxy, fxyy = parameters        
        f =  f00*self.unitMatrix + fx*self.xMatrix_n + fy*self.yMatrix_n + (1/2)*fxx*self.xx_n + (1/2)*fyy*self.yy_n + fxy*self.xy_n + \
                (1/6)*fxxx*self.xxx_n + (1/6)*fyyy*self.yyy_n + (1/2)*fxxy*self.xxy_n + (1/2)*fxyy*self.xyy_n        
        phi = self.decay * f
        norm = np.sqrt(np.sum(p*np.power(phi,2)))
        phi = strength * phi / norm
        phi = smoothing (phi, self.dl, self.coeff_smoothing, self.precision_smoothing)
        phi = phi - np.sum(p*phi)
        return phi
    
    def getPeriodicPerturbation(self, p, epsilon, k, theta, phase):
        pre_phi = epsilon*np.cos(k*(np.cos(theta)*self.xMatrix +np.sin(theta)*self.yMatrix)+phase)
        pre_phi = smoothing (pre_phi, self.dl, self.coeff_smoothing, self.precision_smoothing)        
        phi = pre_phi - np.sum(p*pre_phi)
        return phi
        
    def getDisplacementPerturbation(self, p, epsilon, theta, phi_smoothing): 
        pre_phi = -epsilon * dot_prod(direction_vec (theta), grad_ln (p, self.dl))
        pre_phi = smoothing (pre_phi, self.dl, phi_smoothing, self.precision_smoothing)      
        phi = pre_phi - np.sum(p*pre_phi)
        return phi
    
    def getRandomField(self, force, parameters_x, parameters_y, conf_par1, conf_par2):        
        f00, fx, fy, fxx, fyy, fxy, fxxx, fyyy, fxxy, fxyy = parameters_x
        g00, gx, gy, gxx, gyy, gxy, gxxx, gyyy, gxxy, gxyy = parameters_y
        f =  f00*self.unitMatrix + fx*self.xMatrix + fy*self.yMatrix + (1/2)*fxx*self.xx + (1/2)*fyy*self.yy + fxy*self.xy + \
                (1/6)*fxxx*self.xxx + (1/6)*fyyy*self.yyy + (1/2)*fxxy*self.xxy + (1/2)*fxyy*self.xyy        
        g =  g00*self.unitMatrix + gx*self.xMatrix + gy*self.yMatrix + (1/2)*gxx*self.xx + (1/2)*gyy*self.yy + gxy*self.xy + \
                (1/6)*gxxx*self.xxx + (1/6)*gyyy*self.yyy + (1/2)*gxxy*self.xxy + (1/2)*gxyy*self.xyy
        confining_x = np.outer(-np.exp(conf_par2*self.x/self.l)+np.exp(-conf_par2*self.x/self.l),np.ones_like(self.y))
        confining_y = np.outer(np.ones_like(self.x),-np.exp(conf_par2*self.y/self.l)+np.exp(-conf_par2*self.y/self.l))
        x_comp = force*(f*self.decay+conf_par1*confining_x)
        y_comp = force*(g*self.decay+conf_par1*confining_y)
        x_comp = smoothing(x_comp, self.dl, self.coeff_smoothing, self.precision_smoothing)
        y_comp = smoothing(y_comp, self.dl, self.coeff_smoothing, self.precision_smoothing)
        return np.array([x_comp, y_comp])
    
    def getOUField(self, gamma):
        x_comp = -self.xMatrix
        y_comp = gamma * (-self.yMatrix + self.xMatrix)
        x_comp = smoothing(x_comp, self.dl, self.coeff_smoothing, self.precision_smoothing)
        y_comp = smoothing(y_comp, self.dl, self.coeff_smoothing, self.precision_smoothing)
        return np.array([x_comp, y_comp])
  
    def getQuadraticField(self, gamma, mu1, mu2):
        x_comp = -self.xMatrix
        y_comp = gamma * (-self.yMatrix + mu1*self.xx -mu2*self.unitMatrix)
        x_comp = smoothing(x_comp, self.dl, self.coeff_smoothing, self.precision_smoothing)
        y_comp = smoothing(y_comp, self.dl, self.coeff_smoothing, self.precision_smoothing)
        return np.array([x_comp, y_comp])
  
    