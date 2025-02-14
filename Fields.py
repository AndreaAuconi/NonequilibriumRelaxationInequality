import numpy as np
from functions import *

class Fields:
    
    def __init__(self, l, dl, nonlinearity, n_std, coeff_smoothing, precision_smoothing, smooth_exponent):
        norm = l/nonlinearity
        self.l = l
        self.dl = dl
        self.x = np.array([-l + (i+0.5)*dl for i in range(0,int(2*l/dl))])
        self.y = np.array([-l + (i+0.5)*dl for i in range(0,int(2*l/dl))])
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
        border_smooth_x = np.cos((np.pi/2)*np.power(self.xMatrix/l, smooth_exponent))
        border_smooth_y = np.cos((np.pi/2)*np.power(self.yMatrix/l, smooth_exponent))
        self.border_smooth = border_smooth_x*border_smooth_y
        
    def getRandomDistribution(self, parameters):      
        f00, fx, fy, fxx, fyy, fxy, fxxx, fyyy, fxxy, fxyy = parameters        
        f =  f00*self.unitMatrix + fx*self.xMatrix_n + fy*self.yMatrix_n + (1/2)*fxx*self.xx_n + (1/2)*fyy*self.yy_n + fxy*self.xy_n + \
                (1/6)*fxxx*self.xxx_n + (1/6)*fyyy*self.yyy_n + (1/2)*fxxy*self.xxy_n + (1/2)*fxyy*self.xyy_n        
        p = self.decay * np.power(f,2)
        p /= np.sum(p)
        p = smoothing (p, self.dl, self.coeff_smoothing, self.precision_smoothing)
        return p
    
    def getPeriodicPerturbation(self, p, epsilon, k, theta, phase):
        phi = epsilon*np.sin(k*(np.cos(theta)*self.xMatrix +np.sin(theta)*self.yMatrix)+phase)
        phi = phi - E(p, phi)
        phi *= self.border_smooth
        phi = smoothing (phi, self.dl, self.coeff_smoothing, self.precision_smoothing)        
        phi = phi - E(p, phi)
        return phi
    
    def getDisplacementPerturbation(self, p, epsilon, theta, dl):
        Fx, Fy = epsilon * direction_vec (theta)
        F_displacement = np.array([Fx*np.ones_like(p), Fy*np.ones_like(p)])
        disc = 500
        Rx, Mx, Ry, My = evolution_Matrices (F_displacement, 0., dl, 1/disc)
        displacement_dyn_step = lambda p: dynamics_step(p, Rx, Mx, Ry, My)
        q = p.copy()
        for t in range(disc):
            q = displacement_dyn_step (q)
        phi = q/p - 1.
        phi = phi - E(p, phi)
        bounds = 0.8
        phi = np.maximum(phi, -bounds*np.ones_like(phi))
        phi = np.minimum(phi, bounds*np.ones_like(phi)) 
        phi = smoothing (phi, self.dl, self.coeff_smoothing, self.precision_smoothing)      
        phi = phi - E(p, phi)
        return phi
        
    def getRandomField(self, force, parameters_x, parameters_y, conf_par1, conf_par2):        
        f00, fx, fy, fxx, fyy, fxy, fxxx, fyyy, fxxy, fxyy = parameters_x
        g00, gx, gy, gxx, gyy, gxy, gxxx, gyyy, gxxy, gxyy = parameters_y
        f =  f00*self.unitMatrix + fx*self.xMatrix_n + fy*self.yMatrix_n + (1/2)*fxx*self.xx_n + (1/2)*fyy*self.yy_n + fxy*self.xy_n + \
                (1/6)*fxxx*self.xxx_n + (1/6)*fyyy*self.yyy_n + (1/2)*fxxy*self.xxy_n + (1/2)*fxyy*self.xyy_n        
        g =  g00*self.unitMatrix + gx*self.xMatrix_n + gy*self.yMatrix_n + (1/2)*gxx*self.xx_n + (1/2)*gyy*self.yy_n + gxy*self.xy_n + \
                (1/6)*gxxx*self.xxx_n + (1/6)*gyyy*self.yyy_n + (1/2)*gxxy*self.xxy_n + (1/2)*gxyy*self.xyy_n
        confining_x = np.outer(-np.exp(conf_par2*self.x/self.l)+np.exp(-conf_par2*self.x/self.l),np.ones_like(self.y))
        confining_y = np.outer(np.ones_like(self.x),-np.exp(conf_par2*self.y/self.l)+np.exp(-conf_par2*self.y/self.l))
        x_comp = force*(f*self.decay+conf_par1*confining_x)
        y_comp = force*(g*self.decay+conf_par1*confining_y)
        x_comp *= self.border_smooth
        y_comp *= self.border_smooth
        x_comp = smoothing(x_comp, self.dl, self.coeff_smoothing, self.precision_smoothing)
        y_comp = smoothing(y_comp, self.dl, self.coeff_smoothing, self.precision_smoothing)
        return np.array([x_comp, y_comp])
    
    def getOUField(self, beta, gamma):
        x_comp = -beta*self.xMatrix
        y_comp = gamma * (-self.yMatrix + self.xMatrix)
        x_comp *= self.border_smooth
        y_comp *= self.border_smooth
        x_comp = smoothing(x_comp, self.dl, self.coeff_smoothing, self.precision_smoothing)
        y_comp = smoothing(y_comp, self.dl, self.coeff_smoothing, self.precision_smoothing)
        return np.array([x_comp, y_comp])
  
    def getQuadraticField(self, beta, gamma, mu1, mu2):
        x_comp = -beta*self.xMatrix
        y_comp = gamma * (-self.yMatrix + mu1*self.xx -mu2*self.unitMatrix)
        x_comp *= self.border_smooth
        y_comp *= self.border_smooth
        x_comp = smoothing(x_comp, self.dl, self.coeff_smoothing, self.precision_smoothing)
        y_comp = smoothing(y_comp, self.dl, self.coeff_smoothing, self.precision_smoothing)
        return np.array([x_comp, y_comp])


