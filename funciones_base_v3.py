import numpy as np

S = lambda x, a : 2 * np.sin(x * a / 2.) / x

def funcion_base_0(x, p_no_lineales):
    
    return np.ones_like(x)

def funcion_base_1(x, p_no_lineales):
    
    x = np.array(x)
    h = p_no_lineales[0]
    w = p_no_lineales[1]
    t_h = p_no_lineales[2]
    t_w = p_no_lineales[3]
    beta = p_no_lineales[4]
    
    P_L = 1 / x
    I = np.zeros_like(x)
    dang = (np.pi / 2.) / 25.
    alfa = np.arange(start = dang, stop = np.pi / 2., step = dang)
    x, ang = np.meshgrid(x, alfa)
    #S = lambda x, a : 2 * np.sin(x * a / 2.) / x
    
    x_h = np.multiply(x, np.cos(ang))
    x_w = np.multiply(x, np.sin(ang))
    aux1 = np.multiply(S(x_h,h),S(x_w,w))
    aux2 = beta / 2. * np.multiply(S(x_h, h + 2. * t_h) - S(x_h, h), S(x_w, w))
    aux3 = beta / 2. * np.multiply(S(x_h, h), S(x_w, w + 2. * t_w) - S(x_w, w))
    aux = np.power(aux1 + aux2 + aux3, 2)
    I = np.sum(aux,axis = 0)
    I = np.multiply(I, dang)
    I = np.multiply(I, P_L)
    return I
