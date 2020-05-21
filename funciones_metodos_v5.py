# -*- coding: utf-8 -*-
"""
Notas de versión:
- Se agregó el método de lmfit
    
- Versión incial
"""
#%%
import numpy as np
import json
from funciones_base_v3 import funcion_base_0, funcion_base_1
from lmfit import minimize, Parameters, fit_report
#%%
def valor_funcion_costo(y_exp, y_mod, sigma, cant_p_lineales):
    
    l = float(np.size(y_exp))    
    R = np.divide(y_exp - y_mod, sigma)
    
    return np.sum(np.power(R,2)) / (l - cant_p_lineales)
#%%
def calcular_funciones_base(p_no_lineales, x_exp):
    
    funciones_no_nulas = []
    funciones_base = []
    
    lista_funciones_base = [funcion_base_0(x_exp, p_no_lineales),
                            funcion_base_1(x_exp, p_no_lineales)]
    
    for item in lista_funciones_base:
        if np.count_nonzero(item) > 0:
            funciones_no_nulas.append(True)
            funciones_base.append(item)
        else:
            funciones_no_nulas.append(False)
        
    funciones_base = np.transpose(funciones_base)
    
    return funciones_base, funciones_no_nulas
#%%
def calcular_funcion_modelo(x, funciones_base, p_lineales, p_no_lineales):
    
    if len(funciones_base) != len(p_lineales):
        return print("Error: La cantidad de funciones base no coincide con la cantidad de parámetros lineales")
    resultado = np.zeros_like(x)
    for i in range(len(p_lineales)):
        resultado = resultado + p_lineales[i] *  funciones_base[i](x, p_no_lineales)
        
    return resultado
#%%
def fit_lineal(y_exp, funciones_base_evaluadas, funciones_base_no_nulas, sigma):

    sigma_aux = np.transpose(np.tile(sigma, (sum(funciones_base_no_nulas),1)))

    A = np.divide(funciones_base_evaluadas, sigma_aux)
    A_t = np.transpose(A)
    b = np.divide(y_exp, sigma)

    Alpha = np.dot(A_t, A)
    Beta = np.dot(A_t, b)

    param_lineales_optimos = np.dot(np.linalg.inv(Alpha), Beta)

    return param_lineales_optimos
#%%
def residual(params, x_exp, data, eps_data):
    
    p_lineales = [params[p].value for p in params.valuesdict().keys() if p.startswith("p_l")]
    p_no_lineales = [params[p].value for p in params.valuesdict().keys() if p.startswith("p_no_l")]
    funciones_base_evaluadas, funciones_base_evaluadas_no_nulas = calcular_funciones_base(p_no_lineales, x_exp)
    y_modelo = np.dot(funciones_base_evaluadas, p_lineales)

    return (data - y_modelo) / eps_data
#%%
def imprimirJson(objeto, ruta_de_salida):
    with open(ruta_de_salida, 'w') as outfile:
        json.dump(objeto, outfile, indent = 2)
    #print("terminado de copiar")
    outfile.close()
#%%
def fit_metodo_1(x_exp, y_exp, sigma, valor_funcion_costo_original,
                 p_lineales_original, p_no_lineales_original, 
                 rangos_p_lineales, rangos_p_no_lineales,
                 debug = True,                 
                 n_init = 10, max_iter = 1000, tolerancia = 0                
                ):
    
    # convertimos los input arrays en np.arrays para poder manipular todo correctamente
    rangos_p_lineales = np.array(rangos_p_lineales)
    rangos_p_no_lineales = np.array(rangos_p_no_lineales)
    p_lineales_original = np.array(p_lineales_original)
    p_no_lineales_original = np.array(p_no_lineales_original)
    
    info_debug_inicializaciones = []
    info_debug_optimo_final = []
    valor_funcion_costo_optimo_final = None
    
    cant_p_lineales = len(p_lineales_original)
    cant_p_no_lineales = len(p_no_lineales_original)
    lim_inf_lineal = [rango[0] for rango in rangos_p_lineales]
    lim_sup_lineal = [rango[1] for rango in rangos_p_lineales]  
    lim_inf_no_lineal = [rango[0] for rango in rangos_p_no_lineales]
    lim_sup_no_lineal = [rango[1] for rango in rangos_p_no_lineales]  
        
    for init in range(n_init):
        
        if debug:
            info_debug_inicializaciones.append([])
        
        #seteamos un punto al azar que en principio será el óptimo y que iremos mejorando con las iteraciones
        #seteamos una semilla por cada incialización para poder comparar todos los métodos
        prng = np.random.RandomState(init)
              
        p_lineales_optimo = prng.uniform(low = lim_inf_lineal, 
                                              high = lim_sup_lineal)
      
        p_no_lineales_optimo = prng.uniform(low = lim_inf_no_lineal, 
                                              high = lim_sup_no_lineal)
        
        funciones_base_evaluadas_optimo, funciones_base_evaluadas_optimo_no_nulas = calcular_funciones_base(p_no_lineales_optimo, x_exp)
        y_modelo_optimo = np.dot(funciones_base_evaluadas_optimo, p_lineales_optimo)
        valor_funcion_costo_optimo = valor_funcion_costo(y_exp, y_modelo_optimo, sigma, cant_p_lineales)
            
        #tratamos aparte el caso en que, con cierto parametro, la función base se haga nula
        for i, j in enumerate(funciones_base_evaluadas_optimo_no_nulas):
            if j == False:
                p_lineales_optimo = np.insert(p_lineales_optimo, i, 0)
        
        if debug:
            info_debug = []
            info_debug.append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo,
                               [-1] * cant_p_no_lineales, [-1] * cant_p_lineales, -1])
            info_debug_inicializaciones[init].append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo])
        
        iteracion = 1
        while iteracion < max_iter and np.abs(valor_funcion_costo_optimo - valor_funcion_costo_original) > tolerancia:
            
            p_lineales_new = prng.uniform(low = lim_inf_lineal, 
                                                  high = lim_sup_lineal)
            p_no_lineales_new = prng.uniform(low = lim_inf_no_lineal, 
                                                  high = lim_sup_no_lineal)  
            funciones_base_evaluadas_new, funciones_base_evaluadas_new_no_nulas = calcular_funciones_base(p_no_lineales_new, x_exp)
            
            #calculamos chi 2
            y_modelo_new = np.dot(funciones_base_evaluadas_new, p_lineales_new)
            valor_funcion_costo_new = valor_funcion_costo(y_exp, y_modelo_new, sigma, cant_p_lineales)
            
            #si chi2 decae, guardamos los parametros no lineales
            if valor_funcion_costo_new < valor_funcion_costo_optimo:
                
                for i, j in enumerate(funciones_base_evaluadas_new_no_nulas):
                    if j == False:
                        p_lineales_new = np.insert(p_lineales_new, i, 0)
                        
                p_no_lineales_optimo = p_no_lineales_new
                p_lineales_optimo = p_lineales_new
                valor_funcion_costo_optimo = valor_funcion_costo_new
            
            if debug:    
                info_debug.append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo, 
                                   p_no_lineales_new.tolist(), p_lineales_new.tolist(), valor_funcion_costo_new])
            
            iteracion += 1
        
        if debug:
            info_debug_inicializaciones[init].append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo])
        
        if valor_funcion_costo_optimo_final == None or valor_funcion_costo_optimo < valor_funcion_costo_optimo_final:
            if debug:
                info_debug_optimo_final = info_debug
            valor_funcion_costo_optimo_final = valor_funcion_costo_optimo
            p_lineales_optimo_final = p_lineales_optimo
            p_no_lineales_optimo_final = p_no_lineales_optimo
    
    if debug:
        return [valor_funcion_costo_optimo_final, p_lineales_optimo_final, p_no_lineales_optimo_final, info_debug_optimo_final, info_debug_inicializaciones]
    else:
        return [valor_funcion_costo_optimo_final, p_lineales_optimo_final, p_no_lineales_optimo_final, [], []]


#%%
def fit_metodo_1_1(x_exp, y_exp, sigma, valor_funcion_costo_original,
                 p_lineales_original, p_no_lineales_original, 
                 rangos_p_lineales, rangos_p_no_lineales, 
                 debug = True,                 
                 n_init = 10, max_iter = 1000, tolerancia = 0                
                ):
    
    # convertimos los input arrays en np.arrays para poder manipular todo correctamente
    rangos_p_no_lineales = np.array(rangos_p_no_lineales)
    p_lineales_original = np.array(p_lineales_original)
    p_no_lineales_original = np.array(p_no_lineales_original)
    
    info_debug_inicializaciones = []
    info_debug_optimo_final = []
    valor_funcion_costo_optimo_final = None
      
    cant_p_lineales = len(p_lineales_original)
    cant_p_no_lineales = len(p_no_lineales_original)
    lim_inf_no_lineal = [rango[0] for rango in rangos_p_no_lineales]
    lim_sup_no_lineal = [rango[1] for rango in rangos_p_no_lineales]  
        
    for init in range(n_init):
        
        if debug:
            info_debug_inicializaciones.append([])
        
        #seteamos un punto al azar que en principio será el óptimo y que iremos mejorando con las iteraciones
        #seteamos una semilla por cada incialización para poder comparar todos los métodos
        prng = np.random.RandomState(init)
      
        p_no_lineales_optimo = prng.uniform(low = lim_inf_no_lineal, 
                                              high = lim_sup_no_lineal)
        
        funciones_base_evaluadas_optimo, funciones_base_evaluadas_optimo_no_nulas = calcular_funciones_base(p_no_lineales_optimo, x_exp)
        p_lineales_optimo = fit_lineal(y_exp, funciones_base_evaluadas_optimo, funciones_base_evaluadas_optimo_no_nulas, sigma)
        y_modelo_optimo = np.dot(funciones_base_evaluadas_optimo, p_lineales_optimo)
        valor_funcion_costo_optimo = valor_funcion_costo(y_exp, y_modelo_optimo, sigma, cant_p_lineales)
            
        #tratamos aparte el caso en que, con cierto parametro, la función base se haga nula
        for i, j in enumerate(funciones_base_evaluadas_optimo_no_nulas):
            if j == False:
                p_lineales_optimo = np.insert(p_lineales_optimo, i, 0)
        
        if debug:
            info_debug = []
            info_debug.append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo,
                               [-1] * cant_p_no_lineales, [-1] * cant_p_lineales, -1])
            info_debug_inicializaciones[init].append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo])
        
        iteracion = 1
        while iteracion < max_iter and np.abs(valor_funcion_costo_optimo - valor_funcion_costo_original) > tolerancia:
            
            p_no_lineales_new = prng.uniform(low = lim_inf_no_lineal, 
                                              high = lim_sup_no_lineal)  
            funciones_base_evaluadas_new, funciones_base_evaluadas_new_no_nulas = calcular_funciones_base(p_no_lineales_new, x_exp)
            p_lineales_new = fit_lineal(y_exp, funciones_base_evaluadas_new, funciones_base_evaluadas_new_no_nulas, sigma)
            
            #calculamos chi 2
            y_modelo_new = np.dot(funciones_base_evaluadas_new, p_lineales_new)
            valor_funcion_costo_new = valor_funcion_costo(y_exp, y_modelo_new, sigma, cant_p_lineales)
            
            p_l_en_rango = True
            for i in range(len(p_lineales_new)):
                if (p_lineales_new[i] < rangos_p_lineales[i][0]) or (p_lineales_new[i] > rangos_p_lineales[i][1]):
                    p_l_en_rango = False
                    break
            
            #si chi2 decae, guardamos los parametros no lineales
            if valor_funcion_costo_new < valor_funcion_costo_optimo and p_l_en_rango == True:
                
                for i, j in enumerate(funciones_base_evaluadas_new_no_nulas):
                    if j == False:
                        p_lineales_new = np.insert(p_lineales_new, i, 0)
                        
                p_no_lineales_optimo = p_no_lineales_new
                p_lineales_optimo = p_lineales_new
                valor_funcion_costo_optimo = valor_funcion_costo_new
            
            if debug:    
                info_debug.append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo, 
                                   p_no_lineales_new.tolist(), p_lineales_new.tolist(), valor_funcion_costo_new])
            
            iteracion += 1
        
        if debug:
            info_debug_inicializaciones[init].append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo])
        
        if valor_funcion_costo_optimo_final == None or valor_funcion_costo_optimo < valor_funcion_costo_optimo_final:
            if debug:
                info_debug_optimo_final = info_debug
            valor_funcion_costo_optimo_final = valor_funcion_costo_optimo
            p_lineales_optimo_final = p_lineales_optimo
            p_no_lineales_optimo_final = p_no_lineales_optimo
    
    if debug:
        return [valor_funcion_costo_optimo_final, p_lineales_optimo_final, p_no_lineales_optimo_final, info_debug_optimo_final, info_debug_inicializaciones]
    else:
        return [valor_funcion_costo_optimo_final, p_lineales_optimo_final, p_no_lineales_optimo_final, [], []]

#%%  
def fit_metodo_3(x_exp, y_exp, sigma, valor_funcion_costo_original,
                 p_lineales_original, p_no_lineales_original, 
                 rangos_p_lineales, rangos_p_no_lineales,
                 debug = True,                 
                 n_init = 10, max_iter = 1000, tolerancia = 0,
                 burn_in_max_iter = 0, burn_in_temp = 0
                ):
    
    # convertimos los input arrays en np.arrays para poder manipular todo correctamente
    rangos_p_no_lineales = np.array(rangos_p_no_lineales)
    p_lineales_original = np.array(p_lineales_original)
    p_no_lineales_original = np.array(p_no_lineales_original)
    
    info_debug_inicializaciones = []
    info_debug_optimo_final = []
    valor_funcion_costo_optimo_final = None
      
    cant_p_lineales = len(p_lineales_original)
    cant_p_no_lineales = len(p_no_lineales_original)
    lim_inf_no_lineal = [rango[0] for rango in rangos_p_no_lineales]
    lim_sup_no_lineal = [rango[1] for rango in rangos_p_no_lineales]  
        
    for init in range(n_init):
        
        if debug:
            info_debug_inicializaciones.append([])
        
        #seteamos un punto al azar que en principio será el óptimo y que iremos mejorando con las iteraciones
        #seteamos una semilla por cada incialización para poder comparar todos los métodos
        prng = np.random.RandomState(init)
      
        p_no_lineales_optimo = prng.uniform(low = lim_inf_no_lineal, 
                                              high = lim_sup_no_lineal)
        
        funciones_base_evaluadas_optimo, funciones_base_evaluadas_optimo_no_nulas = calcular_funciones_base(p_no_lineales_optimo, x_exp)
        p_lineales_optimo = fit_lineal(y_exp, funciones_base_evaluadas_optimo, funciones_base_evaluadas_optimo_no_nulas, sigma)
        y_modelo_optimo = np.dot(funciones_base_evaluadas_optimo, p_lineales_optimo)
        valor_funcion_costo_optimo = valor_funcion_costo(y_exp, y_modelo_optimo, sigma, cant_p_lineales)
            
        #tratamos aparte el caso en que, con cierto parametro, la función base se haga nula
        for i, j in enumerate(funciones_base_evaluadas_optimo_no_nulas):
            if j == False:
                p_lineales_optimo = np.insert(p_lineales_optimo, i, 0)

        p_no_lineales_busqueda = p_no_lineales_optimo
        p_lineales_busqueda = p_lineales_optimo
        valor_funcion_costo_busqueda = valor_funcion_costo_optimo
        
        iteracion = 1

        if iteracion < burn_in_max_iter:
            t = burn_in_temp
        elif valor_funcion_costo_optimo < 10:
            t = valor_funcion_costo_optimo
        else:
            t = valor_funcion_costo_optimo / np.log10(valor_funcion_costo_optimo)
        
        if debug:
            info_debug = []
            info_debug.append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo,
                               [-1] * cant_p_no_lineales, [-1] * cant_p_lineales, -1,
                               [-1] * cant_p_no_lineales, [-1] * cant_p_lineales, -1,
                                 1, t])
            info_debug_inicializaciones[init].append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo])
        
        while iteracion < max_iter and np.abs(valor_funcion_costo_optimo - valor_funcion_costo_original) > tolerancia:
            
            if iteracion < burn_in_max_iter:
                t = burn_in_temp
            elif valor_funcion_costo_optimo < 10:
                t = valor_funcion_costo_optimo
            else:
                t = valor_funcion_costo_optimo / np.log10(valor_funcion_costo_optimo)
            
            p_no_lineales_new = np.array([np.random.normal(loc = p_no_lineales_busqueda[i], scale = np.abs(rangos_p_no_lineales[i][1] - rangos_p_no_lineales[i][0]) * .1 ) for i in range(cant_p_no_lineales)])
            funciones_base_evaluadas_new, funciones_base_evaluadas_new_no_nulas = calcular_funciones_base(p_no_lineales_new, x_exp)
            p_lineales_new = fit_lineal(y_exp, funciones_base_evaluadas_new, funciones_base_evaluadas_new_no_nulas, sigma)
            
            #calculamos chi 2
            y_modelo_new = np.dot(funciones_base_evaluadas_new, p_lineales_new)
            valor_funcion_costo_new = valor_funcion_costo(y_exp, y_modelo_new, sigma, cant_p_lineales)
            
            #nos fijamos si los parametros caen en el rango de busqueda
            p_no_l_en_rango = True
            for i in range(len(p_no_lineales_new)):
                if (p_no_lineales_new[i] < rangos_p_no_lineales[i][0]) or (p_no_lineales_new[i] > rangos_p_no_lineales[i][1]):
                    p_no_l_en_rango = False
                    break
            p_l_en_rango = True
            for i in range(len(p_lineales_new)):
                if (p_lineales_new[i] < rangos_p_lineales[i][0]) or (p_lineales_new[i] > rangos_p_lineales[i][1]):
                    p_l_en_rango = False
                    break
                
            alpha = min(np.exp((valor_funcion_costo_busqueda - valor_funcion_costo_new) / t), 1)        
            u = np.random.uniform(0,1)
            if u <= alpha and p_no_l_en_rango == True and p_l_en_rango == True:
                
                for i, j in enumerate(funciones_base_evaluadas_new_no_nulas):
                    if j == False:
                        p_lineales_new = np.insert(p_lineales_new, i, 0)
                
                p_no_lineales_busqueda = p_no_lineales_new
                p_lineales_busqueda = p_lineales_new
                valor_funcion_costo_busqueda = valor_funcion_costo_new
                
                if valor_funcion_costo_new < valor_funcion_costo_optimo:
                    p_no_lineales_optimo = p_no_lineales_new
                    p_lineales_optimo = p_lineales_new
                    valor_funcion_costo_optimo = valor_funcion_costo_new
            
            if debug:    
                info_debug.append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo,
                                   p_no_lineales_busqueda.tolist(), p_lineales_busqueda.tolist(), valor_funcion_costo_busqueda, 
                                   p_no_lineales_new.tolist(), p_lineales_new.tolist(), valor_funcion_costo_new,
                                   alpha, t])
            
            iteracion += 1
        
        if debug:
            info_debug_inicializaciones[init].append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo])
        
        if valor_funcion_costo_optimo_final == None or valor_funcion_costo_optimo < valor_funcion_costo_optimo_final:
            if debug:
                info_debug_optimo_final = info_debug
            valor_funcion_costo_optimo_final = valor_funcion_costo_optimo
            p_lineales_optimo_final = p_lineales_optimo
            p_no_lineales_optimo_final = p_no_lineales_optimo
    
    if debug:
        return [valor_funcion_costo_optimo_final, p_lineales_optimo_final, p_no_lineales_optimo_final, info_debug_optimo_final, info_debug_inicializaciones]
    else:
        return [valor_funcion_costo_optimo_final, p_lineales_optimo_final, p_no_lineales_optimo_final, [], []]
#%%
def fit_metodo_3_1(x_exp, y_exp, sigma, valor_funcion_costo_original,
                 p_lineales_original, p_no_lineales_original, 
                 rangos_p_lineales, rangos_p_no_lineales,
                 debug = True,                 
                 n_init = 10, max_iter = 1000, tolerancia = 0,
                 burn_in_max_iter = 0, burn_in_temp = 0
                ):
    
    # convertimos los input arrays en np.arrays para poder manipular todo correctamente
    rangos_p_no_lineales = np.array(rangos_p_no_lineales)
    p_lineales_original = np.array(p_lineales_original)
    p_no_lineales_original = np.array(p_no_lineales_original)
    
    info_debug_inicializaciones = []
    info_debug_optimo_final = []
    valor_funcion_costo_optimo_final = None
      
    cant_p_lineales = len(p_lineales_original)
    cant_p_no_lineales = len(p_no_lineales_original)
    lim_inf_no_lineal = [rango[0] for rango in rangos_p_no_lineales]
    lim_sup_no_lineal = [rango[1] for rango in rangos_p_no_lineales]  
        
    for init in range(n_init):
        
        if debug:
            info_debug_inicializaciones.append([])
        
        #seteamos un punto al azar que en principio será el óptimo y que iremos mejorando con las iteraciones
        #seteamos una semilla por cada incialización para poder comparar todos los métodos
        prng = np.random.RandomState(init)
      
        p_no_lineales_optimo = prng.uniform(low = lim_inf_no_lineal, 
                                              high = lim_sup_no_lineal)
        
        funciones_base_evaluadas_optimo, funciones_base_evaluadas_optimo_no_nulas = calcular_funciones_base(p_no_lineales_optimo, x_exp)
        p_lineales_optimo = fit_lineal(y_exp, funciones_base_evaluadas_optimo, funciones_base_evaluadas_optimo_no_nulas, sigma)
        y_modelo_optimo = np.dot(funciones_base_evaluadas_optimo, p_lineales_optimo)
        valor_funcion_costo_optimo = valor_funcion_costo(y_exp, y_modelo_optimo, sigma, cant_p_lineales)
            
        #tratamos aparte el caso en que, con cierto parametro, la función base se haga nula
        for i, j in enumerate(funciones_base_evaluadas_optimo_no_nulas):
            if j == False:
                p_lineales_optimo = np.insert(p_lineales_optimo, i, 0)

        p_no_lineales_busqueda = p_no_lineales_optimo
        p_lineales_busqueda = p_lineales_optimo
        valor_funcion_costo_busqueda = valor_funcion_costo_optimo
        
        iteracion = 1

        if iteracion < burn_in_max_iter:
            t = burn_in_temp
        elif valor_funcion_costo_optimo < 10:
            t = valor_funcion_costo_optimo
        else:
            t = valor_funcion_costo_optimo / np.log10(valor_funcion_costo_optimo)
        
        if debug:
            info_debug = []
            info_debug.append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo,
                               [-1] * cant_p_no_lineales, [-1] * cant_p_lineales, -1,
                               [-1] * cant_p_no_lineales, [-1] * cant_p_lineales, -1,
                                 1, t])
            info_debug_inicializaciones[init].append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo])
        
        while iteracion < max_iter and np.abs(valor_funcion_costo_optimo - valor_funcion_costo_original) > tolerancia:
            
            if iteracion < burn_in_max_iter:
                t = burn_in_temp
            elif valor_funcion_costo_optimo < 10:
                t = valor_funcion_costo_optimo
            else:
                t = valor_funcion_costo_optimo / np.log10(valor_funcion_costo_optimo)
            
            p_no_lineales_new = np.array([np.random.normal(loc = p_no_lineales_busqueda[i], scale = np.abs(rangos_p_no_lineales[i][1] - rangos_p_no_lineales[i][0]) * .1 ) for i in range(cant_p_no_lineales)])
            funciones_base_evaluadas_new, funciones_base_evaluadas_new_no_nulas = calcular_funciones_base(p_no_lineales_new, x_exp)
            p_lineales_new = fit_lineal(y_exp, funciones_base_evaluadas_new, funciones_base_evaluadas_new_no_nulas, sigma)
            
            #calculamos chi 2
            y_modelo_new = np.dot(funciones_base_evaluadas_new, p_lineales_new)
            valor_funcion_costo_new = valor_funcion_costo(y_exp, y_modelo_new, sigma, cant_p_lineales)
            
            #nos fijamos si los parametros caen en el rango de busqueda
            p_no_l_en_rango = True
            for i in range(len(p_no_lineales_new)):
                if (p_no_lineales_new[i] < rangos_p_no_lineales[i][0]) or (p_no_lineales_new[i] > rangos_p_no_lineales[i][1]):
                    p_no_l_en_rango = False
                    break
            p_l_en_rango = True
            for i in range(len(p_lineales_new)):
                if (p_lineales_new[i] < rangos_p_lineales[i][0]) or (p_lineales_new[i] > rangos_p_lineales[i][1]):
                    p_l_en_rango = False
                    break
                
            alpha = min(np.exp((valor_funcion_costo_busqueda - valor_funcion_costo_new) / t), 1)        
            u = np.random.uniform(0,1)
            if u <= alpha and p_no_l_en_rango == True and p_l_en_rango == True:
                
                for i, j in enumerate(funciones_base_evaluadas_new_no_nulas):
                    if j == False:
                        p_lineales_new = np.insert(p_lineales_new, i, 0)
                
                p_no_lineales_busqueda = p_no_lineales_new
                p_lineales_busqueda = p_lineales_new
                valor_funcion_costo_busqueda = valor_funcion_costo_new
                
                if valor_funcion_costo_new < valor_funcion_costo_optimo:
                    p_no_lineales_optimo = p_no_lineales_new
                    p_lineales_optimo = p_lineales_new
                    valor_funcion_costo_optimo = valor_funcion_costo_new
            
            if debug:    
                info_debug.append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo,
                                   p_no_lineales_busqueda.tolist(), p_lineales_busqueda.tolist(), valor_funcion_costo_busqueda, 
                                   p_no_lineales_new.tolist(), p_lineales_new.tolist(), valor_funcion_costo_new,
                                   alpha, t])
            
            iteracion += 1
        
        if debug:
            info_debug_inicializaciones[init].append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo])
        
        if valor_funcion_costo_optimo_final == None or valor_funcion_costo_optimo < valor_funcion_costo_optimo_final:
            if debug:
                info_debug_optimo_final = info_debug
            valor_funcion_costo_optimo_final = valor_funcion_costo_optimo
            p_lineales_optimo_final = p_lineales_optimo
            p_no_lineales_optimo_final = p_no_lineales_optimo
    
    if debug:
        return [valor_funcion_costo_optimo_final, p_lineales_optimo_final, p_no_lineales_optimo_final, info_debug_optimo_final, info_debug_inicializaciones]
    else:
        return [valor_funcion_costo_optimo_final, p_lineales_optimo_final, p_no_lineales_optimo_final, [], []]
#%%
def fit_metodo_2(x_exp, y_exp, sigma, valor_funcion_costo_original,
                 p_lineales_original, p_no_lineales_original, 
                 rangos_p_lineales, rangos_p_no_lineales,     
                 debug = True,          
                 n_init = 10               
                ):
    
    # convertimos los input arrays en np.arrays para poder manipular todo correctamente
    rangos_p_lineales = np.array(rangos_p_lineales)
    rangos_p_no_lineales = np.array(rangos_p_no_lineales)
    p_lineales_original = np.array(p_lineales_original)
    p_no_lineales_original = np.array(p_no_lineales_original)
    
    info_debug_inicializaciones = []
    info_debug_optimo_final = []
    valor_funcion_costo_optimo_final = None
    
    cant_p_lineales = len(p_lineales_original)
    cant_p_no_lineales = len(p_no_lineales_original)
    lim_inf_lineal = [rango[0] for rango in rangos_p_lineales]
    lim_sup_lineal = [rango[1] for rango in rangos_p_lineales]  
    lim_inf_no_lineal = [rango[0] for rango in rangos_p_no_lineales]
    lim_sup_no_lineal = [rango[1] for rango in rangos_p_no_lineales]  
        
    for init in range(n_init):
       
        #seteamos un punto al azar que en principio será el óptimo y que iremos mejorando con las iteraciones
        #seteamos una semilla por cada incialización para poder comparar todos los métodos
        prng = np.random.RandomState(init)
              
        p_lineales_optimo = prng.uniform(low = lim_inf_lineal, 
                                              high = lim_sup_lineal)
      
        p_no_lineales_optimo = prng.uniform(low = lim_inf_no_lineal, 
                                              high = lim_sup_no_lineal)
        
        funciones_base_evaluadas_optimo, funciones_base_evaluadas_optimo_no_nulas = calcular_funciones_base(p_no_lineales_optimo, x_exp)
        #calculamos chi 2
        y_modelo_optimo = np.dot(funciones_base_evaluadas_optimo, p_lineales_optimo)
        valor_funcion_costo_optimo = valor_funcion_costo(y_exp, y_modelo_optimo, sigma, cant_p_lineales)

        if debug:
            info_debug = []
            info_debug.append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo,
                               p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo])
            info_debug_inicializaciones.append([])
            info_debug_inicializaciones[init].append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo])

        params = Parameters()
        for k in range(len(p_lineales_optimo)):
        	params.add('p_l_' + str(k), min=rangos_p_lineales[k][0], max=rangos_p_lineales[k][1], value = p_lineales_optimo[k])
        for k in range(len(p_no_lineales_optimo)):
            params.add('p_no_l_' + str(k), min=rangos_p_no_lineales[k][0], max=rangos_p_no_lineales[k][1], value = p_no_lineales_optimo[k])
	    
        out = minimize(residual, params, args=(x_exp, y_exp, sigma))
        best_params = out.params.valuesdict()
        p_lineales_optimo = np.array([best_params["p_l_" + str(k)] for k in range(len(p_lineales_optimo))])
        p_no_lineales_optimo = np.array([best_params["p_no_l_" + str(k)] for k in range(len(p_no_lineales_optimo))])
        funciones_base_evaluadas_optimo, funciones_base_evaluadas_optimo_no_nulas = calcular_funciones_base(p_no_lineales_optimo, x_exp)
        #calculamos chi 2
        y_modelo_optimo = np.dot(funciones_base_evaluadas_optimo, p_lineales_optimo)
        valor_funcion_costo_optimo = valor_funcion_costo(y_exp, y_modelo_optimo, sigma, cant_p_lineales)       
        
        if debug:
            info_debug.append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo,
                               p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo])
            info_debug_inicializaciones[init].append([p_no_lineales_optimo.tolist(), p_lineales_optimo.tolist(), valor_funcion_costo_optimo])
 
        
        if valor_funcion_costo_optimo_final == None or valor_funcion_costo_optimo < valor_funcion_costo_optimo_final:
            if debug:
                info_debug_optimo_final = info_debug
            valor_funcion_costo_optimo_final = valor_funcion_costo_optimo
            p_lineales_optimo_final = p_lineales_optimo
            p_no_lineales_optimo_final = p_no_lineales_optimo
    
    if debug:
        return [valor_funcion_costo_optimo_final, p_lineales_optimo_final, p_no_lineales_optimo_final, info_debug_optimo_final, info_debug_inicializaciones]
    else:
        return [valor_funcion_costo_optimo_final, p_lineales_optimo_final, p_no_lineales_optimo_final, [], []]