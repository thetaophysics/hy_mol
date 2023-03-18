import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import time
import os

def main():
    start = time.time()
    N_t = 300       #num of trials     
    r = 0.529       #Bohr's radius [Angstrom]    
    s = 0.74        #distance between 2 protons [Angstrom]
    red_s = s/r     #[reduced unit]
    Ry = 13.6       #[eV]
    e0 = 2*Ry       #unit energy [eV] 

    # step = 0.33
    step = np.arange(0.1, 2., 0.01)

    #variational params
    alpha = 2.      
    red_a = newton_ralphson(red_s)   #[Angstrom]
    print(f's = {s}[A] or = {round(red_s,2)}[unitless]')
    print(f'a = {round(red_a*r,2)}[A] or = {round(red_a,2)}[unitless]')
    b = 0.8

    # Chosing stepsize
    for dx in step:
        Pa = accept_prob(red_a, alpha, b, N_t, dx, red_s, e0)
        if Pa >= 50 and Pa <= 55:
            print(f'P_A = {round(Pa)}% at step = {round(dx, 3)}')
            N_t = 400000   #update N_tot to achieve desired precision of 1e-3
            stepsize = dx   #if the condition of acceptance probability does not meet, 
            break
    stepsize = round(stepsize, 2)

    beta = np.arange(0.2, 2.1, 0.1)
    energy_final = []
    var = []

    #Varying beta
    for b in beta:
        b = round(b,1)
        E0_final, N_e, sigma = main_loop2(red_a, alpha, b, N_t, stepsize, red_s, e0)
        energy_final.append(E0_final)
        var.append(sigma)

    
    E_final = np.array(energy_final)
    idx_beta = E_final.argmin()             #index of the minimum energy 
    idx_sigma = E_final.argmin()
    best_beta = beta[idx_beta]              #beta value at minimum energy
    E_sig = var[idx_sigma]

    print('################# Varying Beta ###################')
    print(f'alpha = {alpha}')
    print(f'beta at minimum = {round(best_beta,2)}')
    print(f'Minimum energy E0 = {round(E_final.min(),5)}[eV] +/- {round(E_sig,5)}')
    print(f'Binding energy = {abs(round(E_final.min(),5)) - 2*Ry}\n')

    #Optional part 12: varying s
    energy_final_opt = []
    var_opt = []
    vary_s = np.arange(1, 2, 0.1)
    for idx in vary_s:
        idx = round(idx, 1)
        E0_final_opt, N_e_opt, sigma_opt = main_loop2(red_a, alpha, round(best_beta,2), N_t, stepsize, idx, e0)
        energy_final_opt.append(E0_final_opt)
        var_opt.append(sigma_opt)

    E_final_opt = np.array(energy_final_opt)
    idx_s = E_final_opt.argmin()             #index of the minimum energy 
    idx_sigma_s = E_final_opt.argmin()
    best_s = vary_s[idx_s]
    E_sig_opt = var_opt[idx_sigma_s]

    print('################# Varying s ###################')
    print(f's at minimum E = {round(best_s,1)*r}[Ang] and {round(best_s,1)}[unitless]')
    print(f'Minimum energy E0 with varrying s = {round(E_final_opt.min(),5)}[eV] +/- {round(E_sig_opt,5)}')

    ################################### PLOTTING ###################################
    plt.figure(figsize=(8, 8))
    plt.title('Energy with varrying s')
    plt.xlabel('s')
    plt.ylabel('E0_final')
    plt.errorbar(vary_s,energy_final_opt, yerr = var_opt, color='b' ,marker='o',linestyle='none',linewidth=2.0)
    plt.minorticks_on()

    name_str = "ground_state_energy_"
    filename = name_str + "step_" + str(stepsize) + "_b_" + str(round(best_beta,2)) + ".png"
    print(filename)
    plt.figure(figsize=(8, 8))
    plt.title(f'Ground State Energy with varying beta at ({N_t}) ')
    plt.xlabel('beta')
    plt.ylabel('E0_final')
    plt.errorbar(beta,energy_final, yerr = var, color='r' ,marker='o',linestyle='none',linewidth=2.0)
    plt.minorticks_on()
    plt.show()  
    plt.savefig(filename, format='png')
    plt.close()  
    
    end = time.time()   
    print(f'computing time = {end - start}')
    return

# function definition to compute magnitude of the vector
def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))

def init(r, s):
    # print(f'check for reduced unit: r = {r}, s = {s}')
    r_L = [s/2., 0, 0]
    r_R = [-s/2., 0, 0]
    r_1L = r[:3] + r_L
    r_1R = r[:3] + r_R
    r_2L = r[3:] + r_L
    r_2R = r[3:] + r_R
    r12 = r[:3] - r[3:]     #r12 = r1 - r2

    mag_r1l = magnitude(r_1L)
    mag_r1r = magnitude(r_1R)
    mag_r2l = magnitude(r_2L)
    mag_r2r = magnitude(r_2R)
    mag_r12 = magnitude(r12)

    #dot product
    dot_r1l = np.dot(r_1L, r12)
    dot_r1r = np.dot(r_1R, r12)
    dot_r2l = np.dot(r_2L, r12)
    dot_r2r = np.dot(r_2R, r12)

    mag = [mag_r1l, mag_r1r, mag_r2l, mag_r2r, mag_r12]
    dot = [dot_r1l, dot_r1r, dot_r2l, dot_r2r]
    return r_1L, r_1R, r_2L, r_2R, r12, mag, dot

def wave_func(mag, a, alpha, beta):
    wave1 = np.exp(-mag[0]/a) + np.exp(-mag[1]/a)
    wave2 = np.exp(-mag[2]/a) +np.exp(-mag[3]/a)
    f12 = np.exp(mag[4]/(alpha*(1+beta*mag[4])))
    return [wave1, wave2, f12]

def energy(a, n , wave, mag, dot):
    x = -(1/pow(a,2)) - pow(n,3) * (n/4.+1/mag[4])
    y1 = (1/a) * (np.exp(-mag[0]/a)/mag[0] + np.exp(-mag[1]/a)/mag[1])
    y2 = (pow(n,2)/(2.*a)) * (np.exp(-mag[0]/a) * (dot[0]/(mag[0]*mag[4])) + np.exp(-mag[1]/a) * (dot[1] / (mag[1]*mag[4])))
    z1 = (1/a) * (np.exp(-mag[2]/a)/mag[2] + np.exp(-mag[3]/a)/mag[3])
    z2 = (pow(n,2)/(2.*a)) * (np.exp(-mag[2]/a) * (dot[2]/(mag[2]*mag[4])) + np.exp(-mag[3]/a) * (dot[3] / (mag[3]*mag[4])))
    w = - 1/mag[0] - 1/mag[1] - 1/mag[2] - 1/mag[3] + 1/mag[4]

    E = x + ((1/wave[0]) * (y1 + y2)) + ((1/wave[1]) * (z1 - z2)) + w 
    return E

def main_loop2(a, alpha, beta, N_t, step, s, e0): 
    sum = 0
    sum2 = 0
    E = 0
    E2 = 0
    N_e = 0     #counts of num of times energy is added
    N_a = 0     #Metropolis trial steps
    #Step 1: start with random r_k=[r1,r2]
    rk = np.random.uniform(-0.5,0.5, 6)     #rk = [r1, r2] 
    rk_1L, rk_1R, rk_2L, rk_2R, rk_12, mag_k, dot_k = init(rk,s)
    wave_k = wave_func(mag_k, a, alpha, beta) 
    w_k = pow(wave_k[0] * wave_k[1] * wave_k[2],2) 

    x1 = []
    y1 = []
    z1 = []
    x2 = []
    y2 = []
    z2 = []
    E_arr = []

    #Loop start at step 2
    for trial in range(N_t):
        #Step 2: change coordinates (generate trial configuration)
        dk = np.random.uniform(-0.5,0.5, 6)
        rt = rk + dk * step 
        
        # if (beta == 0.7):
        x1.append(rt[0])
        y1.append(rt[1])
        z1.append(rt[2])
        x2.append(rt[3])
        y2.append(rt[4])
        z2.append(rt[5])

        rt_1L, rt_1R, rt_2L, rt_2R, rt_12, mag_t, dot_t = init(rt,s)
        wave_t = wave_func(mag_t, a, alpha, beta)

        #Step 3: Calculate chi_rt_sqsuared
        w_t = pow(wave_t[0] * wave_t[1] * wave_t[2],2)   
        #ratio
        w_ratio = w_t / w_k

        #Step 4: If conditions
        if (w_ratio >= 1.): 
            rk = rt 
            w_k = w_t
            new_rk_1L, new_rk_1R, new_rk_2L, new_rk_2R, new_rk_12, new_mag_k, new_dot_k = init(rk,s)
            new_wave_k = wave_func(new_mag_k, a, alpha, beta) 
            n = 1./(1. + beta*new_mag_k[4])
            E = energy(a, n, new_wave_k, new_mag_k, new_dot_k)
            E2 = pow(energy(a, n, new_wave_k, new_mag_k, new_dot_k),2)
            E_arr.append(E)
            sum = sum + E
            sum2 = sum2 + E2
            N_e += 1
            N_a += 1
        else:
            p0 = np.random.uniform(0,1)
            if (w_ratio >= p0): 
                rk = rt
                w_k = w_t
                new_rk_1L, new_rk_1R, new_rk_2L, new_rk_2R, new_rk_12, new_mag_k, new_dot_k = init(rk,s)
                new_wave_k = wave_func(new_mag_k, a, alpha, beta) 
                n = 1./(1. + beta*new_mag_k[4])
                E = energy(a, n, new_wave_k, new_mag_k, new_dot_k)
                E2 = pow(energy(a, n, new_wave_k, new_mag_k, new_dot_k),2)
                E_arr.append(E2)
                sum = sum + E
                sum2 = sum2 + E2
                N_e += 1
                N_a += 1
            else: 
                sum = sum + E 
                sum2 = sum2 + E2
                N_e+=1
                continue
    
    plot(x1,y1,z1,x2,y2,z2, E_arr, beta)
        
    #Step 8: Calculate E0 = <E>
    if N_e!=0:
        energy0 = sum / N_e
        energy0_sqrt = sum2 / N_e
        E0_final = (energy0 + (1./s)) * e0    #[eV]
        sigma = np.sqrt((1/N_e)*(energy0_sqrt - energy0))
        print(f'Metropolis steps = {N_a} at beta = {beta}')

    else:
        print('All trials failed!')
        return 0
    
    return E0_final, N_e, sigma

#Add Newton Ralphson function to solve for a
def f(a, red_s):
    return 1/(1+np.exp(-red_s/a)) - a
def f_prime(a,red_s):
    return -1 - (red_s*np.exp(-red_s/a))/(pow(a,2) * pow(np.exp(-red_s/a) + 1, 2))
def newton_ralphson(red_s):
    max_Iter = 10
    tot = 1E-8
    n = 0           #iteration counter
    a0 = 0.8        #given initial guess a
    ai_1 = a0
    s = red_s
    # print("n\ta\tfunc\tfunc_prime")
    while abs(f(ai_1,s)) > tot or n > max_Iter:
        n+=1
        ai = ai_1 - f(ai_1,s)/f_prime(ai_1,s)       #Newton-Raphson equation
        ai_1 = ai 
    return ai_1
#Find stepsize an P_A acceptance probability
def accept_prob(a, alpha, beta, N_t, step, s, e0): 
    sum = 0
    N_e = 0     #counts of num of times energy is added\
    N_a = 0
    #Step 1: start with random r_k=[r1,r2]
    rk = np.random.uniform(-0.5,0.5, 6)     #rk = [r1, r2] 
    rk_1L, rk_1R, rk_2L, rk_2R, rk_12, mag_k, dot_k = init(rk,s)
    wave_k = wave_func(mag_k, a, alpha, beta) 
    w_k = pow(wave_k[0] * wave_k[1] * wave_k[2],2) 

    #Loop start at step 2
    for trial in range(N_t):
        #Step 2: change coordinates (generate trial configuration)
        dk = np.random.uniform(-0.5,0.5, 6)
        rt = rk + dk * step #rt depends on rk
        rt_1L, rt_1R, rt_2L, rt_2R, rt_12, mag_t, dot_t = init(rt,s)
        wave_t = wave_func(mag_t, a, alpha, beta)

        #Step 3: Calculate chi_rt_sqsuared
        w_t = pow(wave_t[0] * wave_t[1] * wave_t[2],2)   
        w_ratio = w_t / w_k
        #Step 4: If conditions
        if (w_ratio >= 1.): 
            rk = rt 
            w_k = w_t
            N_a += 1
        else:
            p0 = np.random.uniform(0,1)
            if (w_ratio >= p0): 
                rk = rt
                w_k = w_t
                N_a+=1
            else: 
                w_t = w_t
                continue

    # print(f'N_t = {N_t}\tN_a = {N_a}')
    P_a = N_a/N_t*100
    
    return P_a
def plot(x1,y1,z1,x2,y2,z2, E_arr, beta):
    a1 = np.array(x1)
    b1 = np.array(y1)
    c1 = np.array(z1)
    a2 = np.array(x2)
    b2 = np.array(y2)
    c2 = np.array(z2)
    Earr = np.array(E_arr)
    
    indices_1 = (c1 >= -0.1) & (c1 <= 0.1)
    indices_2 = (c2 >= -0.1) & (c2 <= 0.1)
    x1_filtered = a1[indices_1]
    y1_filtered = b1[indices_1]
    x2_filtered = a2[indices_2]
    y2_filtered = b2[indices_2]

    fig, ax = plt.subplots()
    ax.scatter(x1_filtered, y1_filtered, c='blue', label='Electron 1')
    ax.scatter(x2_filtered, y2_filtered, c='red', label='Electron 2')
    ax.set_title("Location of the two electrons during the MCMC at beta = " + str(beta))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    filename = f"figure_{beta}.png"
    # filename = os.path.join(directory, f"figure_{beta}.png")
    fig.savefig(filename)
    plt.close(fig)
    return 

if __name__ == '__main__':
    main()

