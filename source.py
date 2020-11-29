#Duy Anh Nguyen - 260745838
#MECH 535 - Project Fall 2020

import math
from array import *
import numpy as np
import matplotlib.pyplot as plt

#Assumption (an abitrary value of blade width does not affect the result)
gamma       = 1.4
R           = 287 #J/kg.K
C_p         = 1005 #J/kg.K
T           = 298 #K
blade_width = 0.04 #m

#Given
N           = (6000*2*math.pi)/(60) #rev/s
C_x         = 175 #m/s
p           = 100000 #Pa
d_tip       = 1.0 #m
r_tip       = d_tip/2
d_hub       = 0.9 #m
r_hub       = d_hub/2
rho         = p/(R*T) #kg/m3
m_dot       = rho*C_x*math.pi*(r_tip**2-r_hub**2) #kg/s
T0          = T+C_x**2/(2*C_p)
beta1_tip   = 60*2*math.pi/360 #rad
beta2_tip   = 30*2*math.pi/360 #rad
beta1_hub   = 55.15*2*math.pi/360 #rad
beta2_hub   = 8.64*2*math.pi/360 #rad
loss_total  = 0.05

#Generate a grid with equal spacing in x and r directions (divide blade width to n equal portion)
n           = 4
grid_x_dim  = blade_width * 5
grid_r_dim  = r_tip - r_hub
element_dim = blade_width/n
x_size      = round(grid_x_dim/element_dim)+1
x_pos       = [i for i in range(x_size)]
r_size      = round(grid_r_dim/element_dim)+1
r_pos       = [j for j in range(r_size)]

#Blade position
x_hub_inlet     = int((x_size-n-1)/2)
r_hub_inlet     = 0
x_hub_outlet    = int(x_hub_inlet + n)
r_hub_outlet    = 0

x_tip_inlet     = int((x_size-n-1)/2)
r_tip_inlet     = int(r_size - 1)
x_tip_outlet    = int(x_hub_inlet + n)
r_tip_outlet    = int(r_size - 1)

#Calculating distance of a point on the blade w.r.t reference position (inlet at the hub)
def x_cal(i, j):
    r_result = ((i-x_hub_inlet)/(x_size-1))*blade_width
    return r_result

def r_cal(i, j):
    r_result = r_hub+(j/(r_size-1))*(r_tip-r_hub)
    return r_result
        
#Linear interpolation of loss in x direction
def loss_cal(i, j):
    if i <= x_hub_inlet or i > x_hub_outlet:
        loss_result = 0
    else:
        loss_result = loss_total*(i-x_hub_inlet)/n
    return loss_result

#Bilinear interpolation of beta angle in x and r directions
def beta_cal(i, j):
    if i <= x_hub_inlet or i > x_hub_outlet:
        beta_result = 0
    else:
        beta_result = (beta1_hub*(x_hub_outlet-i)*(r_tip_outlet-j))/((x_hub_outlet-x_hub_inlet)*(r_tip_outlet-r_hub_inlet)) \
                    + (beta2_hub*(i-x_hub_inlet)*(r_tip_outlet-j))/((x_hub_outlet-x_hub_inlet)*(r_tip_outlet-r_hub_inlet))  \
                    + (beta1_tip*(x_hub_outlet-i)*(j-r_hub_inlet))/((x_hub_outlet-x_hub_inlet)*(r_tip_outlet-r_hub_inlet))  \
                    + (beta2_tip*(i-x_hub_inlet)*(j-r_hub_inlet))/((x_hub_outlet-x_hub_inlet)*(r_tip_outlet-r_hub_inlet))
    return beta_result

#Calculating psi to correspond to uniform flow with no swirl
def psi_cal(i, j):
    psi_result = (r_cal(i, j)**2-r_hub**2)/(r_tip**2-r_hub**2)
    return psi_result

#Initialize loss coefficient, alpha, beta angle and psi distribution matrix
loss_grid   = [[loss_cal(i, j) for i in range(x_size)] for j in range(r_size)]
alpha_grid  = [[0 for i in range(x_size)] for j in range(r_size)]
beta_grid   = [[beta_cal(i, j) for i in range(x_size)] for j in range(r_size)]
psi_grid    = [[psi_cal(i, j) for i in range(x_size)] for j in range(r_size)]

#Initialize velocity C_x matrix with special cases at the bottom and top edges of the grid
C_x_grid = [[0 for i in range(x_size)] for j in range(r_size)]
for i in x_pos:
    for j in r_pos:
        if j == 0:
            C_x_grid[j][i] =  m_dot*(psi_grid[j+1][i]-psi_grid[j][i])/(2*math.pi*rho*r_cal(i, j)*element_dim)
        elif j == max(r_pos):
            C_x_grid[j][i] =  m_dot*(psi_grid[j][i]-psi_grid[j-1][i])/(2*math.pi*rho*r_cal(i, j)*element_dim)
        else:
            C_x_grid[j][i] =  m_dot*(psi_grid[j+1][i]-psi_grid[j-1][i])/(2*math.pi*rho*r_cal(i, j)*2*element_dim)

#Initialize velocity C_r matrix with special cases at the left and right edges of the grid
C_r_grid = [[0 for i in range(x_size)] for j in range(r_size)]
for i in x_pos:
    for j in r_pos:
        if i == 0:
            C_r_grid[j][i] =  -m_dot*(psi_grid[j][i+1]-psi_grid[j][i])/(2*math.pi*rho*r_cal(i, j)*element_dim)
        elif i == max(x_pos):
            C_r_grid[j][i] =  -m_dot*(psi_grid[j][i]-psi_grid[j][i-1])/(2*math.pi*rho*r_cal(i, j)*element_dim)
        else:
            C_r_grid[j][i] =  -m_dot*(psi_grid[j][i+1]-psi_grid[j][i-1])/(2*math.pi*rho*r_cal(i, j)*2*element_dim)

#Initialize velocity C_m matrix
C_m_grid = [[(C_x_grid[j][i]**2+C_r_grid[j][i]**2)**(1/2) for i in range(x_size)] for j in range(r_size)]

#Initialize all other parameters grid for calculation block
T_grid          = [[T0 for i in range(x_size)] for j in range(r_size)]
vorticity_grid  = [[0 for i in range(x_size)] for j in range(r_size)]
V_theta_grid    = [[N*r_cal(i, j) for i in range(x_size)] for j in range(r_size)]
V_sqrt_grid     = [[0 for i in range(x_size)] for j in range(r_size)]
C_sqrt_grid     = [[0 for i in range(x_size)] for j in range(r_size)]
h0_grid         = [[C_p*T0 for i in range(x_size)] for j in range(r_size)]
h0r_grid        = [[C_p*T0 for i in range(x_size)] for j in range(r_size)]
h_grid          = [[h0r_grid[j][i]-V_sqrt_grid[j][i]/2 for i in range(x_size)] for j in range(r_size)]
p0_grid         = [[p for i in range(x_size)] for j in range(r_size)]
p0r_grid        = [[p for i in range(x_size)] for j in range(r_size)]
p0r_ideal_grid  = [[p for i in range(x_size)] for j in range(r_size)]
p_grid          = [[p0_grid[j][i]*(h_grid[j][i]/h0_grid[j][i])**(gamma/(gamma-1)) for i in range(x_size)] for j in range(r_size)]
rho_grid        = [[rho for i in range(x_size)] for j in range(r_size)]
s_grid          = [[0 for i in range(x_size)] for j in range(r_size)]

#Starting swirl inside the rotor
C_theta_grid  = [[0 for i in range(x_size)] for j in range(r_size)]
rC_theta_grid = [[0 for i in range(x_size)] for j in range(r_size)]
for i in x_pos:
    for j in r_pos:
        if i > x_hub_inlet and i <= x_hub_outlet:
            C_theta_grid[j][i]  = N*r_cal(i, j)-C_m_grid[j][i]*math.tan(beta_grid[j][i])
            rC_theta_grid[j][i] = r_cal(i, j)*C_theta_grid[j][i]

for run in range(21):
    #Storing values of previous run
    psi_old_grid          = [[psi_grid[j][i] for i in range(x_size)] for j in range(r_size)]
    vorticity_old_grid    = [[vorticity_grid[j][i] for i in range(x_size)] for j in range(r_size)]
    rho_old_grid          = [[rho_grid[j][i] for i in range(x_size)] for j in range(r_size)]

    #Updating vorticity
    for i in x_pos:
        for j in range(1, max(r_pos)):
            m, n = [1, 1]
            if j == 0:
                n = 0
            elif j == max(r_pos):
                m = 0
            C_theta_grid[j][i]   = rC_theta_grid[j][i]/r_cal(i, j)
            vorticity_grid[j][i] = math.pi/(element_dim*m_dot*C_x_grid[j][i])* \
                                (C_theta_grid[j][i]*(rC_theta_grid[j+1*m][i]-rC_theta_grid[j-1*n][i])/r_cal(i, j) \
                                + T_grid[j][i]*(s_grid[j+1*m][i]-s_grid[j-1*n][i])-(h0_grid[j+1*m][i]-h0_grid[j-1*n][i]))
    
    #Updating psi                            
    A_grid = [[0 for i in range(x_size)] for j in range(r_size)]
    B_grid = [[0 for i in range(x_size)] for j in range(r_size)]
    for i in range(1, max(x_pos)+1):
        for j in r_pos:
            k, l, m, n = [1, 1, 1, 1]
            #Special cases left edge
            if i == 0:
                l = 0
                if j == 0:
                    n = 0
                elif j == max(r_pos):
                    m = 0
            #Special cases bottom edge
            elif j == 0:
                n = 0
                if i == max(x_pos):
                    k = 0
            else:
                #Special cases right edge
                if i == max(x_pos):
                    k = 0
                    if j == max(r_pos):
                        m = 0
                #Special cases top edge
                elif j == max(r_pos):
                    m = 0

            A_grid[j][i] = 1/(2/((rho_grid[j][i+1*k]+rho_grid[j][i])*r_cal(i+0.5*k, j))+2/((rho_grid[j][i-1*l]+rho_grid[j][i])*r_cal(i-0.5*l, j)) \
                        +2/((rho_grid[j+1*m][i]+rho_grid[j][i])*r_cal(i, j+1/2*m))+2/((rho_grid[j-1*n][i]+rho_grid[j][i])*r_cal(i, j-0.5*n)))
            B_grid[j][i] = (1/rho_grid[j][i])*(psi_grid[j][i+1*k]/r_cal(i+0.5*k, j)+psi_grid[j][i-1*l]/r_cal(i-0.5*l, j)\
                        +psi_grid[j+1*m][i]/r_cal(i, j+1/2*m)+psi_grid[j-1*n][i]/r_cal(i, j-0.5*n))                                
            if j == 0:
                psi_grid[j][i] = 0
            elif j == max(r_pos):
                psi_grid[j][i] = 1
            else:
                psi_grid[j][i] = A_grid[j][i]*(B_grid[j][i]+(element_dim**2)*vorticity_grid[j][i])
    
    #Updating velocities
    for i in x_pos:
        for j in r_pos:
            k, l, m, n = [1, 1, 1, 1]
            o, p       = [0, 0]
            if j == 0:
                n, p = [0, 1]
            elif j == max(r_pos):
                m, p = [0, 1]
            if i == 0:
                l, o = [0, 1]
            elif i == max(x_pos):
                k, o = [0, 1]

            C_r_grid[j][i] = -m_dot*(psi_grid[j][i+1*k]-psi_grid[j][i-1*l])/(2*math.pi*rho_grid[j][i]*r_cal(i, j)*(2-o)*element_dim)
            C_x_grid[j][i] = m_dot*(psi_grid[j+1*m][i]-psi_grid[j-1*n][i])/(2*math.pi*rho_grid[j][i]*r_cal(i, j)*(2-p)*element_dim)
            C_m_grid[j][i] = (C_x_grid[j][i]**2+C_r_grid[j][i]**2)**(1/2)
                    
    #Updating swirl
    for i in x_pos:
        for j in r_pos:
            if i <= x_hub_inlet:
                C_theta_grid[j][i]      = 0
            elif i > x_hub_inlet and i <= x_hub_outlet:
                C_theta_grid[j][i]      = N*r_cal(i, j)-C_m_grid[j][i]*math.tan(beta_grid[j][i])
            else:
                C_theta_grid[j][i]      = rC_theta_grid[j][i]/r_cal(i, j)

    #Tracing back the stream line and calculating common block
    for i in range(1, max(x_pos)+1):
        i_origin = i-1
        if i > x_hub_inlet and i <= x_hub_outlet:
            i_origin = x_hub_inlet
            N_rotate = N
        else:
            N_rotate = 0
        j_step = 0
        psi_upper = psi_grid[j_step+1][i_origin]
        psi_lower = psi_grid[j_step][i_origin]
        for j in r_pos:
            while(True):
                if psi_grid[j][i] <= psi_upper and psi_grid[j][i] >= psi_lower:
                    break;
                j_step = j_step+1;
                if j_step == max(r_pos):
                    print('Error')
                    j_step = j_step-1
                    break;                
                psi_upper = psi_grid[j_step+1][i_origin]
                psi_lower = psi_grid[j_step][i_origin]
            
            frac = (psi_grid[j][i]-psi_lower)/(psi_upper-psi_lower)
            radius1 = r_cal(i_origin, j_step+frac)
            radius2 = r_cal(i, j)

            h0_origin               = h0_grid[j_step][i_origin]+frac*(h0_grid[j_step+1][i_origin]-h0_grid[j_step][i_origin])
            p0_origin               = p0_grid[j_step][i_origin]+frac*(p0_grid[j_step+1][i_origin]-p0_grid[j_step][i_origin])
            s_origin                = s_grid[j_step][i_origin]+frac*(s_grid[j_step+1][i_origin]-s_grid[j_step][i_origin])
       
            C_m_origin              = C_m_grid[j_step][i_origin]+frac*(C_m_grid[j_step+1][i_origin]-C_m_grid[j_step][i_origin])
            C_theta_origin          = C_theta_grid[j_step][i_origin]+frac*(C_theta_grid[j_step+1][i_origin]-C_theta_grid[j_step][i_origin])
            rC_theta_origin         = radius1*C_theta_origin
            V_theta_origin          = radius1*N_rotate-C_theta_origin
            C_sqrt_origin           = C_m_origin**2+C_theta_origin**2

            h0r_origin              = h0_origin-N_rotate*rC_theta_origin+(N_rotate*radius1)**2/2
            p0r_origin              = p0_origin*(h0r_origin/h0_origin)**(gamma/(gamma-1))
            h_origin                = h0_origin-C_sqrt_origin/2
            p_origin                = p0_origin*(h_origin/h0_origin)**(gamma/(gamma-1))

            h0r_grid[j][i]          = h0r_origin-N_rotate**2*(radius1**2-radius2**2)/2
            p0r_ideal_grid[j][i]    = p0r_origin*(h0r_grid[j][i]/h0r_origin)**(gamma/(gamma-1))
            p0r_grid[j][i]          = p0r_ideal_grid[j][i]-loss_grid[j][i]*(p0r_origin-p_origin)

            if i <= x_hub_outlet:
                rC_theta_grid[j][i] = radius2*C_theta_grid[j][i]
            else:
                rC_theta_grid[j][i] = rC_theta_origin

            C_theta_grid[j][i]      = rC_theta_grid[j][i]/radius2
            V_theta_grid[j][i]      = N_rotate*radius2-C_theta_grid[j][i]
            V_sqrt_grid[j][i]       = C_m_grid[j][i]**2+V_theta_grid[j][i]**2
            C_sqrt_grid[j][i]       = C_m_grid[j][i]**2+C_theta_grid[j][i]**2
            h_grid[j][i]            = h0r_grid[j][i]-V_sqrt_grid[j][i]/2
            h0_grid[j][i]           = h_grid[j][i]+C_sqrt_grid[j][i]/2
            p0_grid[j][i]           = p0r_grid[j][i]*(h0_grid[j][i]/h0r_grid[j][i])**(gamma/(gamma-1))
            T_grid[j][i]            = (h0_grid[j][i]-C_sqrt_grid[j][i])/(2*C_p)
            p_grid[j][i]            = p0_grid[j][i]*(h_grid[j][i]/h0_grid[j][i])**(gamma/(gamma-1))
            rho_grid[j][i]          = C_p*p_grid[j][i]/(R*h_grid[j][i])
            s_grid[j][i]            = s_origin+C_p*math.log(h0r_grid[j][i]/h0r_origin)-R*math.log(p0r_grid[j][i]/p0r_origin)

    #Checking convergence:
    psi_difference_grid          = [[0 for i in range(x_size)] for j in range(r_size)]
    vorticity_difference_grid    = [[0 for i in range(x_size)] for j in range(r_size)]
    rho_difference_grid          = [[0 for i in range(x_size)] for j in range(r_size)]
    for i in x_pos:
        for j in r_pos:
            psi_difference_grid[j][i]       = abs(psi_grid[j][i] - psi_old_grid[j][i])
            vorticity_difference_grid[j][i] = abs(vorticity_grid[j][i] - vorticity_old_grid[j][i])   
            rho_difference_grid[j][i]       = abs(rho_grid[j][i] - rho_old_grid[j][i])
    delta_psi       = max(max(psi_difference_grid))
    delta_vorticity = max(max(vorticity_difference_grid))
    delta_rho       = max(max(rho_difference_grid))
    print(delta_psi, delta_vorticity, delta_rho)

#Calculating the angles
for i in x_pos:
    for j in r_pos:
        V_theta_grid[j][i]      = N*r_cal(i, j)-C_theta_grid[j][i]
        alpha_grid[j][i]        = math.atan(C_theta_grid[j][i]/C_x_grid[j][i])
        beta_grid[j][i]         = math.atan(V_theta_grid[j][i]/C_m_grid[j][i])   
         
#Plotting
A = psi_grid
B = vorticity_grid
C = rho_grid
D = C_theta_grid
E = h0_grid
F = s_grid
G = beta_grid
H = alpha_grid

fig, ax = plt.subplots(8, 1, figsize=(17,9))
graph0 = ax[0].matshow(A, origin='lower', vmin=min(min(A)), vmax=max(max(A)))
graph1 = ax[1].matshow(B, origin='lower', vmin=min(min(B)), vmax=max(max(B)))
graph2 = ax[2].matshow(C, origin='lower', vmin=min(min(C)), vmax=max(max(C)))
graph3 = ax[3].matshow(D, origin='lower', vmin=min(min(D)), vmax=max(max(D)))
graph4 = ax[4].matshow(E, origin='lower', vmin=min(min(E)), vmax=max(max(E)))
graph5 = ax[5].matshow(F, origin='lower', vmin=min(min(F)), vmax=max(max(F)))
graph6 = ax[6].matshow(G, origin='lower', vmin=min(min(G)), vmax=max(max(G)))
graph7 = ax[7].matshow(H, origin='lower', vmin=min(min(H)), vmax=max(max(H)))

fig.colorbar(graph0, ax=ax[0])
fig.colorbar(graph1, ax=ax[1])
fig.colorbar(graph2, ax=ax[2])
fig.colorbar(graph3, ax=ax[3])
fig.colorbar(graph4, ax=ax[4])
fig.colorbar(graph5, ax=ax[5])
fig.colorbar(graph6, ax=ax[6])
fig.colorbar(graph7, ax=ax[7])

ax[0].set_title('Stream function')
ax[1].set_title('Vorticity')
ax[2].set_title('Density')
ax[3].set_title('C_theta')
ax[4].set_title('Total enthalpy')
ax[5].set_title('Entropy')
ax[6].set_title('Beta')
ax[7].set_title('Alpha')


ax[0].xaxis.set_ticks_position('bottom')
ax[1].xaxis.set_ticks_position('bottom')
ax[2].xaxis.set_ticks_position('bottom')
ax[3].xaxis.set_ticks_position('bottom')
ax[4].xaxis.set_ticks_position('bottom')
ax[5].xaxis.set_ticks_position('bottom')
ax[6].xaxis.set_ticks_position('bottom')
ax[7].xaxis.set_ticks_position('bottom')

plt.show()
