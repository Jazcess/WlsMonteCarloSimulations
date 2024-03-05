import matplotlib.pyplot as plt
import MonteCarlo_V1 as MC
import Sim_data as data
import Photon_Class as P
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import math

#updating position and velocity arrays
def up_arr(x, y, z, vx, vy, vz, x_pos, y_pos, z_pos, vx_arr, vy_arr, vz_arr):
    x_pos.append(x)
    y_pos.append(y)
    z_pos.append(z)
    vx_arr.append(vx)
    vy_arr.append(vy)
    vz_arr.append(vz)

# next position calculated 
def step(x, y, z, dt, vx, vy, vz, t):

    x  += vx*dt
    y  += vy*dt
    z  += vz*dt

    t += dt

    # Updated method to minimise steps taken
    # 1) compare whether al distance is larger or shorter than distance to next boundary
    # 2) calculate this difference if boundary is hit
    # 3) if satisfied, remove difference from total distance still needing to be travelled
    # 4) apply boundary conditions, reflection or refraction check
    # 5) re-do steps 1 and 2
    # 6) until full distance has been travelled

    return x, y, z, t

#setup for plots
def plot_setup(ax):
    plt.grid(color = 'white')
    ax.set_facecolor("lavender")
    ax.patch.set_alpha(0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

#plotting the simualtion output
def plot_sim(out_data, Run, dimension):
    # obtaining variables from outout data
    x_pos = np.asarray(out_data[Run]['X_positions_m'][1:-1].split(',')).astype(float)
    y_pos = np.asarray(out_data[Run]['Y_positions_m'][1:-1].split(',')).astype(float)
    z_pos = np.asarray(out_data[Run]['Z_positions_m'][1:-1].split(',')).astype(float)
    abs_len = out_data[Run]['Absorption_length_m']
    refract_coord = np.asarray(out_data[Run]['Refraction_coordinate_m'][2:-2].split(','))
    det = out_data[Run]['Detected']
    hit = out_data[Run]['Hit']
    time = out_data[Run]['Time_s']
    timedOut = out_data[Run]['Timed_out']
    label = f"Time: {round(time*1e7,5)} (ns)\nAbs Len: {round(abs_len,4)} (1/cm)"
    abs_i = len(x_pos)
    Qeff = out_data[Run]['Quantumn_efficiency'].astype(float)

    if out_data[Run]['Absorbed'].astype(bool) == True: #if absorbed, update plot accordingly
        abs_coords = np.asarray(out_data[Run]['Absorption_coordinate_m'][1:-1].split(',')).astype(float)
        iso_theta_ang = out_data[Run]['Isotropic_emmission_angle_theta_deg'].astype(float)
        iso_phi_ang = out_data[Run]['Isotropic_emmission_angle_phi_deg'].astype(float)
        label += f"\nAbs: {np.round(abs_coords,4)} (m)\ntheta {round(iso_theta_ang,3)}, phi {round(iso_phi_ang,3)} (deg)"
        abs_i = np.where((x_pos == abs_coords[0]) & (y_pos == abs_coords[1]) & (z_pos == abs_coords[2]))[0][0]
    if len(refract_coord) > 1: #if refracted update plot accordingly
        refract_coord = np.asarray(refract_coord.astype(float))
        label += f"\nRefr: {refract_coord} (m)"
    if hit == "Yes": #if it was hit
        hit_coords = np.asarray(out_data[Run]['Hit_coordinate_m'][1:-1].split(',')).astype(float)
        label += f"\nHit: {np.round(hit_coords,4)} (m)\nQE: {np.round(Qeff,3)*100}%"
        label += f"\nDet: {det}" #if it was detected
    if timedOut.astype(bool) == True: #if it timed-out
        label += "\nTimed out"

    abs_x = x_pos[abs_i:] #positions arrays after absorption
    abs_y = y_pos[abs_i:]
    abs_z = z_pos[abs_i:]

    if dimension == "3d": #3D layout

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter3D(x_pos,y_pos,z_pos,c='purple',label = "Before absorption")
        ax.scatter3D(abs_x,abs_y,abs_z,c = "royalblue",label = label)
        ax.set_ylim(0,MC.h)
        ax.set_xlim(0,MC.w)
        ax.set_zlim(0,MC.l)
        ax.set_xlabel("xdirection (m)")
        ax.set_ylabel("y-direction (m)")
        ax.set_zlabel("z-direction (m)")
        plt.title(f"Simulation Results - Run {Run}/{MC.nsteps}")
        plot_setup(ax)
        print(label)

        

    else: # 2D layout
        fig, ax = plt.subplots(2,1, layout="tight", figsize=(9,4))

        #setup for drawing PMT as ellipse on plot
        u=MC.w/2     #x-position of the center
        vy=0       #y-position of the center
        vz=MC.l/2    #z-position of the center
        a=MC.R     #radius on the x-axis
        b=0.04    #radius on the y-axis
        bz=MC.R #radius on the z-axis
        t = np.linspace(0, 2*np.pi, 100)

        # x-y view
        plt.subplot(2,1,1)
        plt.ylabel("y-direction (m)")
        plt.xlabel("x-direction(m)")
        plt.title(f"Simulation Results - Run {Run}/{MC.nruns}")
        ax[0].plot(x_pos,y_pos,c='purple',label = "Before absorption")
        ax[0].plot(abs_x,abs_y,c = "royalblue",label = label)
        ax[0].plot(u+a*np.cos(t),vy+b*np.sin(t),color='black')
        plt.ylim(0,MC.h)
        plt.xlim(0,MC.w)
        plot_setup(ax[0])
        ax[0].legend(loc='lower center',prop={'size': 6},bbox_to_anchor=(1.05, 1))

        # z-y view
        plt.subplot(2,1,2)
        ax[1].plot(z_pos,y_pos,c='purple',label = "Before absorption")
        ax[1].plot(abs_z,abs_y,c = "royalblue",label = label)
        ax[1].plot(u+a*np.cos(t),vy+b*np.sin(t),color='black')
        plt.ylim(0,MC.h)
        plt.xlim(0,MC.l)
        plt.xlabel("z-direction (m)")
        plt.ylabel("y-direction (m)")
        plot_setup(ax[1])

        # x-z view
        fig, ax = plt.subplots(1,1, layout="tight", figsize=(5,5))
        plt.subplot(1,1,1)
        ax.plot(x_pos,z_pos,c='purple',label = "Before absorption")
        ax.plot(abs_x,abs_z,c = "royalblue",label = label)
        ax.plot(u+a*np.cos(t),vz+bz*np.sin(t),color='black')
        plt.ylim(0,MC.l)
        plt.xlim(0,MC.w)
        plt.xlabel("x-direction (m)")
        plt.ylabel("z-direction (m)")
        plot_setup(ax)



