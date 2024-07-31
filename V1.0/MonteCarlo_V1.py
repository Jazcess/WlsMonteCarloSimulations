 ##########################################################################
 ## MonteCarlo.py, v-1.0,  10-11-2023
 ##
 ## Author(s): J. Stewart (SWGO)
 ##
 ## This is a Monte Carlo ray tracing script for WLS plates
 ##########################################################################

import scipy.constants as scp
import csv
import statistics as stats
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Sim_funcs_V1 as func
import Sim_data as data
import Photon_Class as P



#---------------------VARIABLES------------------

nruns=1000
dt = 1e-13 # time difference between each step in seconds
R = 0.0762/2 #pmt radius
n_wls = 1.58 # refractive index of PVT WLS plate
n_water = 1.33 # refractive index of water
w, l, h = 0.3, 0.3, 0.005 # width and height of WLS plate
ca = math.asin(n_water/n_wls) # critical angle between WLS and water
v_mag = (scp.c*n_wls) # velocity magnitude of cherenkov photon
theta = (280) * (math.pi/180) # angle of refraction into the WLS plate
phi = (90) * (math.pi/180) # angle of refraction into the WLS plate
vx_orig, vy_orig, vz_orig = v_mag*math.sin(theta)*math.cos(phi), v_mag*math.sin(theta)*math.sin(phi), v_mag*math.cos(theta) # velocity origin directions 
x_orig, y_orig, z_orig = w/4, h, l/4 # origin positions

CR_WL = np.arange(250,600, 10) # range of Cherenkov wavelengths
CR_E = ((scp.c * scp.h)/CR_WL) / (scp.e *1e-9) # converted to energies

header_list = ['Run', 'Initial_wl_nm', 'Initial_energy_eV_', 'Absorption_length_m', 'Attenuation_coefficient_cm',
               'Absorbed', 'Absorption_coordinate_m', 'Isotropic_emmission_angle_theta_deg', 'Isotropic_emmission_angle_phi_deg', 'Shifted_wl_nm',
               'Shifted_energy_eV', 'Refraction_coordinate_m', 'Hit', 'Hit_coordinate_m', 'Detected',
               'Num_reflections', 'Reflection_coordinates_m', 'Quantumn_efficiency', 'X_positions_m', 'Y_positions_m', 'Z_positions_m', 'Time_s', 'Timed_out']
    


#-----------------------------------------------



#check whether photon is at a boundary and react accordingly
def checks(x,y,z,vx,vy,vz,row,ref_coords,refract_coord, shifted_wl): 
    passed = True
    hit = False
    det = False

    if y < 0: # if y hits bottom boundary, reflect
        vy *= -1
        ref_coords.append([round(x,3),round(y,3),round(z,3)])
        

    if (y > h): #check if hit upper boundary
        theta_i = math.atan(vx/vy) #calculate incident angle
        phi_i = math.atan(vz/vy)
        if (abs(theta_i) or abs(phi_i)) >= ca: #if incident angle is bigger or equal to critical angle
            vy = -vy #then internally reflects
            ref_coords.append([round(x,3),round(y,3),round(z,3)])
        else: #else refracts
            passed = False
            refract_coord.append([round(x,3),round(y,3),round(z,3)])

    if x < 0 or x > w:
        vx *= -1
        ref_coords.append([round(x, 3), round(y, 3), round(z, 3)])

    if z < 0 or z > l:
        vz *= -1
        ref_coords.append([round(x, 3), round(y, 3), round(z, 3)])


    x_e = ((x-w/2)**2)/R**2
    y_e = (y**2/0.04**2)
    z_e = ((z-l/2)**2)/R**2

    if((x_e + y_e + z_e) < 1): #check if hit PMT , lies in ellipsoid (3D ellipse)
        hit = True
        diff = abs(data.QE_wls-shifted_wl)
        i = diff.argmin()
        eff = data.QE_probs[i] #finding the associated efficiency with the det wl

        pmtqe = np.random.choice([1,0],p=[eff,1-eff]) #randomly choosing the probability to see if det

        if pmtqe == 1: #check if PMT detected (QE)
            det = True
            row.extend([refract_coord , 'Yes' , [x,y,z] , 'Yes', len(ref_coords) , ref_coords , eff])
        else:
            passed = False
    
    return passed, vx, vy, vz, hit, det



#---------------------START--SIM----------------------
def sim(out_file):
    with open(out_file, 'w', newline='') as file:
        writer = csv.writer(file,  delimiter=';')
        writer.writerow(header_list)
        
        for n in range(nruns): #repeat for nruns amount of photons, but treat them all individually; seperate runs
            row = [str(n)]
            ref_coords = [] #list of reflection coords
            refract_coord = [] #list of refraction coords (doesnt need to be a list, as each run will only have MAX 1 refraction)
            t = 0 #total time elapsed
            timedOut = False

            init_wl = 380 # cherenkov wavelength (nm)
            init_E = ((scp.c * scp.h)/init_wl) / (scp.e *1e-9) #energy
            p1 = P.Photon([vx_orig,vy_orig,vz_orig], [x_orig,y_orig,z_orig], init_wl, 0.006)
            p1.al() #run absorption length fucntion from class
            shifted_wl = 0
            
            row.extend([init_wl , init_E , p1.absl , p1.mu])

            #updating position and velocity vectors
            x_pos, y_pos, z_pos, vx_arr, vy_arr, vz_arr = [], [], [], [], [], []
            func.up_arr(x_orig,y_orig,z_orig,vx_orig,vy_orig,vz_orig,x_pos,y_pos,z_pos,vx_arr,vy_arr,vz_arr)
            #setting first values to origins
            vx, vy, vz = vx_orig, vy_orig, vz_orig 
            x, y, z = x_orig, y_orig, z_orig
            absorbed = True #true unless photon fails checks, due to while loop
            dist = 0 #current distance travelled


            step_x = 0
            step_y = 0
            step_z = 0

            while (dist) < p1.absl: #keep going until distance is larger than abl
                x_old = x
                y_old = y
                z_old = z
                x, y, z, t = func.step(x, y, z, dt, vx, vy, vz, t) #calculate next step
                if t > 1e-8: #time-out if this runs for more than 10 ns
                    row.extend([absorbed, '', '', '', '', '', refract_coord , 'No' , '' , 'No', len(ref_coords), ref_coords , '0']) #write to file
                    timedOut = True
                    break
                func.up_arr(x,y,z,vx,vy,vz,x_pos,y_pos,z_pos,vx_arr,vy_arr,vz_arr) #update arrays
                step_x = abs(x-x_old) #calculate step between old and new x,y,z positions
                step_y = abs(y-y_old)
                step_z = abs(z-z_old)
                dist += math.sqrt(step_x**2 + step_y**2 + step_z**2) #calculate distance travelled
            
            
                passed, vx, vy, vz, hit, det = checks(x,y,z,vx,vy,vz,row,ref_coords,refract_coord,shifted_wl) #run through checks function

                if (det == True): #if detected before being absorbed break out of while loop
                    break

                if passed == False:
                    absorbed = False
                    if hit == True: #if it hit the PMT but wasn't detected
                        row.extend([absorbed, '', '', '', '', '', refract_coord , 'Yes' , [x,y,z] , 'No', len(ref_coords), ref_coords , '0'])
                    else: 
                        row.extend([absorbed, '', '', '', '', '', refract_coord , 'No' , '' , 'No', len(ref_coords), ref_coords , '0'])
                    break


            if absorbed == True:
                
                p1.swl(data.emms_cdf, data.emms_wls) #calculate the emitted shifted wavelength
                shifted_wl = p1.wl
                shifted_E = ((scp.c * scp.h)/shifted_wl) / (scp.e *1e-9)
                p1.energy = shifted_E

                v_mag = math.sqrt(vx**2 + vy**2 + vz**2) #original velocity magnitude
                theta_iso = random.uniform(0,2*math.pi) #new isotropic emmission angle theta
                phi_iso =  math.asin(random.uniform(-1,1)) #new isotropic emmission angle phi
                vy = v_mag * math.sin(theta_iso)*math.sin(phi_iso)
                vx = v_mag * math.sin(theta_iso)*math.cos(phi_iso)
                vz = v_mag * math.cos(theta_iso)

                row.extend([absorbed, [x,y,z] , theta_iso , phi_iso, p1.wl , p1.energy])


                while (((((x-w/2)**2)/R**2) + (y**2/0.04**2) + ((z-l/2)**2)/R**2) > 1): #while photon has NOT been detected; lies outside PMT
                    #Again this can be replaced to just consider collision points, rather than individual dt steps.
                    x, y, z, t = func.step(x, y, z, dt, vx, vy, vz, t) #calculate next step
                    if t > 1e-8: #time out if runs longer than 10ns
                        row.extend([refract_coord , 'No' , '' , 'No', len(ref_coords) , ref_coords , '0'])
                        timedOut = True
                        break
                    func.up_arr(x,y,z,vx,vy,vz,x_pos,y_pos,z_pos,vx_arr,vy_arr,vz_arr) #update arrays
                    passed, vx, vy, vz, hit, det = checks(x,y,z,vx,vy,vz,row,ref_coords,refract_coord,shifted_wl)
                    #peform boundary checks

                    if passed == False: #if fails do the following and break out of loop
                        if hit == True:
                            row.extend([refract_coord , 'Yes' , [x,y,z] , 'No', len(ref_coords) , ref_coords , '0'])
                        else:
                            row.extend([refract_coord , 'No' , '' , 'No', len(ref_coords) , ref_coords , '0'])
                        break

            row.extend([x_pos, y_pos, z_pos, t, timedOut])            
            writer.writerow(row) #update output file
