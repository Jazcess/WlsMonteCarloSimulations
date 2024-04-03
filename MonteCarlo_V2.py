 ##########################################################################
 ## MonteCarlo.py, v-2.0,  20-02-2024
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
import Sim_data as data
import Photon_Class as P


# Function for calculating the next collision point
def calculate_next_collision(position, velocity, boundaries, pmt_center, pmt_radii, wl, water):
    """
    Calculate the next collision point of a particle with boundaries and a photomultiplier tube (PMT).

    Parameters:
    - position (list or np.array): Current position of the particle.
    - velocity (list or np.array): Current velocity of the particle.
    - boundaries (np.array): Array representing the boundaries of the simulation space.
    - pmt_center (np.array): Center coordinates of the photomultiplier tube (PMT).
    - pmt_radii (np.array): Radii of the PMT along each axis.
    - wl (float): Wavelength associated with the particle.

    Returns:
    - collision_point (list): Coordinates of the collision point.
    - min_positive_time (float): Time taken to reach the collision point.
    - axis_hit (int): Axis of collision.
    - hit (bool): True if the particle hits the PMT.
    - hit_coords (list): Coordinates of the hit point on the PMT.
    - detected (bool): True if the hit is detected by the PMT.
    - pmtqe (int or None): Randomly chosen detection outcome based on the PMT's quantum efficiency.
    """
    # Initialize variables
    hit = False
    #hit_pmt, hit_wls
    hit_coords = [] #for both pmt and wls, as only one can be hit in water run
    detected = False
    tol = 1e-05
    pmtqe = None
    velocity = np.array(velocity)

    # Calculate time to reach each boundary
    t_boundaries = [(boundaries[i] - position[i]) / velocity[i] for i in range(3)]

    # if water: #if in the water, consider the WLS as an additional boundary
    #     t_boundaries.append((wls[i] - position[i]) / velocity[i] for i in range(3))


    # Calculate time to reach the PMT using quadratic intersection formula
    delta_position = position - pmt_center
    a = np.sum(velocity**2 / pmt_radii**2)
    b = 2 * np.sum(delta_position * velocity / pmt_radii**2)
    c = np.sum(delta_position**2 / pmt_radii**2) - 1
    discriminant = b**2 - 4*a*c

    if discriminant >= 0:
        tpmt1 = (-b + np.sqrt(discriminant)) / (2*a)
        tpmt2 = (-b - np.sqrt(discriminant)) / (2*a)
    else:
        tpmt1 = np.inf
        tpmt2 = np.inf

    t_boundaries.extend(np.array([[tpmt1,tpmt2]]))

    # Find the minimum positive time to reach a boundary or the PMT
    min_positive_time = np.min(np.array(t_boundaries)[np.array(t_boundaries) > 0])


    axis_hit, boundary = np.where(t_boundaries == min_positive_time)
    axis_hit = axis_hit[0]
    boundary = boundary[0]

    # Calculate the collision point based on the minimum positive time
    collision_point = [position[i] + velocity[i] * min_positive_time for i in range(3)]

    # Check if the collision point is inside the PMT
    if (np.sum((collision_point - pmt_center)**2 / pmt_radii**2) <= 1 + tol):
        hit = True
        hit_coords = collision_point

        # Check if PMT hit
        #print("\nPMT Hit!")
        
        # Find the associated efficiency with the detector wavelength
        diff = abs(data.QE_wls-wl)
        i = diff.argmin()
        pmtqe = data.QE_probs[i]

        # Randomly choose whether the detector detects the hit based on quantumn efficiency
        detected = True if np.random.choice([1, 0], p=[pmtqe, 1-pmtqe]) == 1 else False
        #if water and collision point is inside wls plate, register coords and break simulation
    else:
        collision_point[axis_hit] = boundaries[axis_hit, boundary]  # Set the axis hit to be exactly at the boundary


    return collision_point, min_positive_time, axis_hit, hit, hit_coords, detected, pmtqe #hit_pmt, hit_wls




def start_sim(pos, velocity, init_wl, satl, latl, boundaries, pmt_center, pmt_radii, ca, water):
    """
    Simulates the movement of a photon in a medium, considering absorption, reflection, and refraction.

    Parameters:
        pos (list): Initial position of the photon [x, y, z].
        velocity (list): Initial velocity vector of the photon [vx, vy, vz].
        init_wl (float): Initial wavelength of the photon.
        atl (float): Absorption length in the medium.
        boundaries (numpy.ndarray): Boundaries of the simulation space.
        pmt_center (list): Coordinates of the photomultiplier tube (PMT) center [x, y, z].
        pmt_radii (list): Radii of the PMT [x_radius, y_radius, z_radius].
        ca (float): Critical angle for reflection or refraction.

    Returns:
        tuple: A tuple containing various simulation results and parameters.

    Comments:
        - Initializes variables to track photon position, time, distance, and simulation state.
        - Creates a photon object using the Photon class, initializing its properties.
        - Enters a simulation loop until the photon is absorbed, reflects, refracts, or times out.
        - Calculates collision points, distances, and updates position and velocity accordingly.
        - Handles absorption by adjusting the position, updating the wavelength, and setting isotropic emission angles.
        - Handles reflection when the photon hits a boundary and the incident angle is greater than or equal to the critical angle.
        - Handles refraction when the incident angle is less than the critical angle.
        - Tracks the total distance traveled, reflection count, and simulation timeout.
        - Returns a tuple with relevant simulation data.

    Note: The comments provide a detailed explanation of the key steps and decisions made during the simulation.
    """

    xpos, ypos, zpos = [pos[0]], [pos[1]], [pos[2]] # Position arrays
    t = 0.0 # Total time elapsed
    total_dist = 0 #total distance travelled
    abs_coords = [] #photon ansorption coordinates
    refract_coords = [] #photon refraction coordinates
    reflects = 0 #num of reflections / bounces
    theta_iso, phi_iso = 0, 0 #isotropic emmission angles
    hit = False
    absorbed = False
    absorbed_again = False
    timeout = False
    abs_dist = 0
    sabsl, labsl = 0, 0

    p1 = P.Photon(velocity, pos, init_wl, satl) # Call photon class
    p1.al() # Run absorption length function from class
    sabsl = p1.absl
    

    while not hit: #while not hit_pmt and not hit_wls (hit_wls == True if not water)
        # Calculate collision point
        coll_point, coll_time, axis_hit, hit, hit_coords, detected, pmtqe = calculate_next_collision(pos, velocity, 
                                                                                            boundaries, pmt_center, pmt_radii, p1.wl, water)
        if coll_time is None:  # error handling
            #print("No collision time found. Exiting simulation.")
            break

        # Calculate distance travelled between origin point and collision boundary
        dist = math.sqrt((pos[0] - coll_point[0])**2 + (pos[1] - coll_point[1])**2 + (pos[2] - coll_point[2])**2)
        total_dist += dist
        abs_dist +=dist
        t += coll_time


        # Check if total distance exceeds absorption length
        if abs_dist >= p1.absl and (not absorbed or not absorbed_again):
            #print("\nAbsorbed!")
            if absorbed:
                absorbed_again = True
            absorbed = True


            overshoot = abs_dist - p1.absl  # Calculate overshoot distance
            total_dist -= overshoot  # Adjust total distance
            #print("Total distance travelled:", total_dist, "m")
            [overX,overY,overZ] = [overshoot * velocity[i]/np.linalg.norm(velocity) for i in range(3)]
            overxyz = [overX,overY,overZ]
            # Update position to the point of absorption
            pos = [coll_point[i] - overxyz[i]  for i in range(3)]
            abs_coords.append(pos)

            # Append positions to individual axis arrays
            xpos.append(pos[0])
            ypos.append(pos[1])
            zpos.append(pos[2])
            #print("Absorption Point:", abs_coords)

            if absorbed_again: #fully absorbed at this point
                if hit == True:
                    hit = False
                    hit_coords = []
                    detected = False
                break

            #only do wavelength shifting and emmission with wls plate, not water
            #dont wls again on second absorption
            elif not water or not absorbed_again: 
                # Perform wavelength shift calculation
                p1.swl(data.emms_cdf, data.emms_wls) # Calculate the emitted shifted wavelength

                # Update velocity for isotropic emission
                theta_iso = random.uniform(0, 2*math.pi) # New isotropic emission angle theta
                phi_iso =  math.asin(random.uniform(-1, 1)) # New isotropic emission angle phi
                vy = np.linalg.norm(velocity) * math.sin(theta_iso) * math.sin(phi_iso)
                vx = np.linalg.norm(velocity) * math.sin(theta_iso) * math.cos(phi_iso)
                vz = np.linalg.norm(velocity) * math.cos(theta_iso)
                velocity = [vx, vy, vz]


            #Setup for secondary absorption
            p1.mu = 1/latl #longer atl metres
            p1.al() #new longer absorption length
            abs_dist = 0 #restart distance counter
            labsl = p1.absl
        else:
            # Update position
            pos = coll_point
            # Append positions to individual axis arrays
            xpos.append(pos[0])
            ypos.append(pos[1])
            zpos.append(pos[2])

            if not water and pos[1] == boundaries[1,1]: #if upper boundary is hit
                #print("\nupper boundary hit")
                theta_i = math.atan(velocity[0]/velocity[1]) #calculate incident angles
                phi_i = math.atan(velocity[2]/velocity[1])
                if (abs(theta_i) or abs(phi_i)) >= ca: #if incident angle is bigger or equal to critical angle
                    velocity[axis_hit] *= -1
                else: #else refracts
                    #print("refracted")
                    refract_coords = pos
                    break
            elif not hit: #just reflects off boundary
                velocity[axis_hit] *= -1
                reflects += 1
                #print("\nAxis hit:", coord[axis_hit], "=", coll_point[axis_hit], "m")
            

            # print("Time between collisions:", round(coll_time*1e7,5), "ns")
            # print("Coordinates at the point of collision (x, y, z):", coll_point, "m")
            # print("Velocity", velocity, "m/s")
            # print("Distance travelled:", dist, "m")
            # print("Total distance travelled:", total_dist, "m")

        # Check for simulation timeout
        # if t > 1e-8:
        #     timeout = True
        #     #print("Timeout!")
        #     break


    #print("Total distance travelled:", total_dist, "m")
    #print("Total time:", round(t*1e7,5), "ns")

    return (sabsl, labsl, abs_coords, theta_iso, phi_iso, p1.wl, refract_coords, hit_coords, detected, 
            reflects, pmtqe, xpos, ypos, zpos, t, timeout, total_dist)