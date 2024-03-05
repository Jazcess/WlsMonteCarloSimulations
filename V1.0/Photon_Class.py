import random
import math
import numpy as np
import Sim_data as data
import scipy.constants as scp

# Class for photon used in Monte Carlo simulation

class Photon:
    def __init__(self, velocity, pos, initwl, atl):
        """
        Initializes a Photon object with specified attributes.

        Parameters:
        - velocity (list): Initial velocity vector [vx, vy, vz].
        - pos (list): Initial position vector [x, y, z].
        - initwl (float): Initial wavelength of the photon (nm).
        - atl (float): Attenuation length for the medium.

        Attributes:
        - energy (float): Energy of the photon (eV).
        - pos (list): Current position vector [x, y, z].
        - velocity (list): Current velocity vector [vx, vy, vz].
        - wl (float): Current wavelength of the photon (nm).
        - mu (float): Inverse of the attenuation length.
        - absl (float): Absorption length for the photon.
        - atl (float): Attenuation length for the medium.
        """
        # Setting up photon variables
        self.energy = ((scp.c * scp.h) / initwl) / (scp.e * 1e-9)  # Energy in eV
        self.pos = pos
        self.velocity = velocity
        self.wl = initwl
        self.mu = 1 / atl
        self.absl = 0
        self.atl = atl

    def al(self):
        """
        Calculates the absorption length for the photon.

        Generates a random number between 0 and 1 and uses it to determine
        the absorption length based on the exponential distribution.

        Parameters: None

        Returns: None
        """
        rand = random.random()  # Random number generation between 0 and 1
        self.absl = math.log(1 - rand) / (-self.mu)

    def swl(self, emms_cdf, emms_wls):
        """
        Calculates the shifted wavelength of the photon after absorption.

        Generates a random number between 0 and 1 and compares it with the
        cumulative distribution function (CDF) of emitted wavelengths. Updates
        the photon's wavelength based on the matched CDF value.

        Parameters:
        - emms_cdf (numpy array): Cumulative distribution function of emitted wavelengths.
        - emms_wls (numpy array): Array of emitted wavelengths.

        Returns: None
        """
        rand = random.random()  # Random number generation between 0 and 1
        diff = abs(emms_cdf - rand)  # Differences between rand num and CDF spectrum
        val = min(diff)  # Smallest difference in the differences array
        i = np.where(diff == val)  # Index of the smallest difference
        self.wl = emms_wls[i][0]  # Updated wavelength based on CDF input
        self.energy = ((scp.c * scp.h) / self.wl) / (scp.e * 1e-9)  # Update energy
