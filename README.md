# Monte Carlo Simulation for WLS Plates

This repository contains scripts and code for a Monte Carlo simulation for ray tracing in Wavelength Shifting (WLS) plates. 
The simulation models the movement of photons in a medium, considering absorption, reflection, and refraction. 
It includes a Jupyter notebook for analyzing simulation output, a Photon class for representing photons, and scripts for processing simulation data.

## Contents

1. `MonteCarlo.py`: The main Python script containing the Monte Carlo simulation for WLS plates.
2. `Simulation_Analysis.ipynb`: Jupyter notebook for analyzing simulation output data.
3. `Photon_Class.py`: Python script defining the Photon class used in the simulation.
4. `Sim_data.py`: Python script containing simulation data, such as emission spectrum and quantum efficiency.
5. `data_processing.py`: Python script for reading and processing simulation data files.
6. `README.md`: This file providing an overview of the repository.

## Usage

1. **Monte Carlo Simulation**: Run the `MonteCarlo.py` script to perform the Monte Carlo simulation. Adjust parameters and inputs as needed.

2. **Simulation Analysis**: Open and run the Jupyter notebook `Simulation_Analysis.ipynb` to analyze and visualize simulation output.

3. **Photon Class**: The `Photon_Class.py` script contains the definition of the Photon class. Modify this class if additional functionalities are required.

4. **Simulation Data**: The `Sim_data.py` script contains data used in the simulation, such as emission spectrum and quantum efficiency. Update this script if you have new data.

5. **Data Processing**: The `data_processing.py` script reads and processes simulation data files. Adjust file paths or formats as necessary.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Seaborn (for visualization in the Jupyter notebook)

## File Descriptions

- `MonteCarlo.py`: Main simulation script.
- `Simulation_Analysis.ipynb`: Jupyter notebook for analysis.
- `Photon_Class.py`: Photon class definition.
- `Sim_data.py`: Simulation data (emission spectrum, quantum efficiency).
- `data_processing.py`: Script for reading and processing simulation data.

