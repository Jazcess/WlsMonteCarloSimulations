#%matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
import math


nruns = 8000

thicknesses = np.arange(5,26,4) #mm
sizes = np.arange(20,61,5) #cm

satls = np.arange(2*1e-3,50*1e-3,5*1e-3) #m
latls = np.arange(2,5,0.5) #m

size = 60
thickness = 10
satl = 8*1e-3  # Short attenuation length
latl = 3.5  # Long attenuation length

rx, ry, rz = 0.0762/2, 40e-3, 0.0762/2  # PMT radii
area_half_pmt = (math.pi * rx**2)/2 #circle
vol_half_pmt = (4 * math.pi * (((rx*ry)**1.6 + (rx*rz)**1.6 + (ry*rz)**1.6)/3)**(1/1.6))/2

det_times = []
all_times = []
dist = []




for s in sizes:
# for la in latls:

    hit_eff = []
    det_av_time = []
    area_half_wls = ((s*1e-2)**2)/2
    area_total = area_half_wls - area_half_pmt
    #print(area_total)
    
    for th in thicknesses:
    # for sa in satls:
    
        vol_half_wls = ((th*1e-3) * (s*1e-2)**2)/2
        vol_total = vol_half_wls - vol_half_pmt
        #print(vol_total)

            
        #path = 'outputs/'
        #out_file = path + 'sim_output_' + str(int(thickness*1e+3)) + 'mm_' + str(int(size*1e+2)) + 'cm' + '.txt'
        path = 'outputs/thicknesses/seed(1)/' + str(th) + 'mm_' + str(s) + 'cm_'
        out_file = path + 'sim_output_' + str(satl) + '_m' + str(latl) + '_m.txt'
        #out_file = path + 'sim_output_' + str(satl) + '_m' + str(latl) + '_m.txt'
        out_data = np.genfromtxt(out_file, names=True, delimiter=';', dtype=None, encoding=None) #read data
        hits = np.where(out_data['Hit_coordinate_m'] != '[]')[0]
        det_times_ = out_data[hits]['Time_s']
        all_times_ = out_data['Time_s']
        det_times.append(det_times_)
        all_times.append(all_times_)
        det_av_time.append(np.mean(det_times_))
        hit_eff.append((len(hits)/nruns) * area_total)
        dist.append(out_data['Total_distance_m'])   

    plt.plot(thicknesses,hit_eff, label = 'Size: ' + str(int(s)) + 'cm')
    # plt.plot(satls,hit_eff, label = 'Long Attenuation Length: ' + str(la) + 'm')


plt.xlabel("Thicknesses (mm)")
# plt.xlabel("Short Attenuation Lengths (m)")
plt.ylabel("Hit Efficiency * area (m^2)")
# plt.ylabel("Hit Efficiency (%)")
# plt.title("WLS Plate Attenuation Lengths VS Hit Efficiency\nFor a 30 cm X 30cm X 10mm WLS Plate")
plt.title("WLS Plate Thicknesses and Sizes VS Hit Efficiency")
plt.legend(loc='center right', bbox_to_anchor=(1.57, 0.5))
plt.grid()
plt.show()
