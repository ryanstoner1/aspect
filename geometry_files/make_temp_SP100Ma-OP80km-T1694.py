#!/usr/bin/env python3 

import sys
import numpy as np
import scipy, scipy.special

ofile="outputs/temp_70flatdip_SP100Ma-OP30km-T1694.txt"

# box dimensions (i.e. "extent" in ASPECT input)
xmin=0;xmax=5800.e3;
ymin=0;ymax=1450.e3;

# number of cells in input geometry
xnum=5800
ynum=1450

# geometrical parameters, meters
x_gap = 500.e3;         # distance between plate edges and domain sides
x_SP  = 3000.e3;        # subducting plate length
depth_notch  = 200e3;   # initial slab depth
radius_outer = 245e3;   # initial slab radius of curvature
slab_dip = 70.;         # slab dip
thick_op = 30.e3       # overriding plate thickness
ridge_extent = 1000.e3; # length of subducting plate that is a "ridge"

# thermal parameters
Tmax = 1694.; Tmin = 273.;     # min/max temperatures, K
Tcutoff = 1694.0;               # if T > Tcutoff, T = Tmax
age_ma=100;                    # subucting plate age [Ma]
age=age_ma*1e6*365*24*60*60;   # plate age in secs
k = 1e-6                       # thermal diffusivity (for half-space cooling)

# empty array to store geometry
No_nodes= (xnum + 1) * (ynum + 1)
T=np.zeros([No_nodes,3],float)
 
ind=0
print("writting text file...")
for j in range(ynum + 1): 

	for i in range(xnum + 1):

		x = xmin + i * ((xmax - xmin)/xnum)
		y = ymin + j * ((ymax - ymin)/ynum) 
  
		T[ind,0] = x
		T[ind,1] = y
		T[ind,2] = Tmax

		# mini ridge at end of subd plate (SP)
		if x > (x_gap) and x <= (x_gap + ridge_extent):
			age_ridge = (x - x_gap) * (age/ridge_extent)
			erf_term=(ymax-y)/(2*np.sqrt(k*age_ridge))
			T[ind,2]='%.5f'  %   (Tmax - (Tmax - Tmin)*scipy.special.erfc(erf_term))
		# flat portion of SP
		elif x > (x_gap + ridge_extent) and x <= (x_gap + x_SP - radius_outer):
			erf_term=(ymax-y)/(2*np.sqrt(k*age))
			T[ind,2]='%.5f'  %   (Tmax - (Tmax - Tmin)*scipy.special.erfc(erf_term))
		# overiding plate (OP) - linear geotherm
		elif x >= (x_gap + x_SP) and x < (xmax - x_gap) and y > (ymax - thick_op): 
			T[ind,2]='%.5f'  %   (Tmin + (Tcutoff - Tmin)*((ymax-y)/thick_op))

		# curved portion of SP ("notch")
		if x > (x_gap + x_SP - radius_outer) and x < (x_gap + x_SP):
			x1 = x_gap + x_SP - radius_outer; 
			y1 = ymax - radius_outer;
			if ((x-x1)**2 + (y-y1)**2) < radius_outer**2 and y > (ymax - depth_notch): 
				# angle=np.arctan((y-y1)/(x-x1));
				# if angle > np.radians(90. - slab_dip):
				# 	# ynotch = radius_outer - np.sqrt((x-x1)**2 + (y-y1)**2)
				# 	# erf_term=(ynotch)/(2*np.sqrt(k*age))
				# 	erf_term=(ymax-y)/(2*np.sqrt(k*age))
				erf_term=(ymax-y)/(2*np.sqrt(k*age))
				T[ind,2]='%.5f'  %   (Tmax - (Tmax - Tmin)*scipy.special.erfc(erf_term))
			elif ((x-x1)**2 + (y-y1)**2) >= radius_outer**2 and y > (ymax - thick_op): 
				T[ind,2]='%.5f'  %   (Tmin + (Tcutoff - Tmin)*((ymax-y)/thick_op))

		if T[ind,2] > Tcutoff:
		   T[ind,2] = Tmax

		ind=ind+1;
 
# write to file in ASPECT format
f= open(ofile,"w+")
f.write("# POINTS: %s %s\n" % (str(xnum+1),str(ynum+1)))
f.write("# Columns: x y temperature\n")
for k in range(0,ind):
	f.write("%.4f %.4f %.4f\n" % (T[k,0],T[k,1],T[k,2]))
f.close() 
