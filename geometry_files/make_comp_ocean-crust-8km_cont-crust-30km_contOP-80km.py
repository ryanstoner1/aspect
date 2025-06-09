#!/usr/bin/env python3 

import sys
import numpy

ofile = "outputs/res2_1x_weakbox-20km_50dip_freesurf_basalt-ocean-crust-8km_cont-crust-8km_contOP-10km_sed-1km_serp-0km_rad100_melth2o.txt"

# box dimensions (i.e. "extent" in ASPECT input)
xmin=0;xmax=5800.e3;
ymin=0;ymax=1450.e3;

# number of cells in input geometry
xnum= 5800
ynum= 2900

# geometrical parameters, meters
x_gap = 500.e3; 
x_SP  = 3000.e3; 
depth_notch  = 50e3;
radius_outer = 245e3;
slab_dip = 50.;
y_basalt = 2.0e3;
y_basalt2 = 2e3
y_gabbro = 5e3
OPthick = 100.e3;
CCthick = 8.e3;
y_sed = 2.5e3; # because free surface gobbles up ~1.5km OC
y_sed2 = 1.0e3;
y_serp = 0.0e3
voflayer = 100e3

# empty array to store geometry
No_nodes= (xnum + 1) * (ynum + 1)
C=numpy.zeros([No_nodes,12],float)
 
ind=0

for j in range(ynum + 1): 
	for i in range(xnum + 1):

		x = xmin + i * ((xmax - xmin)/xnum)
		y = ymin + j * ((ymax - ymin)/ynum) 
  
		C[ind,0] = x
		C[ind,1] = y

		# sed along top of flat portion of SP
		if x > (x_gap) and x <= (x_gap + x_SP - 1.5*radius_outer) and y > (ymax - y_sed):
			C[ind,3]=1


		# crust along top of flat portion of SP
		if x > (x_gap) and x <= (x_gap + x_SP - 1.5*radius_outer) and y > (ymax - y_basalt - y_sed) and y <= (ymax - y_sed):
			C[ind,6]=1


		# gabbro along top of flat portion of SP
		if x > (x_gap) and x <= (x_gap + x_SP - 1.5*radius_outer) and y > (ymax - y_basalt - y_sed - y_gabbro) and y <= (ymax - y_sed - y_basalt):
			C[ind, 8] = 1


		# crust along top of flat portion of SP
		if x > (x_gap) and x <= (x_gap + x_SP - 1.5*radius_outer) and y > (ymax - y_basalt - y_sed - y_gabbro - y_serp) and y <= (ymax - y_sed - y_basalt - y_gabbro):
			C[ind, 5] = 1

		# sed along top of flat portion of SP
		if x > (x_gap + x_SP - 1.5*radius_outer) and x <= (x_gap + x_SP - radius_outer) and y > (ymax - y_sed):
			C[ind,3]=1

		# basalt along top of flat portion of SP
		if x > (x_gap + x_SP - 1.5*radius_outer) and x <= (x_gap + x_SP - radius_outer) and y > (ymax - y_basalt - y_sed) and y <= (ymax - y_sed):
			C[ind,6]=1

		# gabbro along top of flat portion of SP
		if x > (x_gap + x_SP - 1.5*radius_outer) and x <= (x_gap + x_SP - radius_outer) and y > (ymax - y_basalt - y_sed - y_gabbro) and y <= (ymax - y_sed - y_basalt):
			C[ind, 8] = 1

		# serpentinite along top of flat portion of SP
		if x > (x_gap) and x <= (x_gap + x_SP - 1.5*radius_outer) and y > (ymax - y_basalt - y_sed - y_gabbro - y_serp) and y <= (ymax - y_sed - y_basalt - y_gabbro):
			C[ind, 5] = 1

		# serpentinite along top of flat portion of SP
		if x > (x_gap + x_SP - 1.5*radius_outer) and x <= (x_gap + x_SP - radius_outer) and y > (ymax - y_basalt - y_sed - y_gabbro - y_serp) and y <= (ymax - y_sed - y_basalt - y_gabbro):
			C[ind, 5] = 1

		# if x > (x_gap) and x <= (x_gap + x_SP - radius_outer) and y > (ymax - y_serp):
		# 	C[ind,3]=1

		# curved portion of crust ("notch")
		elif x > (x_gap + x_SP - radius_outer) and x < (x_gap + x_SP):
			x1 = x_gap + x_SP - radius_outer;
			y1 = ymax - radius_outer;
			if ((x-x1)**2 + (y-y1)**2) < radius_outer**2 and y > (ymax - depth_notch): 
				angle=numpy.arctan((y-y1)/(x-x1));
				if ((x-x1)**2 + (y-y1)**2) > (radius_outer-y_basalt2-y_sed2)**2 and ((x-x1)**2 + (y-y1)**2) <= (radius_outer-y_sed2)**2:
					if angle > numpy.radians(90. - slab_dip):
						C[ind,2]=1
				if ((x-x1)**2 + (y-y1)**2) > (radius_outer - y_gabbro - y_basalt2 - y_sed2)**2 and ((x-x1)**2 + (y-y1)**2) <= (radius_outer - y_basalt2 - y_sed2)**2:
					if angle > numpy.radians(90. - slab_dip):
						C[ind, 9] = 1
				if ((x-x1)**2 + (y-y1)**2) > (radius_outer-y_sed2)**2:
					if angle > numpy.radians(90. - slab_dip):
						C[ind,3]=1
	

		# #overriding plate (OP) above notch
		# if x > (x_gap + x_SP - radius_outer) and x < (x_gap + x_SP):
		# 	x1 = x_gap + x_SP - radius_outer; 
		# 	y1 = ymax - radius_outer;
		# 	if ((x-x1)**2 + (y-y1)**2) >= (radius_outer+y_serp)**2 and y > (ymax - OPthick): 
		# 		C[ind,3]= 1

		# #rest of the OP
		# if  x > (xmax - x_gap) and x <= (xmax - x_gap + 50e3) and y > (ymax - OPthick): # x >= (x_gap + x_SP) and 
		# 	C[ind,3]= 1
				
		# Continental crust
		if x > (x_gap + x_SP - radius_outer) and x < (x_gap + x_SP):
			x1 = x_gap + x_SP - radius_outer; 
			y1 = ymax - radius_outer;
			if ((x-x1)**2 + (y-y1)**2) >= radius_outer**2 and y > (ymax - CCthick): 
				C[ind,4]= 1
				C[ind,3]= 0

		# # Lithosphere vof
		# if x > (x_gap + x_SP - radius_outer) and x < (x_gap + x_SP):
		# 	x1 = x_gap + x_SP - radius_outer
		# 	y1 = ymax - radius_outer
		# 	if ((x-x1)**2 + (y-y1)**2) >= radius_outer**2 and y <= (ymax - CCthick) and y>(ymax-CCthick-voflayer):
		# 		C[ind, 10] = 1
		# 		C[ind, 3] = 0

		# # sediment
		# x1 = x_gap + x_SP - radius_outer;
		# y1 = ymax - radius_outer;
		# if ((x-x1)**2 + (y-y1)**2) < (radius_outer+y_serp)**2  and x > (x_gap + x_SP - radius_outer) and x < (x_gap + x_SP): # and y > (ymax - depth_notch) and y<=(ymax - CCthick) 
		# 	angle=numpy.arctan((y-y1)/(x-x1));
		# 	if ((x-x1)**2 + (y-y1)**2) > (radius_outer)**2:
		# 		if angle > numpy.radians(90. - slab_dip):
		# 			C[ind,3]=1	

		# rest of the OP
		if  x >= (x_gap + x_SP) and x < (xmax - x_gap) and y > (ymax - CCthick): 
			C[ind,4]= 1
			C[ind,3]= 0

		# weak box left
		if  x <= (x_gap) and x > (x_gap - 30e3) and y > (ymax - 250e3): 
			C[ind,7]= 1
			C[ind,3]= 0

		# weak box right
		if  x >= (xmax - x_gap) and x < (xmax - x_gap + 30e3) and y > (ymax - 250e3): 
			C[ind,7]= 1
			C[ind,3]= 0

		if sum(C[ind,2:])>1:
			Warning("More than one composition assigned \n")
		if sum(C[ind,2:])<0:
			Warning("Less than one composition assigned \n")
			raise Exception('spam', 'eggs')

		ind=ind+1;
 

# write to file in ASPECT format
f= open(ofile,"w+")
f.write("# POINTS: %s %s\n" % (str(xnum+1),str(ynum+1)))
f.write("# Columns: x y composition1 composition2 composition3 composition4 composition5 composition6\n")
for k in range(0,ind):
	#f.write("%.6f %.6f %.2f %.2f %.2f%.2f\n" % (C[k,0],C[k,1],C[k,2],C[k,3],C[k,4],C[k,5]))
	# f.write("%.6f %.6f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % (
	# 	C[k, 0], C[k, 1], C[k, 2], C[k, 3], C[k, 4], C[k, 7], 0.02*C[k, 5], C[k, 6], 0, 0.0*C[k, 5], 0, 0, C[k, 8],C[k, 9]))

	f.write("%.6f %.6f %.6f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % (
		C[k, 0],  C[k, 1], 0.0, C[k, 8], C[k, 9], C[k, 2], C[k, 3], C[k, 4], C[k, 7], 0.02*C[k, 5], C[k, 6], 0, 0.0*C[k, 5], 0))
f.close() 

