import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import os

fonts_saved = [f.name for f in font_manager.fontManager.ttflist if "Times New Roman" in f.name]

plt.rcParams['font.family']="Times New Roman"
plt.rcParams['font.size']=10
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.linewidth'] = 1.5

# Add every font at the specified location
# font_dir = ['/home/rstoner/Desktop/MyriadPro']
# for font in font_manager.findSystemFonts(font_dir):
#     font_manager.fontManager.addfont(font)
#matplotlib.rcParams['font.family'] = 'MYRIADPRO'
#lgfont = {'fontname':'Merriweather-Bold'}

run_folder0 = "rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_01_cmu0_04_deserp_erase_res5_5_run22"
run_folder1 = "rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run18"
run_folder2 = "rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_03_cmu0_04_deserp_erase_res5_5_run23"
run_folder3 = "rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_04_cmu0_04_deserp_erase_res5_5_run24"
kin_files = [run_folder0+".txt",run_folder1+".txt",run_folder2+".txt",run_folder3+".txt",]

legends = ["pelite 6 wt%","pelite 4 wt%", "pelite 2 wt%","pelite 0 wt%"]
# kin_files = ["dd200_orig.txt","dd200.txt"]
# legends = ["no lookup\nMDD=200 km","MDD=200 km"]
colors = ["navy","skyblue","maroon","coral"]
f = np.loadtxt(os.getcwd()+"/kinematics/"+kin_files[0])
t = f[:,1]
vsp = f[:,2]
vt = f[:,3]
vc = f[:,4]
dip_deep = f[:,5]
dip_deep250 = f[:,6]
is_serpentinizable = f[:,10]

Tsurf = f[:,-5]
Tmoho = f[:,-4]
z500 = f[:,-3]
serptotal = f[:,-2]
serptotal120 = f[:,-1]

f_q = np.loadtxt(os.getcwd()+"/kinematics/"+run_folder1+"_freefluid.txt")
f_s = np.loadtxt(os.getcwd()+"/kinematics/"+run_folder1+"_serp.txt")

ymax=1450.e3
ymin_plot=870.e3; # 1125
grid_res=4.0e3; # 4e3
y_low =  np.linspace(ymax-ymin_plot,0,int((ymax-ymin_plot)/grid_res))

fig0_fluid = plt.figure()

y_cut = 1
T, Y = np.meshgrid(t[1:],y_low[:(-y_cut)])

T1, Y1 = np.meshgrid(t[:-1],y_low[:(-y_cut)])

# if (len(f_q)>0):
#     plt.contourf(T1,Y1/1e3,(f_q.T)[:(-y_cut),:],levels=50)
#     cbar1 = plt.colorbar(ticks=[
#                             -0.0450, -0.0300, -0.0150, 0, 0.0150])
#     cbar1.set_label("free fluid (area)",size=9.5)

#     plt.locator_params(axis='y', nbins=5)
#     plt.locator_params(axis='x', nbins=7)
#     plt.xlabel('time [Myr]')
#     plt.ylabel('depth [km]')
#     plt.savefig("plots/"+run_folder1+"/cohes-5MPa_f_allfluidbig_t1", bbox_inches='tight', format='png', dpi=600)

# if (len(f_s)>0):
#     fig0_serp = plt.figure(figsize=(7,3))

#     gs=GridSpec(1,3) 

#     print("plotting temp field...")
#     ax1=fig0_serp.add_subplot(gs[0,0])
#     ax1.plot(f_s.T[1:(-y_cut),1],Y[1:,1]/1e3,color="tomato",linewidth=2)
#     ax1.set_xlabel('$serpentinite$ $volume$')
#     ax1.set_ylabel('$depth$ [km]')
#     ax1.set_ylim((0,200))
#     ax1.set_xlim((0,0.3))
#     ax1.annotate(''.join(['t = ',str("%.1f" % (T[1,1])),' Myr']), xy=(0.1,0.93), xycoords='axes fraction',verticalalignment='center',horizontalalignment='left',fontsize=12,fontweight="bold",color='tomato')       

#     ax2=fig0_serp.add_subplot(gs[0,1])
#     ax2.plot(f_s.T[1:(-y_cut),len(Y[1,1:])//2],Y[1:,1]/1e3,color="darkgreen",linewidth=2)
#     ax2.set_xlabel('$serpentinite$ $volume$')
#     ax2.set_ylabel('$depth$ [km]')
#     ax2.set_ylim((0,200))
#     ax2.set_xlim((0,0.3))
#     ax2.annotate(''.join(['t = ',str("%.1f" % (T[1,len(Y[1,1:])//2])),' Myr']), xy=(0.1,0.93), xycoords='axes fraction',verticalalignment='center',horizontalalignment='left',fontsize=12,fontweight="bold",color='darkgreen')       
    
#     ax3=fig0_serp.add_subplot(gs[0,2])
#     ax3.plot(f_s.T[1:(-y_cut),-1],Y[1:,-1]/1e3,color="slateblue",linewidth=2)
#     ax3.set_xlabel('$serpentinite$ $volume$')
#     ax3.set_ylim((0,200))
#     ax3.set_xlim((0,0.3))
#     ax3.set_ylabel('$depth$ [km]')

#     ax3.annotate(''.join(['t = ',str("%.1f" % (T[1,-1])),' Myr']), xy=(0.1,0.93), xycoords='axes fraction',verticalalignment='center',horizontalalignment='left',fontsize=12,fontweight="bold",color='slateblue')       
            
#     #plt.contourf(T,Y/1e3,(f_s.T)[:(-y_cut),:],levels=50)
#     # cbar1 = plt.colorbar()
#     # cbar1.set_label("serpentinite (formed)",size=7.5)
#     fig0_serp.suptitle('cohesion 5 (MPa); OC 8 (km); with pelite', fontsize=10,fontweight="bold")
#     fig0_serp.tight_layout()

#     plt.savefig("plots/"+run_folder1+"/cohes-5MPa_f_serpbig_t1", format='png', dpi=600)

fig1 = plt.figure()
plt.plot(t,vt)
plt.xlabel('time (Myr)')
plt.ylabel('v$_{t}$ (cm/yr)')
plt.savefig("plots/"+run_folder1+"/cohes-5MPa_lookup_vt_t1", bbox_inches='tight', format='png', dpi=600)

fig1.clf()
fig2, axs = plt.subplots(nrows=1,ncols=3, figsize=(9, 3))

axs[0].set_title('subduction velocity', fontstyle='italic')
axs[0].plot(t[1:],vsp[1:])
axs[0].set_xlabel("time (Myr)")
axs[0].set_ylabel("v$_{sp}$ (cm/yr)")

axs[1].set_title('convergence velocity', fontstyle='italic')
axs[1].plot(t[1:],vc[1:])
axs[1].set_xlabel("time (Myr)")
axs[1].set_ylabel("v$_{c}$ (cm/yr)")

axs[2].set_title('trench velocity', fontstyle='italic')
axs[2].plot(t[1:],vt[1:])
axs[2].set_xlabel("time (Myr)")
axs[2].set_ylabel("v$_{t}$ (cm/yr)")

plt.tight_layout()
plt.savefig("plots/"+run_folder1+"/cohes-5MPa_1xserp_OC5_nolookup_kinematics_t1", bbox_inches='tight', format='png', dpi=600)

# comparison
fig3, axs = plt.subplots(nrows=1,ncols=3, figsize=(9, 3))
for kin_file,color,legend in zip(kin_files,colors,legends):
    f3 = np.loadtxt("kinematics/"+kin_file)
    print(f3)
    t = f3[:,1]
    vsp = f3[:,2]
    vt = f3[:,3]
    vc = f3[:,4]
    dd = f3[:,8]     

    axs[0].set_title('subduction velocity', fontstyle='italic')
    axs[0].plot(t[1:],vsp[1:],color,label=legend)
    axs[0].set_xlabel("time (Myr)")
    axs[0].set_ylabel("v$_{sp}$ (cm/yr)")

    axs[1].set_title('convergence velocity', fontstyle='italic')
    axs[1].plot(t[1:],vc[1:],color,label=legend)
    axs[1].set_xlabel("time (Myr)")
    axs[1].set_ylabel("v$_{c}$ (cm/yr)")

    axs[2].set_title('trench velocity', fontstyle='italic')
    axs[2].plot(t[1:],vt[1:],color,label=legend)
    axs[2].set_xlabel("time (Myr)")
    axs[2].set_ylabel("v$_{t}$ (cm/yr)")
    axs[0].legend(fontsize="small")
    plt.tight_layout()

plt.savefig("plots/"+run_folder1+"/comparison_kinematics_cohes-5MPa_1xserp_OC5_t1.png", bbox_inches='tight')

# comparison
fig4, axs = plt.subplots(nrows=1,ncols=3, figsize=(9, 3))
for kin_file,color,legend in zip(kin_files,colors,legends):
    f4 = np.loadtxt("kinematics/"+kin_file)
    print(f4)
    t = f4[:,1]
    vsp = f4[:,2]
    vt = f4[:,3]
    vc = f4[:,4]    
    dd = f4[:,9]

    axs[0].set_title('a)', fontstyle='italic')
    axs[0].plot(t,dd,color,label=legend)
    axs[0].set_xlabel("time [Myr]")
    axs[0].set_ylabel("depth [km]")

    axs[1].set_title('slab top and Moho at 100 km', fontstyle='italic')
    #axs[1].plot(t,Tsurf,colors[1],label=legends[0])
    axs[1].set_xlabel("time [Myr]")
    axs[1].set_ylabel("T [$^{\circ}$C]")
    axs[1].set_ylim(bottom=0)
    #axs[2].set_title('slab Moho at 100 km', fontstyle='italic')
    #axs[1].plot(t,Tmoho,colors[-1],label=legends[1])
    axs[1].set_xlabel("time [Myr]")
    axs[1].set_ylabel("T [$^{\circ}$C]")
    axs[1].legend(fontsize="small")

    axs[2].set_title('slab top and Moho at 100 km', fontstyle='italic')
    #axs[2].plot(t,Tsurf,colors[1],label=legends[0])
    axs[2].set_xlabel("time [Myr]")
    axs[2].set_ylabel("T [$^{\circ}$C]")
    axs[2].set_ylim(bottom=0)
    #axs[2].set_title('slab Moho at 100 km', fontstyle='italic')
    #axs[2].plot(t,Tmoho,colors[-1],label=legends[1])
    axs[2].set_xlabel("time [Myr]")
    axs[2].set_ylabel("T [$^{\circ}$C]")
    plt.tight_layout()

plt.savefig("plots/"+run_folder1+"/temperature_decoupling_stats_cohes-5MPa_1xserp_t1", bbox_inches='tight', format='png', dpi=600)

# serp comparison
fig4, axs = plt.subplots(nrows=1,ncols=3, figsize=(9, 3))
for kin_file,color,legend in zip(kin_files,colors,legends):
    f4 = np.loadtxt("kinematics/"+kin_file)
    print(f4)
    t = f4[:,1]
    vsp = f4[:,2]
    vt = f4[:,3]
    vc = f4[:,4]    

    axs[0].set_title('serpentinite fraction ($km^3/km$)', fontstyle='italic')
    #axs[0].plot(t,(100*serptotal/0.12)/is_serpentinizable,colors[1],label=legend)
    axs[0].set_xlabel("time [$Myr$]")
    axs[0].set_ylabel("serpentinization extent (%)")

    axs[1].set_title('serpentinite area <120 km ($km^2$)', fontstyle='italic')
    #axs[1].plot(t,serptotal120/1e6,colors[2],label=legend)
    axs[1].set_xlabel("time [$Myr$]")
    axs[1].set_ylabel("area [$km^{2}$]")

    axs[2].set_title('plotted wet perid. ($km^3/km$)', fontstyle='italic')
    #axs[2].plot(t,serptotal/1e6,colors[1],label=legend)
    axs[2].set_xlabel("time [$Myr$]")
    axs[2].set_ylabel("area [$km^{2}$]")



    # axs[1].set_title('slab top and Moho at 100 km', fontstyle='italic')
    # axs[1].plot(t,Tsurf,colors[1],label=legends[0])
    # axs[1].set_xlabel("time [Myr]")
    # axs[1].set_ylabel("T [$^{\circ}$C]")
    # axs[1].set_ylim(bottom=0)
    # #axs[2].set_title('slab Moho at 100 km', fontstyle='italic')
    # axs[1].plot(t,Tmoho,colors[-1],label=legends[1])
    # axs[1].set_xlabel("time [Myr]")
    # axs[1].set_ylabel("T [$^{\circ}$C]")
    # axs[1].legend(fontsize="small")

    # axs[2].set_title('slab top and Moho at 100 km', fontstyle='italic')
    # axs[2].plot(t,Tsurf,colors[1],label=legends[0])
    # axs[2].set_xlabel("time [Myr]")
    # axs[2].set_ylabel("T [$^{\circ}$C]")
    # axs[2].set_ylim(bottom=0)
    # #axs[2].set_title('slab Moho at 100 km', fontstyle='italic')
    # axs[2].plot(t,Tmoho,colors[-1],label=legends[1])
    # axs[2].set_xlabel("time [Myr]")
    # axs[2].set_ylabel("T [$^{\circ}$C]")
    plt.tight_layout()
plt.savefig("plots/"+run_folder1+"/temperature_serp_stats_cohes-5MPa_1xserp_OC5_t1", bbox_inches='tight', format='png', dpi=600)
# serp comparison
fig5, axs = plt.subplots(nrows=1,ncols=3, figsize=(9, 3))
for kin_file,color,legend in zip(kin_files,colors,legends):
    f4 = np.loadtxt("kinematics/"+kin_file)
    print(f4)
    t = f4[:,1]
    vsp = f4[:,2]
    vt = f4[:,3]
    vc = f4[:,4]    


    axs[0].set_title('serpentinite area <120 km ($km^2$)', fontstyle='italic')
    #axs[0].plot(t,serptotal120/1e6,colors[2],label=legend)
    axs[0].set_xlabel("time [$Myr$]")
    axs[0].set_ylabel("area [$km^{2}$]")

    axs[1].set_title('plotted wet perid. ($km^2$)', fontstyle='italic')
    #axs[1].plot(t,serptotal/1e6,colors[1],label=legend)
    axs[1].set_xlabel("time [$Myr$]")
    axs[1].set_ylabel("area [$km^{2}$]")

    # axs[1].set_title('slab top and Moho at 100 km', fontstyle='italic')
    # axs[1].plot(t,Tsurf,colors[1],label=legends[0])
    # axs[1].set_xlabel("time [Myr]")
    # axs[1].set_ylabel("T [$^{\circ}$C]")
    # axs[1].set_ylim(bottom=0)
    # #axs[2].set_title('slab Moho at 100 km', fontstyle='italic')
    # axs[1].plot(t,Tmoho,colors[-1],label=legends[1])
    # axs[1].set_xlabel("time [Myr]")
    # axs[1].set_ylabel("T [$^{\circ}$C]")
    # axs[1].legend(fontsize="small")

    axs[2].set_title(r'$dip_{125km}$', fontstyle='italic')
    #axs[2].plot(t,dip_deep,colors[3],label=legends[0])
    axs[2].set_xlabel("time [Myr]")
    axs[2].set_ylabel("dip [$^{\circ}$]")
    axs[2].set_ylim(bottom=0)
    #axs[2].set_title('slab Moho at 100 km', fontstyle='italic')
    # axs[2].plot(t,dip_deep400,colors[-1],label=legends[1])
    # axs[2].set_xlabel("time [Myr]")
    # axs[2].set_ylabel("dip [$^{\circ}$]")
    plt.tight_layout()
plt.savefig("plots/"+run_folder1+"/dip_serp_stats_cohes-5MPa_1xserp_1init_1pel_OC5_t1", bbox_inches='tight', format='png', dpi=600)

legends_vertical = ["v$_{c}$","v$_{t}$"]
legends_dip = ["dip$_{125 km}$","dip$_{250 km}$"]

fig6, axs = plt.subplots(nrows=3,ncols=1, figsize=(4, 9))
for kin_file,color,legend in zip(kin_files,colors,legends):
    f4 = np.loadtxt("kinematics/"+kin_file)
    print(f4)
    t = f4[:,1]
    vsp = f4[:,2]
    vt = f4[:,3]
    vc = f4[:,4] 
    dip125 = f4[:,5]
    dip250 = f4[:,6]
    dd = f4[:,9]

    axs[0].set_title('a) velocities')
    axs[0].plot(t[1:],vc[1:],colors[-1],label=legends_vertical[0],linewidth=2)
    axs[0].plot(t[1:],vt[1:],colors[0],label=legends_vertical[1],linewidth=2)
    
    axs[0].set_xlabel("time [Myr]",fontweight='bold')
    axs[0].set_ylabel("velocity [cm/yr]",fontweight='bold')
    axs[0].legend(fontsize="medium")

    axs[1].set_title('b) decoupling depth')
    axs[1].plot(t[1:],dd[1:],colors[2],linewidth=2)
    axs[1].set_xlabel("time [Myr]",fontweight='bold')
    axs[1].set_ylabel("depth [km]",fontweight='bold')
    axs[1].legend(fontsize="medium")

    axs[2].set_title('c) dip')
    axs[2].plot(t[dip125>0],dip125[dip125>0],colors[0],linewidth=2,label=legends_dip[0])
    axs[2].plot(t[dip250>0],dip250[dip250>0],colors[-1],linewidth=2,label=legends_dip[1])
    axs[2].set_xlabel("time [Myr]",fontweight='bold')
    axs[2].set_ylabel("depth [km]",fontweight='bold')
    axs[2].legend(fontsize="medium")
    # axs[1].set_title('decoupling depth', fontstyle='italic')
    # axs[1].plot(t,dd,color,label=legend)
    # axs[1].set_xlabel("time [Myr]")
    # axs[1].set_ylabel("depth [km]")

    # axs[1].set_title('slab top and Moho at 100 km', fontstyle='italic')
    # axs[1].plot(t,Tsurf,colors[1],label=legends[0])
    # axs[1].set_xlabel("time [Myr]")
    # axs[1].set_ylabel("T [$^{\circ}$C]")
    # axs[1].set_ylim(bottom=0)
    # #axs[2].set_title('slab Moho at 100 km', fontstyle='italic')
    # axs[1].plot(t,Tmoho,colors[-1],label=legends[1])
    # axs[1].set_xlabel("time [Myr]")
    # axs[1].set_ylabel("T [$^{\circ}$C]")
    # axs[1].legend(fontsize="small")

    # axs[2].set_title('slab top and Moho at 100 km', fontstyle='italic')
    # axs[2].plot(t,Tsurf,colors[1],label=legends[0])
    # axs[2].set_xlabel("time [Myr]")
    # axs[2].set_ylabel("T [$^{\circ}$C]")
    # axs[2].set_ylim(bottom=0)
    # #axs[2].set_title('slab Moho at 100 km', fontstyle='italic')
    # axs[2].plot(t,Tmoho,colors[-1],label=legends[1])
    # axs[2].set_xlabel("time [Myr]")
    # axs[2].set_ylabel("T [$^{\circ}$C]")
    plt.tight_layout()

save_loc = "plots/"+run_folder1+"/alldip_serp_stats_cohes-5MPa_1xserp_1init_1pel_OC5_t1.jpg"
plt.savefig(save_loc, bbox_inches='tight', format='jpg', dpi=400)
print("saved at: "+save_loc)