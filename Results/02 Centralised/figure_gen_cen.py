'''SFH house - case study for VIC & NRW - Post-processing'''
# AUTHOR: DBPR
#### file - database handling and housekeeping
import scienceplots
import os
from pathlib import Path
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import matplotlib as mpl
import numpy as np

#####
## Colors and plot style
#####

# plt.style.use(['science']) # Turn off if scienceplot matplotlib style package is not desired
# mpl.rc('text', usetex=True)


linestyle = ['-', '--', ':', '-.',(0, (3, 5, 1, 5, 1, 5))]
colors = ['#0d49fb', '#e6091c', '#26eb47', '#8936df', '#fec32d', '#25d7fd', '#FF00FF','#0d49fb', '#e6091c', '#26eb47', '#8936df', '#fec32d', '#25d7fd', '#FF00FF', #13 x 3
'#0d49fb', '#e6091c', '#26eb47', '#8936df', '#fec32d', '#25d7fd', '#FF00FF','#0d49fb', '#e6091c', '#26eb47', '#8936df', '#fec32d', '#25d7fd', '#FF00FF',
'#0d49fb', '#e6091c', '#26eb47', '#8936df', '#fec32d', '#25d7fd', '#FF00FF','#0d49fb', '#e6091c', '#26eb47', '#8936df', '#fec32d', '#25d7fd', '#FF00FF',
] # SciencePlots colors
markers = ['o','^','s','*','D','P','X']
# colors = None
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) # Using SciencPlots defaultstyle
month_year_formatter = mpl.dates.DateFormatter('%b') #Date format
monthly_locator = mpl.dates.MonthLocator()

#nature requires 5-7 pt
SMALLER_SIZE = 7
SMALL_SIZE = 7
MEDIUM_SIZE = 7
BIGGER_SIZE = 7

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# axis
## x axis
plt.rc('xtick', **{ 'direction' : 'in',
                    'major.size' : 3,
                    'major.width' : 0.5,
                    'minor.size' : 1.5,
                    'minor.width' : 0.5,
                    'minor.visible' : True,
                    'top' : 'True'})
## y axis
plt.rc('ytick', **{ 'direction' : 'in',
                    'major.size' : 3,
                    'major.width' : 0.5,
                    'minor.size' : 1.5,
                    'minor.width' : 0.5,
                    'minor.visible' : True,
                    'right' : True})

# line widths
plt.rc('axes', **{'linewidth': 0.5})
plt.rc('grid', **{'linewidth': 0.5})
plt.rc('lines', **{'linewidth': 1.0})

#legend
plt.rc('legend', **{'frameon': False})

#savefig
mpl.rc('savefig',**{'bbox': 'tight',
                    'pad_inches':0.05})

# size
mm = 1/25.4 ## mm to inches 
# plt.rcParams['figure.constrained_layout.use'] = True
def figsize(width = 'small',height = 60):
    if type(height) == int:
        if width == 'small':
            return (90*mm,height*mm)
        elif width == 'medium':
            return (140*mm,height*mm)
        elif width == 'large':
            return (180*mm,height*mm)
        elif type(width) == int:
            return (width*mm,height*mm)
        else:
            ValueError('input is width,height. width must be specified as "small", "medium","large" or an int.')
    else:
        ValueError('input is width,height. Height must be an int value')

def draw_text(ax,string, fontsize = MEDIUM_SIZE, loc = 'upper left'):
    """
    Draw two text-boxes, anchored by different corners to the upper-left
    corner of the figure.
    """
    from matplotlib.offsetbox import AnchoredText

    if type(string) == str:
        pass
    else:
        ValueError('Text box requires a string as an input.')

    at = AnchoredText(string,
                      loc = loc, prop=dict(size=fontsize), frameon=False,
                      )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

def K2C(temperature):
    """
    Convert temperature from Kelvin to Celsius.
    """
    return temperature - 273.15

#########
#### Results import
#########

#Path handling
script_dir = Path(__file__) ### script directory, including file
parent_path = Path(__file__).parents[0]

base_folder = 'Results' ## 1st paper base folder 
for p in script_dir.parents:
    if p.name == base_folder:
        base_path = Path(p)
    
base_run_path = base_path / '02 Centralised' #base run Paths
results_base_path = base_run_path / '_results' # stored results

os.chdir(str(parent_path))

###### Database dict
keys = ['NRW_NoIns', 'VIC_NoIns', 'NRW_S3', 'VIC_S3']
db_dict = dict.fromkeys(keys)

##### Opening already generated db
for key in keys:
    dirfile = [f for f in results_base_path.glob(f'*{key}.pk')]
    with open(f'{dirfile[0]}', 'rb') as f: # last generated pickle
        db_dict[key] = pickle.load(f)

######
### Commodities Prices Data
######

###### commodities costs
elec_cons_NRW_ori = round(32.79/100,4)
elec_ind_NRW_ori = round(18.33/100,4)
elec_fac_NRW = round(elec_ind_NRW_ori/elec_cons_NRW_ori,4)
gas_cons_NRW = round(8.06/100,4)
gas_ind_NRW = round(5.03/100,4)

elec_cons_VIC_ori = round(33.55/1.5/100,4)
elec_ind_VIC_ori = round(27.09/1.5/100,4)
elec_fac_VIC = round(elec_ind_VIC_ori/elec_cons_VIC_ori,4)
gas_cons_VIC = round(10.24/1.5/100,4)
gas_ind_VIC = round(7.70/1.5/100,4)

elec_cons = {'NRW_NoIns': elec_cons_NRW_ori, 'VIC_NoIns': elec_cons_VIC_ori,'NRW_S3': elec_cons_NRW_ori, 'VIC_S3': elec_cons_VIC_ori}
elec_fac = {'NRW': elec_fac_NRW, 'VIC': elec_fac_VIC}
elec_ind = {'NRW': elec_ind_NRW_ori, 'VIC': elec_ind_VIC_ori}
gas_cons = {'NRW': gas_cons_NRW, 'VIC': gas_cons_VIC}
gas_ind = {'NRW': gas_ind_NRW, 'VIC': gas_ind_VIC}

### SFH LCOE & TAC - elec price = [0.1, 0.2, 0.3, 0.4, 0.5, base cost]
x_SFH = np.arange(-5000, 150000, 1000) # only for plotting/coloring purposes
SFH_NRW_LCOE = [72.87455876557871, 118.17048251691664, 163.4664062682543, 168.3000122546595, 168.4452107587107, 168.19532416435635]
SFH_NRW_TAC = [1127.92725885627, 1829.0019271579479, 2530.0765954596213, 2604.88947999694, 2607.136812309322, 2603.2691538813424]
SFH_VIC_LCOE = [65.03206581590275, 106.78868165739262, 148.5452975719377, 161.3423741098446, 162.58609806912528, 116.67108073987839]
SFH_VIC_TAC = [1273.574943, 2091.328139, 2909.081337, 3159.696719, 3184.053559, 2284.863062]

elec_vect_NRW = [0.1, 0.2, 0.3, 0.4, 0.5, elec_cons_NRW_ori]
elec_vect_VIC = [0.1, 0.2, 0.3, 0.4, 0.5, elec_cons_VIC_ori]
SFH_NRW_dict = dict(zip(elec_vect_NRW, SFH_NRW_LCOE))
SFH_VIC_dict = dict(zip(elec_vect_VIC, SFH_VIC_LCOE))

######
### Vectors and string list of parameters
######

Tin_vect = [x + 273.15 for x in [5,10,20,30,40,50]] 
length_vect = [50,100,500,1000,2500,5000,7500,10000]
n_cons_vect = [50,100,500,1000,5000,10000,20000]
elec_vect = [0.1,0.2,0.3,0.4,0.5] ### not including base case, this is for use in the last figure
length_str = [str(x) for x in length_vect]
n_cons_str = [str(x) for x in n_cons_vect]
Tin_str = [str(x) for x in Tin_vect]

###### Additional Postprocessing

for (k,db) in db_dict.items():
    db['el_price'] =  db['el_price'].apply(lambda x: round(x,4)) # rounding el_price for accurate referencing
    db['norm_Q'] = db.apply(lambda row: np.round(row['tot_dem']/row['length'],4), axis = 1) # Total demand divided by length

    if 'NRW' in k:
        db['SFH_LCOE'] = db.apply(lambda row: SFH_NRW_dict[row['el_price']], axis = 1)
    elif 'VIC' in k:
        db['SFH_LCOE'] = db.apply(lambda row: SFH_VIC_dict[row['el_price']], axis = 1)    
    db['comp_LCOE'] = db.apply(lambda row: (row['SFH_LCOE'] - row['upd_LCOE'])/row['SFH_LCOE'], axis = 1) # Total demand divided by length  

###### Fig list used for plotting and saving figures
fig_list = []

##DB Filtering - Base case - Tsource = 283.15 K, elec price = base
db_base_plt = {key: None for key in keys} 
db_base = {key: None for key in keys} 
for (k,db) in db_dict.items():
    db_res = db[(db['el_price'] == elec_cons[k]) & (db['T_source'] == (10+273.15))].copy() #ensure that no double slicing is occuring
    db_res.reset_index(drop = True, inplace=True)
    db_base[k] = db_res
    db_base_plt[k] = db_res.pivot_table(index='length', columns='n_cons', values='upd_LCOE')

db_base_ele = {key: None for key in keys} 
for (k,db) in db_dict.items():
    db_res = db[(db['el_price'] == elec_cons[k])].copy() #ensure that no double slicing is occuring
    db_res.reset_index(drop = True, inplace=True)
    db_base_ele[k] = db_res

db_full = {key: None for key in keys} ## All Variables 
for (k,db) in db_dict.items():
    db_res = db.copy() #ensure that no double slicing is occuring
    db_res.reset_index(drop = True, inplace=True)
    db_full[k] = db_res

#########
##### FIG 2-Supplementary
#########

### Plotting and format
fig4,ax4 = plt.subplots(nrows=1,ncols=2, tight_layout = True, figsize = figsize('medium',60))
ax4[0].fill_between(x_SFH,0, SFH_NRW_LCOE[-1], color = 'gray', alpha = 0.6)
ax4[1].fill_between(x_SFH,0, SFH_VIC_LCOE[-1], color = 'gray', alpha = 0.6)
db_base_plt['NRW_NoIns'].plot(marker = 'o', markersize = 4, linestyle = '-', ax=ax4[0])
db_base_plt['VIC_NoIns'].plot(marker = 'o', markersize = 4, linestyle = '-', ax=ax4[1])
ax4[0].legend(bbox_to_anchor=(1.0, 1.0))
ax4[0].legend(title = 'No. of consumers', ncol=2, columnspacing=0.6)
ax4[1].legend(bbox_to_anchor=(1.0, 1.0))
ax4[1].legend(title = 'No. of consumers', ncol=2, columnspacing=0.6)
ax4[0].set(
       ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
       xlabel="Transmission length [m]")
ax4[1].set( yticklabels = [], 
       xlabel="Transmission length [m]")         
ax4[0].set_ylim([0, 500])
ax4[1].set_ylim([0, 500])
ax4[0].set_xlim([-500, 10500])
ax4[1].set_xlim([-500, 10500])
draw_text(ax4[0], '(c) - NRW', MEDIUM_SIZE, 'upper right')
draw_text(ax4[1],'(d) - VIC', MEDIUM_SIZE, 'upper right')
draw_text(ax4[0], 'Centralised', SMALL_SIZE, 'lower right')
draw_text(ax4[1], 'Centralised', SMALL_SIZE, 'lower right')
fig_list.append(fig4)

#########
### FIG 1
#########

# markers_dict_n = {k:d for (k,d) in zip(n_cons_vect,markers[0:len(n_cons_str)])}
color_dict_n = {k:d for (k,d) in zip(n_cons_vect,colors[0:len(n_cons_str)])}

fig5,ax5 = plt.subplots(1,2,tight_layout = True, figsize = figsize('medium',60))

i = 0
ax_list = []
for db in [db_base.get(k) for k in ['NRW_NoIns','VIC_NoIns']]:
    ax = plt.subplot(1,2,i + 1)
    db_plot_fig5 = db.copy()
    ax.set_xscale('log')
    ax.set_ylim([0, 500])
    ax.set_xlim([5E-2, 1E4])
    if i == 0:
        ax.fill_between(x_SFH,-10, SFH_NRW_LCOE[-1], color = 'gray', alpha = 0.6)
        ax.set(
            ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
            xlabel="$Q/L \: [\mathrm{MWh} \: \mathrm{m}^{-1}]$")
        draw_text(ax, '(c) - NRW', SMALL_SIZE, 'upper left')
        draw_text(ax, 'Centralised', SMALL_SIZE, 'lower right')

    else:
        ax.fill_between(x_SFH,-10, SFH_VIC_LCOE[-1], color = 'gray', alpha = 0.6)
        ax.set( yticklabels = [], 
            xlabel="$Q/L \: [\mathrm{MWh} \: \mathrm{m}^{-1}]$")        
        draw_text(ax,'(d) - VIC', SMALL_SIZE, 'upper left')
        draw_text(ax, 'Centralised', SMALL_SIZE, 'lower right')
    for n_cons,d in db_plot_fig5.groupby('n_cons'):
        ax.scatter(d['norm_Q'],d['upd_LCOE'], c = color_dict_n[n_cons], s = 10, label = str(n_cons))
    i =+ 1
    ax.legend(title = 'No. of \n consumers', ncol=2, columnspacing=0.6)
fig_list.append(fig5)



#########
### FIG 2
#########

color_dict_T = {k:d for (k,d) in zip(Tin_vect,colors[0:len(Tin_vect)])}


#### One plot
fig6,ax6 = plt.subplots(tight_layout = True, figsize = figsize('medium',60), sharex=True, sharey=True)
ax6.remove() #removing overlapping axes
ax_list = []
for i,db in enumerate([db_base_ele.get(k) for k in ['NRW_NoIns','VIC_NoIns']]):
    if ax in locals():
        ax.remove()
    ax = plt.subplot(1,2,i + 1)
    db_plot_fig6 = db.copy()
    text_unit = '$^\circ\mathrm{{C}}$'

    if i == 0:
        ax.fill_between(x_SFH,-10, SFH_NRW_LCOE[-1], color = 'gray', alpha = 0.6)
        ax.set(
            ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
            xlabel="$Q/L \: [\mathrm{MWh} \: \mathrm{m}^{-1}]$")
        draw_text(ax, '(c) - NRW', SMALL_SIZE, 'upper center')
        draw_text(ax, 'Centralised', SMALL_SIZE, 'lower right')
    else:
        ax.fill_between(x_SFH,-10, SFH_VIC_LCOE[-1], color = 'gray', alpha = 0.6)
        ax.set( yticklabels = [], 
        xlabel="$Q/L \: [\mathrm{MWh} \: \mathrm{m}^{-1}]$")        
        draw_text(ax,'(d) - VIC', SMALL_SIZE, 'upper center')
        draw_text(ax, 'Centralised', SMALL_SIZE, 'lower right')
    ax.set_ylim([0, 500])
    ax.set_xlim([5E-2, 1E4])
    ax.set_xscale('log')
    for T,d in db_plot_fig6.groupby('T_source'):
        T_var = np.round(K2C(T),0)
        ax.scatter(d['norm_Q'],d['upd_LCOE'], s = 5, c = color_dict_T[T], label = f'{T_var}' + text_unit)
    ax.legend(title = '$T_{\mathrm{source}}$', labelspacing = 0.4)
fig_list.append(fig6)


# #######
# # Futher processing - TEMP Dependance
# #######
color_dict_L = {k:d for (k,d) in zip([100,1000,5000,10000],colors[0:4])}

db_Tsource_L = {key:None for key in keys}
db_Tsource_L_pvt = {key:None for key in keys}
db_Tsource_n = {key:None for key in keys}
db_Tsource_n_pvt = {key:None for key in keys}

for k,db in db_base_ele.items():
    db['T_source'] = db['T_source'].apply(lambda x: K2C(x)) # To Celsius
    db_holder_L = {}
    db_holder_n = {}
    db_holder_L_pvt = {}
    db_holder_n_pvt = {}   

    for L,name in zip(length_vect,length_str):
        db_holder_L[name] = db[db['length'] == L].copy()
        db_holder_L_pvt[name] =  db_holder_L[name].pivot_table(index='T_source', columns='n_cons', values='upd_LCOE')
    for n,name in zip(n_cons_vect,n_cons_str):
        db_holder_n[name] = db[(db['n_cons'] == n) & (db['length'].isin([100,1000,5000,10000]))].copy()
        db_holder_n_pvt[name] =  db_holder_n[name].pivot_table(index='T_source', columns='length', values='upd_LCOE')    

    db_Tsource_L[k] = db_holder_L
    db_Tsource_L_pvt[k] = db_holder_L_pvt
    db_Tsource_n[k] = db_holder_n
    db_Tsource_n_pvt[k] = db_holder_n_pvt

#########
##### FIG 3 (a) and (b)
#########

##### 
# N = 100,1000

f7,ax7 = plt.subplots(nrows=2,ncols=2, tight_layout = True, figsize=figsize('medium',90))
db_Tsource_n_pvt['NRW_NoIns']['100'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax7[0,0], color = color_dict_L)
db_Tsource_n_pvt['NRW_S3']['100'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax7[0,0], color = color_dict_L)
db_Tsource_n_pvt['VIC_NoIns']['100'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax7[0,1], color = color_dict_L)
db_Tsource_n_pvt['VIC_S3']['100'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax7[0,1], color = color_dict_L)
db_Tsource_n_pvt['NRW_NoIns']['1000'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax7[1,0], color = color_dict_L)
db_Tsource_n_pvt['NRW_S3']['1000'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax7[1,0], color = color_dict_L)
db_Tsource_n_pvt['VIC_NoIns']['1000'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax7[1,1], color = color_dict_L)
db_Tsource_n_pvt['VIC_S3']['1000'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax7[1,1], color = color_dict_L)
ax7[0,0].fill_between(x_SFH,0, SFH_NRW_LCOE[-1], color = 'gray', alpha = 0.6)
ax7[0,1].fill_between(x_SFH,0, SFH_VIC_LCOE[-1], color = 'gray', alpha = 0.6)
ax7[1,0].fill_between(x_SFH,0, SFH_NRW_LCOE[-1], color = 'gray', alpha = 0.6)
ax7[1,1].fill_between(x_SFH,0, SFH_VIC_LCOE[-1], color = 'gray', alpha = 0.6)
ax7[0,0].get_legend().remove()
ax7[0,1].get_legend().remove()
ax7[1,0].get_legend().remove()
ax7[1,1].get_legend().remove()


from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=colors[i]) for i in range(4)] ## Custom Legend
patches.reverse() #showing colors at they appear at the plot
myHandle = [Line2D([], [], marker='o', markerfacecolor='None', markeredgecolor='k', markersize=10, linestyle='-', color = 'k'),
          Line2D([], [], marker='^', markerfacecolor='None', markeredgecolor='k', markersize=10, linestyle='--', color = 'k')] ##Create custom handles for 2nd legend
myHandle.reverse() #showing symbols at they appear at the plot

l1 = ax7[0,1].legend(handles=patches, labels = [str(x) for x in [10000,5000,1000,100]],bbox_to_anchor=(1.5,1.0), title='Transmission\n length [m]') ##Add 2nd legend
l2 = ax7[0,1].legend(handles=myHandle, labels = ['Ins S3','No Ins'],bbox_to_anchor=(1.5,0.3), title='Insulation') ##Add 2nd legend
ax7[0,1].add_artist(l1) # 2nd legend will erases the first, so need to add it

ax7[0,0].set(
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
        xlabel=None)
ax7[0,1].set(yticklabels = [],
xlabel = None)

ax7[1,0].set(
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
       xlabel='$T_{\mathrm{{source}}} [^\circ\mathrm{{C}}]$')
ax7[1,1].set(yticklabels = [],
       xlabel='$T_{\mathrm{{source}}} [^\circ\mathrm{{C}}]$')

ax7[0,0].set_ylim([0, 500])
ax7[0,1].set_ylim([0, 500])
ax7[0,0].set_xlim([0, 55])
ax7[0,1].set_xlim([0, 55])
ax7[1,0].set_ylim([0, 300])
ax7[1,1].set_ylim([0, 300])
ax7[1,0].set_xlim([0, 55])
ax7[1,1].set_xlim([0, 55])
draw_text(ax7[0,0],'(a) - NRW', SMALL_SIZE, 'upper center')
draw_text(ax7[0,0], 'No. Cons = 100', SMALL_SIZE, 'lower left')
draw_text(ax7[0,1],'(b) - VIC', SMALL_SIZE, 'upper center')
draw_text(ax7[0,1], 'No. Cons = 100', SMALL_SIZE, 'lower left')
draw_text(ax7[1,0],'(c) - NRW', SMALL_SIZE, 'upper center')
draw_text(ax7[1,0], 'No. Cons = 1000', SMALL_SIZE, 'lower left')
draw_text(ax7[1,1],'(d) - VIC', SMALL_SIZE, 'upper center')
draw_text(ax7[1,1], 'No. Cons = 1000', SMALL_SIZE, 'lower left')
ax7[0,0].xaxis.set_major_locator(plt.MultipleLocator(10))
ax7[0,1].xaxis.set_major_locator(plt.MultipleLocator(10))
ax7[1,0].xaxis.set_major_locator(plt.MultipleLocator(10))
ax7[1,1].xaxis.set_major_locator(plt.MultipleLocator(10))
fig_list.append(f7)


#######
### FIG 3 (c) and (d)
#######

####
# Further processing - Electricity Costs
####

color_dict_el_NRW = {k:d for (k,d) in zip(elec_vect_NRW,colors[0:len(elec_vect_NRW)])}
color_dict_el_VIC = {k:d for (k,d) in zip(elec_vect_VIC,colors[0:len(elec_vect_VIC)])}

##DB Filtering - Base case - Tsource = 283.15 K, elec price = base
# db_ele = {key: None for key in keys} 
db_ele_pvt_len = {key: None for key in keys} 
db_ele_pvt_ele = {key: None for key in keys} 
for (k,db) in db_dict.items():
    db_holder_len = {}
    db_holder_ele = {}
    db_holder_pvt_len = {}
    db_holder_pvt_ele = {}
    for n,name in zip(n_cons_vect,n_cons_str):
        db_holder_len[name] = db[(db['T_source'] == (10+273.15)) & (db['n_cons'] == n) & (~db['el_price'].isin([0.2,0.4]))].copy() #ensure that no double slicing is occuring
        db_holder_ele[name] = db[(db['T_source'] == (10+273.15)) & (db['n_cons'] == n) & (db['length'].isin([100,1000,5000,10000]))].copy() #ensure that no double slicing is occuring
        db_holder_pvt_len[name] = db_holder_len[name].pivot_table(index='length', columns='el_price', values='upd_LCOE')
        db_holder_pvt_ele[name] = db_holder_ele[name].pivot_table(index='el_price', columns='length', values='upd_LCOE')
    # db_ele[k] = db_holder
    db_ele_pvt_len[k] = db_holder_pvt_len
    db_ele_pvt_ele[k] = db_holder_pvt_ele

el_price_color = [0.1, 0.3, 0.4, 0.5,0.7]
LCOE_NRW_color = [72.87455876557871, 163.4664062682543,168.3000122546595,168.4452107587107,168.4452107587107]
LCOE_VIC_color = [65.03206581590275, 148.5452975719377, 161.3423741098446, 162.58609806912528, 162.58609806912528]

f11,ax11 = plt.subplots(nrows=2,ncols=2, tight_layout = True, figsize=figsize('medium',90))
db_ele_pvt_ele['NRW_NoIns']['100'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax11[0,0], color = color_dict_L)
db_ele_pvt_ele['NRW_S3']['100'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax11[0,0], color = color_dict_L)
db_ele_pvt_ele['VIC_NoIns']['100'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax11[0,1], color = color_dict_L)
db_ele_pvt_ele['VIC_S3']['100'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax11[0,1], color = color_dict_L)
db_ele_pvt_ele['NRW_NoIns']['1000'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax11[1,0], color = color_dict_L)
db_ele_pvt_ele['NRW_S3']['1000'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax11[1,0], color = color_dict_L)
db_ele_pvt_ele['VIC_NoIns']['1000'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax11[1,1], color = color_dict_L)
db_ele_pvt_ele['VIC_S3']['1000'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax11[1,1], color = color_dict_L)
ax11[0,0].fill_between(el_price_color,0, LCOE_NRW_color, color = 'gray', alpha = 0.6)
ax11[0,1].fill_between(el_price_color,0, LCOE_VIC_color, color = 'gray', alpha = 0.6)
ax11[1,0].fill_between(el_price_color,0, LCOE_NRW_color, color = 'gray', alpha = 0.6)
ax11[1,1].fill_between(el_price_color,0, LCOE_VIC_color, color = 'gray', alpha = 0.6)
ax11[0,0].get_legend().remove()
ax11[0,1].get_legend().remove()
ax11[1,0].get_legend().remove()
ax11[1,1].get_legend().remove()

patches = [mpatches.Patch(color=colors[i]) for i in range(4)] ## Custom Legend
patches.reverse()
myHandle = [Line2D([], [], marker='o', markerfacecolor='None', markeredgecolor='k', markersize=10, linestyle='-', color = 'k'),
          Line2D([], [], marker='^', markerfacecolor='None', markeredgecolor='k', markersize=10, linestyle='--', color = 'k')] ##Create custom handles for 2nd legend
myHandle.reverse()

l1 = ax11[0,1].legend(handles=patches, labels = [str(x) for x in [10000,5000,1000,100]],bbox_to_anchor=(1.5,1.0), title='Transmission\n length [m]') ##Add 2nd legend
l2 = ax11[0,1].legend(handles=myHandle, labels = ['Ins S3','No Ins'],bbox_to_anchor=(1.5,0.3), title='Insulation') ##Add 2nd legend
ax11[0,1].add_artist(l1) # 2nd legend will erases the first, so need to add it

ax11[0,0].set(
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
        xlabel=None)
ax11[0,1].set(yticklabels = [],
                xlabel = None)

ax11[1,0].set(
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
       xlabel='Electricity Price $[\mathrm{EUR} \: \mathrm{kWh}^{-1}]$')
ax11[1,1].set(yticklabels = [],
       xlabel='Electricity Price $[\mathrm{EUR} \: \mathrm{kWh}^{-1}]$')

ax11[0,0].set_ylim([0, 300])
ax11[0,1].set_ylim([0, 300])
ax11[0,0].set_xlim([0.05, 0.55])
ax11[0,1].set_xlim([0.05, 0.55])
ax11[1,0].set_ylim([0, 300])
ax11[1,1].set_ylim([0, 300])
ax11[1,0].set_xlim([0.05, 0.55])
ax11[1,1].set_xlim([0.05, 0.55])
draw_text(ax11[0,0],'(a) - NRW', SMALL_SIZE, 'upper right')
draw_text(ax11[0,0], 'No. Cons = 100', SMALL_SIZE, 'upper left')
draw_text(ax11[0,1],'(b) - VIC', SMALL_SIZE, 'upper right')
draw_text(ax11[0,1], 'No. Cons = 100', SMALL_SIZE, 'upper left')
draw_text(ax11[1,0],'(c) - NRW', SMALL_SIZE, 'upper right')
draw_text(ax11[1,0], 'No. Cons = 1000', SMALL_SIZE, 'upper left')
draw_text(ax11[1,1],'(d) - VIC', SMALL_SIZE, 'upper right')
draw_text(ax11[1,1], 'No. Cons = 1000', SMALL_SIZE, 'upper left')
fig_list.append(f11)
#########
#Figure 5
#########
fig15,ax15 = plt.subplots(tight_layout = True, figsize = figsize('large',120))
ax15.remove() #removing overlapping axes
sorted_elec = elec_vect.copy()
sorted_elec.sort()

import palettable
import matplotlib as mpl
cm = mpl.colors.ListedColormap(palettable.scientific.diverging.Berlin_5.mpl_colors)
offset = lambda p: mpl.transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)

##### Functions for data categorization based on norm_Q

def Q_color_eval_HEX(Q):
    if Q < 1:
        return colors[-1]
    elif Q < 5:
        return colors[0]
    elif Q < 10:
        return colors[1]
    elif Q < 50:
        return colors[2]
    else:
        return colors[3]

def Q_color_eval(Q):
    if Q < 1:
        return 0.5
    # elif Q < 5:
    #     return 1.5
    elif Q < 5:
        return 1.5
    elif Q < 10:
        return 2.5
    elif Q < 50:
        return 3.5
    else:
        return 4.5

def Q_zorder(Q):
    if Q < 1:
        return 1
    # elif Q < 5:
    #     return 1.5
    elif Q < 5:
        return 2
    elif Q < 10:
        return 3
    elif Q < 50:
        return 4
    else:
        return 5
    
def Q_offset(Q):
    if Q < 1:
        return -8.0
    elif Q < 5:
        return -4.0
    elif Q < 10:
        return 0.0
    elif Q < 50:
        return 4.0
    else:
        return 8.0



for j,(db_n,db_s) in enumerate(zip([db_full.get(k) for k in ['VIC_NoIns','NRW_NoIns']],[db_full.get(k) for k in ['VIC_S3','NRW_S3']])):
    Q_array = db_n['norm_Q'].values
    Q_list = np.unique(np.round(Q_array,4)).tolist()
    color_dict_Q = {k:d for (k,d) in zip(Q_list,[Q_color_eval_HEX(q) for q in Q_list])}
    zorder_dict_Q = {k:d for (k,d) in zip(Q_list,[Q_zorder(q) for q in Q_list])}
    offset_dict_Q = {k:d for (k,d) in zip(Q_list,[Q_offset(q) for q in Q_list])}
    fit_param = { }
    counter_f = 0
    def sep_same_Q(grp):
        global counter_f
        grp['group'] = counter_f
        counter_f += 1
        return grp

    def assign_val(grp): # assign group based on norm_Q, requires a global counter
        global counter_f
        if grp['count'].iloc[0] != 36: ## in cases where more than 1 set fit the same Q value, they are identified first
            grp = grp.groupby(['len_con'],group_keys = False).apply(sep_same_Q) # then an additional separator is implemented based on the length, consumer combination.
    
        else:
            grp['group'] = counter_f
            counter_f += 1
        return grp
    for db in [db_n,db_s]:
        db['len_con'] = db.apply(lambda row: str(row['length']) + ',' + str(row['n_cons']), axis = 1)
        db['count'] = db.groupby(['norm_Q'],group_keys = False)['norm_Q'].transform('count')
        db_hold = db.groupby(['norm_Q'], group_keys = False).apply(assign_val)
        db['group'] = db_hold['group']
        db.sort_values(by=['group'], inplace=True)
        counter_f = 0
    minmax_dict = dict.fromkeys(sorted_elec)
    short_T_vect = [x +273.15 for x in [10, 30, 50]]
    for i,T in enumerate(short_T_vect):
        if ax in locals():
            ax.remove()
        minmax_dict = dict.fromkeys(sorted_elec)
        ax = plt.subplot(2,3,(j)*3 + (i+1))
        for (g_n,d_n),(g_s,d_s) in zip(db_n.groupby(db_n['group']),db_s.groupby(db_s['group'])):
            Q = np.unique(d_n['norm_Q'].to_numpy())
            if Q < 1.0:
                pass
            else:
                d_fit_n = d_n[(d_n['T_source'] == T) & (d_n['el_price'].isin(elec_vect))].copy()
                d_fit_n.sort_values(['el_price'], inplace = True)
                d_fit_n.reset_index(inplace = True)
                d_fit_s = d_s[(d_s['T_source'] == T) & (d_s['el_price'].isin(elec_vect))].copy()
                d_fit_s.sort_values(['el_price'], inplace = True)
                d_fit_s.reset_index(inplace = True)
                col_s_name = {}
                for name in d_fit_s.columns:
                    col_s_name[name] = name + '_s'
                d_fit_s.rename(columns = col_s_name, inplace=True)
                ### concat the dataframes and use row
                ## reorder by el_price and ignore index

                conc_d = pd.concat([d_fit_n,d_fit_s], axis = 1)
                conc_d['min_LCOE'] = conc_d.apply(lambda row: row['comp_LCOE'] if row['comp_LCOE']<row['comp_LCOE_s'] else row['comp_LCOE_s'], axis = 1)
                conc_d['marker'] = conc_d.apply(lambda row: 'o' if row['comp_LCOE']<row['comp_LCOE_s'] else '^', axis = 1) ## check marker
                X_pd = conc_d['el_price'].values
                LCOE_Data = conc_d['min_LCOE'].values
                c_pd = []
                for i,c in enumerate(X_pd):
                    c_pd.append(color_dict_Q[Q[0]]) ## accessing the single value array and applying color function 
                m_pd = conc_d['marker'].values
                trans = plt.gca().transData
                for _m, _c, _x, _y in zip(m_pd, c_pd, X_pd, LCOE_Data):
                    plt.scatter(_x,_y, s=20, marker=_m, edgecolors = _c, zorder = zorder_dict_Q[Q[0]], ## no c-map
                                                alpha = 0.6, facecolors = 'none', transform = trans + offset(offset_dict_Q[Q[0]]))
                for x,y in zip(X_pd, LCOE_Data): ## overall band
                    if minmax_dict[x] == None: # initialising dict values, originally empty
                        minmax_dict[x] = [y,y] # [min,max]
                    if y < minmax_dict[x][0]: #min
                        minmax_dict[x][0] = y
                    if y > minmax_dict[x][1]: #max
                        minmax_dict[x][1] = y
        np_minmax = np.array([v for v in minmax_dict.values()])
        plt.axhline(y=0.0, color='black', linestyle='--')
        x_long = np.linspace(sorted_elec[0],sorted_elec[-1],50000)
        y_long_min = np.interp(x_long,sorted_elec,np_minmax[:,0])
        y_long_max = np.interp(x_long,sorted_elec,np_minmax[:,1])
        ax.fill_between(x_long, y_long_min, y_long_max ,where=y_long_max<= 0, color = 'darkkhaki', alpha = 0.6, interpolate = True, ec = 'none')       # if LCOE
        ax.fill_between(x_long, y_long_min, 0 ,where=((y_long_max>=0) & (y_long_min<=0)), color = 'darkkhaki', alpha = 0.6, interpolate = True, ec = 'none')       # if LCOE
        ax.fill_between(x_long, y_long_min, y_long_max,where=((y_long_max>=0) & (y_long_min>=0)), color = 'gray', alpha = 0.6, interpolate = True, ec = 'none')       # if LCOE       
        ax.fill_between(x_long,0, y_long_max,where=((y_long_max>=0) & (y_long_min<=0)), color = 'gray', alpha = 0.6, interpolate = True, ec = 'none')       # if LCOE
        ax.set_ylim([-0.45,0.45])
        ax.set_xlim([0.05, 0.55])
        ax.set_xticks(elec_vect, labels = [str(x) for x in elec_vect])
        ax.set_yticks([-0.4, 0, 0.4], labels = [str(x) for x in [-0.4, 0, 0.4]])

    fig15.supxlabel('Electricity Price $[\mathrm{EUR} \: \mathrm{kWh}^{-1}]$')
    fig15.supylabel(r'Normalised LCOE $\: [-]$')
def remove_inner_ticklabels(fig):
    for ax in fig.axes:
        try:
            ax.label_outer()
        except:
            pass
remove_inner_ticklabels(fig15)

fig_list.append(fig15)

## SAVE PLOT
for i,fig in enumerate(fig_list):
    fig.savefig(f'Fig_{i}_Cen.pdf', dpi=1000)

