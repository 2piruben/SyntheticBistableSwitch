import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import pandas as pd
import copy

import seaborn as sns


sbcolorcycle=sns.color_palette("deep") # Setting Colour palettes
sbcolorcycledark=sns.color_palette("dark")
nodesize = 30 # size of stable node marker
talpha = 0.8 # alpha for trajectory
sns.set_style("ticks") # Set plot properties
sns.set_context("talk")
sns.despine()
newblack = sns.xkcd_rgb["charcoal"]
clw = 3.0 # line width 
#######

#talpha = 0.2 # alpha for filling

folder = "data_example/" # Replace with the right data folder


###################### DECLARATION OF .CSV FILES TO LOAD

if folder == 'data_example/':
	filenames =['DataGREENtime10 copyA.csv',
			'DataGREENtime10 copyB.csv',
			'DataGREENtime10 copyC.csv',
			'DataREDtime10 copyA.csv',
			'DataREDtime10 copyB.csv',
			'DataREDtime10 copyC.csv']
	timepoint = 10	


units_dict = {'AHL':'$\\mu$M','IPTG':'mM'}


################### LOADING DATA INTO A DATAFRAME

df = pd.DataFrame()
df_red = pd.DataFrame() # dataframe containing the gated red cells
df_green = pd.DataFrame() # dataframe containing the gated green cells

for file in filenames:

	tempdf = pd.read_csv(folder+file)

	columns = list(tempdf)
	for icol,column in enumerate(columns):
		if column[0:3] == '\xef\xbb\xbf':
			columns[icol] = column[3:]

	tempdf.columns = columns

	maxredmedian = 695.0 # Scale chosen to regularize fluorescence numbers comparable to 1, the scale does not affect the final data output
	rednorm_df = tempdf[tempdf['ahl'] == 10] # normalization factor for the red intensity taken from the red condition. 
	rednorm_df = rednorm_df[rednorm_df['iptg'] == 1]
	rednorm = rednorm_df['gateRED_medianRed'].iat[0]/maxredmedian
	rednormmode = rednorm_df['gateRED_modeRed'].iat[0]/maxredmedian

	print('rednorm', rednorm, 'file', file)

	maxgreenmedian = 772.0 # Scale chosen to regularize fluorescence numbers comparable to 1, the scale does not affect the final data output
	greennorm_df = tempdf[tempdf['ahl'] == 0] # normalization factor for the green intensity taken from the green condition
	greennorm_df = greennorm_df[greennorm_df['iptg'] == 0]
	greennorm = greennorm_df['gateGREEN_medianGreen'].iat[0]/maxgreenmedian
	greennormmode = greennorm_df['gateGREEN_modeGreen'].iat[0]/maxgreenmedian

	print('greennorm', greennorm, 'file', file)

	print(tempdf.keys())
	# gated RED

	tempdf = tempdf[tempdf['time'] == timepoint] ## time point to choose from

	df_red = tempdf[['ahl','iptg','gateRED_medianRed','gateRED_medianGreen','gateRED_modeRed','gateRED_modeGreen','gateRED_sdGreen','gateRED_sdRed','gateRED_percentageEvent']]
	df_red.columns = ['AHL','IPTG','RED_median','GREEN_median',
		           'RED_mode','GREEN_mode',
		           'GREEN_std','RED_std','percentage']
	df_red = df_red[df_red['percentage']>15] # at least 15% of population
	df_red['GREEN_var'] = df_red['GREEN_std']**2
	df_red['RED_var'] = df_red['RED_std']**2
	df_red['file'] = file
	df_red['gate'] = 'red'
	df_red['RED_median_normed'] = df_red['RED_median']/rednorm 
	df_red['RED_mode_normed'] = df_red['RED_mode']/rednormmode 
	df_red['GREEN_median_normed'] = df_red['GREEN_median'] # note that this is not normed since it is already in the baseline
	df = df.append(df_red)

	# gated GREEN
	df_green = tempdf[['ahl','iptg','gateGREEN_medianRed','gateGREEN_medianGreen','gateGREEN_modeRed','gateGREEN_modeGreen','gateGREEN_sdGreen','gateGREEN_sdRed','gateGREEN_percentageEvent']]
	df_green.columns = ['AHL','IPTG','RED_median','GREEN_median',
		           'RED_mode','GREEN_mode',
		           'GREEN_std','RED_std','percentage']
	df_green = df_green[df_green['percentage']>15] # at least 15% of population
	df_green['GREEN_var'] = df_green['GREEN_std']**2
	df_green['RED_var'] = df_green['RED_std']**2
	df_green['file'] = file
	df_green['gate'] = 'green'
	df_green['RED_median_normed'] = df_green['RED_median'] # note that this is not normed since it is already in the baseline
	df_green['GREEN_median_normed'] = df_green['GREEN_median']/greennorm	
	df_green['GREEN_mode_normed'] = df_green['GREEN_mode']/greennormmode	
	df = df.append(df_green)

	df['aTc'] = 0

	# find all the possible sets of conditions

	set_of_conditions = set()
	for idx, row in df.iterrows():
		set_of_conditions.add(tuple([row['aTc'],row['IPTG'],row['AHL']]))

	# find and write for each condition
	df['multistability'] = 0 # initialize the columns to 0
	# then will be filled with 1, or 2 depending on the system showing bistability

	for condition in set_of_conditions:

		# find the indices for the given condition
		row_indices = (df['aTc']==condition[0]) & (df['IPTG']==condition[1]) & (df['AHL']==condition[2])
		# find how many unique labels are found in their gate column
		statesobserved = df[row_indices]['gate'].nunique()
		# assign that value to the 'multistability' column
		df.loc[row_indices,'multistability'] = statesobserved


####### PLOTTING FUNCTIONS OF THE LOAD DATA FOR CHECKING RESULTS

def plot_condition(x = 'IPTG', y = 'Green',chemlist = ['AHL','aTc'], val = [100,0]):
	'''
	plot the fluorescence for an array of inducer x given a fixed value of ther other inducers
	'''

	subdf = df.copy()
	for ichem,chem in enumerate(chemlist):
		subdf = subdf[subdf[chem]==val[ichem]]

	plt.plot(np.log10(subdf[x]),subdf[y],'o')
	xaxisnonzero = subdf[x][subdf[x]>0]
	minxaxis = min(xaxisnonzero)
	yaxiszero = subdf[y][subdf[x]==0]
	xaxiszero = np.ones_like(yaxiszero)*minxaxis*0.7
	print(xaxiszero,yaxiszero)

	plt.plot(np.log10(xaxiszero),yaxiszero,'v')
	plt.show()


