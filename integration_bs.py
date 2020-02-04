#! /bin/sh
""":"
exec ipython -i $0 ${1+"$@"}
"""

import matplotlib.pyplot  as plt
import matplotlib.ticker as ticker
import flow_load as fll
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import argrelextrema,argrelmin,argrelmax
from scipy.optimize import brentq
from scipy.optimize import brenth
from scipy.optimize import newton
from scipy import signal as scisignal # module to interpolate signal
from scipy import stats
import seaborn as sns
import latticediff as lat
import random
import os,sys
import datetime
import json
import copy
import yol_colors as yolcol
import pandas as pd
import matplotlib.gridspec as gridspec
import fitpars as fp
import cv2
#import line_profiler


sbcolorcycle=sns.color_palette("Set1",n_colors=9,desat=0.8) # Setting Colour palette
sns.set_palette("Set1",n_colors=9,desat=0.8)
#sbcolorcycle=sns.color_palette("deep") # Setting Colour palettes
sbcolorcyclebright=sns.color_palette("bright")
sbcolorcycledark=sns.color_palette("dark")
sbcolorcyclemuted=sns.color_palette("muted")
clw = 2.0 # circle line width (also for ellipse)
talpha = 0.05 # alpha for trajectory
sns.set_style("ticks") # Set plot properties
sns.set_context("talk")
sns.despine()
newblack = sns.xkcd_rgb["charcoal"]

eps = 1E-80
eps2 = 1E-8

colorcycle=sbcolorcycle

def facet_scatter(x,y,c, **kwargs):
    """Draw scatterplot with point colors from a faceted DataFrame columns."""
    kwargs.pop("color")
    plt.scatter(x,y,c=c, **kwargs)

# Create the color palettes used in the manuscript #####################################################
plt.register_cmap(name='RedtoGreen', data=yolcol.cdict_RedtoGreen) 
plt.register_cmap(name='AlphatoBlue', data=yolcol.cdict_AlphatoBlue) 

###### Dictionary with the integration parameters
integdict = {'dt':0.1,'totaltime':1000}

##### Class with the parameters of the model
# Numbers used are not significant
class param():
  alpha_X = 15.25
  alpha_Y = 140.0
  delta_X = 1.0
  delta_Y = 1.0
  beta_X = 1.0
  beta_Y = 1.0
  K_IPTG = 0.295
  K_AHL = 4.05
  K_LACI = 0.295
  K_TETR = 1.0
  K_ATC = 1.0
  n_LACI = 2.47
  n_AHL = 2.08
  n_TETR = 2.0

# Global dictionaries and lists to call the parameters
orderedpars = {"alpha_X":0, "alpha_Y":1, "beta_X":2, "beta_Y":3,
              "K_IPTG":4, "K_AHL":5
             , "K_LACI":6, "K_TETR":7,
             "K_ATC":8, "n_LACI":9, "n_AHL":10,
             "n_TETR":11,"delta_X":12,"delta_Y":13}

parnames = ['$\\alpha_X$','$\\alpha_Y$','$\\beta_X$', '$\\beta_Y$',
            '$K_{IPTG}$','$K_{AHL}$','$K_{LACI}$',
            '$K_{TETR}$','$K_{ATC}$',
            '$n_{LACI}$','$n_{AHL}$','$n_{TETR}$' ]

par = param() # global instance of the class. New local instances can be created if necessary

#### Class with the chemical concentrations
class chem():
  IPTG = 0.0
  AHL = 0.0
  ATC = 0.0

ch = chem() # global instance of the class. New local instances can be created if necessary

##############################################################################
#//////////////////////////// DYNAMICAL SYSTEM DEFINITION /////////////////////////////////
#################################################################################

def flow(X,Y,chh = ch, parr = par):
  ''' returng the value for xdot and ydot for a given set of parameters par
  and a given set of inducers ch.
  '''

# X is expression of the lux - lac promoter encoding TetR and mCherry  
# Y is expression of the tet promote inducing encoding LacI and GFP 

  # X and Y are measurments of fluorescence, to use them I am goind to substract some background
  # to transform it to concentrations

  X = np.maximum(X - parr.alpha_X,0) # removing fluorescence background on X
  Y = np.maximum(Y - parr.alpha_Y,0) # removing fluorescence background on Y
  # X = X - parr.alpha_X
  # Y = Y - parr.alpha_Y
  # note the danger of this expression, values of X should not let drop below X to have analytical significance
  # nevertheless this has meaning (fluorescences below the background correspond to background -> X=0), in addition
  # it can help to have a proper analysis when X- parr.alpha_X goes a bit below zero (due to computational error) and
  # we don't want this to affect the analysis 

  # the role of AHL is an AND gate coded as a Hill activator
  flow_X = np.power(chh.AHL * parr.K_AHL,parr.n_AHL)/(1 + np.power(chh.AHL*parr.K_AHL,parr.n_AHL))
  # LacI can be  available to bind to the lux - lac promoter if they do not form a complex with IPTG molecules
  free_LacI = Y/(parr.K_IPTG * chh.IPTG + 1)
  # free LacI acts as an AND repressor by looping the DNA coded as a Hill function
  flow_X = flow_X / (1 + np.power(parr.K_LACI * free_LacI, parr.n_LACI))
  flow_X = parr.beta_X*flow_X # Note that I am introducing handwavily some leakiness of the promoter (any better way?) 
  flow_X = flow_X - parr.delta_X * X # degradation of protein X (mCherry and TetR are considered the same...)


  # free TeTR can inhibit the expression of the TetO upstream the gene Y
  # aTc can attach to TetR inhibiting the binding, I am going to assume normal binding kinetics
  free_TetR = X/(parr.K_ATC * chh.ATC + 1)
  # For the repression from the free_TetR I will assume Hill function
  flow_Y = 1.0 / (1 + np.power(parr.K_TETR * free_TetR, parr.n_TETR))
  flow_Y = parr.beta_Y*flow_Y # Note that I am introducing handwavily some leakiness of the promoter (any better way?) 
  flow_Y = flow_Y - parr.delta_Y * Y

  # note that the flows do not need to be transformed to fluorescence (without background)
  # since they coincide in both cases

  return flow_X,flow_Y


def getJacobian(X,Y,chh = ch, parr = par):
  ''' Returns Jacobian of the flow
  '''

  Jacobian = np.zeros([2,2])
  X = np.maximum(X - parr.alpha_X,0) # fluorescence background on X
  Y = np.maximum(Y - parr.alpha_Y,0) # fluorescence background on X
  free_LacI = Y/(parr.K_IPTG * chh.IPTG + 1)
  free_TetR = X/(parr.K_ATC * chh.ATC + 1)
  Jacobian[0,0] = -parr.delta_X # Diagonals of the Jacobian corresponds with the degradation rates
  Jacobian[1,1] = -parr.delta_Y
  Jacobian[0,1] = -np.power(chh.AHL * parr.K_AHL,parr.n_AHL)/(1 + np.power(chh.AHL*parr.K_AHL,parr.n_AHL))
  Jacobian[0,1] /= (1 + np.power(parr.K_LACI * free_LacI, parr.n_LACI))
  Jacobian[0,1] /= (1 + np.power(parr.K_LACI * free_LacI, parr.n_LACI))
  Jacobian[0,1] *= parr.K_LACI * parr.n_LACI * np.power(parr.K_LACI * free_LacI, parr.n_LACI-1)
  Jacobian[0,1] *= parr.beta_X
  Jacobian[0,1] /= (parr.K_IPTG * chh.IPTG + 1)

  Jacobian[1,0] = -parr.beta_Y / (1 + np.power(parr.K_TETR * free_TetR, parr.n_TETR))
  Jacobian[1,0] /= (1 + np.power(parr.K_TETR * free_TetR, parr.n_TETR))
  Jacobian[1,0] *= parr.n_TETR * parr.K_TETR * np.power(parr.K_TETR * free_TetR, parr.n_TETR-1)
  Jacobian[1,0] /= (parr.K_ATC * chh.ATC + 1)

  return Jacobian


def flowWithNoiseIntensity(X,Y,chh = ch, parr = par):
# assuming that the amount of morphogen molecules is larger compared to the proteins interacting proteins
# and that the change in protein number is low enough to consider it constant in a timestep
# joining the whole production to noise to a single gaussian then we can apply CLE as 

  # X and Y are measurments of fluorescence, to use them I am going to substract some background
  # to transform it to concentrations

  X = np.maximum(X - parr.alpha_X,0) # fluorescence background on X
  Y = np.maximum(Y - parr.alpha_Y,0) # fluorescence background on X
  # X = X - parr.alpha_X
  # Y = Y - parr.alpha_Y
  # note the danger of this expression, values of X should not let drop below X to have analytical significance
  # nevertheless this has meaning (fluorescences below the background correspond to background -> X=0), in addition
  # it can help to have a proper analysis when X- parr.alpha_X goes a bit below zero (due to computational error) and
  # we don't wnat this to affect the analysis 

  # the role of AHL is an AND gate coded as a Hill activator
  flow_X = np.power(chh.AHL * parr.K_AHL,parr.n_AHL)/(1 + np.power(chh.AHL*parr.K_AHL,parr.n_AHL))
  # LacI can be  available to bind to the lux - lac promoter if they do not form a complex with IPTG molecules
  free_LacI = Y/(parr.K_IPTG * chh.IPTG + 1)
  # free LacI acts as an AND repressor by looping the DNA coded as a Hill function
  flow_X = flow_X / (1 + np.power(parr.K_LACI * free_LacI, parr.n_LACI))
  flow_X = parr.beta_X*flow_X # Note that I am introducing handwavily some leakiness of the promoter (any better way?) 
  noise_X = np.sqrt(flow_X*flow_X + parr.delta_X * X * parr.delta_X * X) # CLE assuming stoichiometry 1
  # noise_X = parr.delta_X * (X+parr.alpha_X) # CLE assuming stoichiometry 1
  flow_X = flow_X - parr.delta_X * X # degradation of protein X (mCherry and TetR are considered the same...)


  # free TeTR can inhibit the expression of the TetO upstream the gene Y
  # aTc can attach to TetR inhibiting the binding, I am going to assume normal binding kinetics
  free_TetR = X/(parr.K_ATC * chh.ATC + 1)
  # For the repression from the free_TetR I will assume Hill function
  flow_Y = 1.0 / (1 + np.power(parr.K_TETR * free_TetR, parr.n_TETR))
  flow_Y = parr.beta_Y*flow_Y # Note that I am introducing handwavily some leakiness of the promoter (any better way?)
  noise_Y = np.sqrt(flow_Y*flow_Y + parr.delta_Y * Y * parr.delta_Y * Y) # CLE assuming stoichiometry 1 
  # noise_Y = parr.delta_Y * (Y+parr.alpha_Y) # CLE assuming stoichiometry 1 
  flow_Y = flow_Y - parr.delta_Y * Y


  # note that the flows do not need to be transformed to fluorescence (without background)
  # since they coincide in both cases

  return flow_X,flow_Y,noise_X,noise_Y

#


# Flow of Gardner et al Nature:
#  flow_X = parr.alpha_X/(1+np.power(Y,parr.n_X)) - parr.delta_X * X
#  flow_Y = parr.alpha_Y/(1+X/np.power(1+chh.IPTG/parr.K_IPTG,parr.n_IPTG)) - parr.delta_Y * Y
#
# with parameters:
  # alpha_X = 156.25
  # alpha_Y = 15.6
  # delta_X = 1.0
  # delta_Y = 1.0
  # K_IPTG = 2.9618E-5
  # n_IPTG = 2.0015
  # n_X = 2.5

###############################################################################
##### INTEGRATION OF THE DYNAMICAL SYSTEM
##############################################################################

#/////////////////////////////////////////////////////////////////////////
def integration(totaltime = None, dt = None,initcond=[0,0], result = "traj", stochastic = False, volume = 100, parr = par, chh = ch):
  ''' Euler integration of a trajectory for an initial set of coordinates initcond 
  result --- 'traj' returns the whole trajectory, otherwise only the last point is returned
  '''
#Integration of the trajectory. Returns a vector of X,Y,Z,t

#
  ti=0 # Initial time
  if not dt :
    dt = integdict["dt"]
  if not totaltime:
    totaltime = integdict["totaltime"]

  sdt = np.sqrt(dt)
  volume_prefactor = np.sqrt(1.0/volume)

  Xi,Yi=initcond

  if result =="traj":
    X=[] # vector containing trajectory
    Y=[]
    t=[]

  while (ti<totaltime):
    if result == "traj":
      X.append(Xi)
      Y.append(Yi)
      t.append(ti)

    if stochastic:
      flow_X,flow_Y,noise_X,noise_Y = flowWithNoiseIntensity(Xi,Yi,chh,parr)  

      Xi = Xi + dt*flow_X + noise_X*sdt*volume_prefactor * np.random.normal()
      Yi = Yi + dt*flow_Y + noise_Y*sdt*volume_prefactor * np.random.normal()
    else:
      flow_X,flow_Y = flow(Xi,Yi,chh,parr) 
      Xi = Xi+dt*flow_X
      Yi = Yi+dt*flow_Y

    ti = ti + dt

  if result == "traj":
    return X,Y,t
  else:
    return Xi,Yi,ti


#//////////////////////////////////////////////////////////////////////
### LINEAR STABILITY ANALYSIS AND COMPARISON WITH EXPERIMENTAL DATA
#/////////////////////////////////////////////////////////////////////////

def steady_states(X0 = -8, XF = 4, NX = 100, parr = par, chh = ch, givenxi = None):
  '''Compute the steady states available for the bistable switch along X = [10^X0,10^XF]
  this range is divided in NX points and the extrema of f(X)-X are located.
  Between each extrema, if there is a change of sign of the flow, it means that
  there is a steady state between these points that is refined by the Brent routine
  If givenxi is different than None, then the list of initial X is given by givenxi

  returns a list of steady states with their coordinates X,Y,
  their stability and the eigenvalues of the Jacobian at that point
  '''

  ss = [] # list of found steady states

  if givenxi is None:
    xi = np.logspace(X0,XF,num=NX)
  else:
    xi = givenxi
    
  yi = flow(xi,parr.alpha_Y,parr = parr, chh = chh)[1]/parr.delta_Y+parr.alpha_Y
  zi = flow(xi,yi,parr = parr,chh = chh)[0] # this is zero at the steady state xstar,
  # so zi computes the deviation from the steady state

  maxi=argrelmax(np.array(zi))[0] # look for the relative maxima
  mini=argrelmin(np.array(zi))[0] # and minima on x

  zcpos=np.concatenate(([0],maxi,mini,[NX-1]))
  zcpos=np.sort(zcpos) # zcpos contains the regions that alocate steady states
  idx=0
  while(idx<(len(zcpos)-1)):
    if ((zi[zcpos[idx]]*zi[zcpos[idx+1]])<0):
    # if there is a change of sign, a brentq algorithm is used to find the zero (steady-state) xstar
      xstar = brenth(lambda x: flow(x,flow(x,parr.alpha_Y,parr = parr, chh = chh)[1]/par.delta_Y+parr.alpha_Y,parr = parr, chh = chh)[0]  ,xi[zcpos[idx]],xi[zcpos[idx+1]])
      # we are using the brenth algorithm because the hyperbolic finds faster minima in this case (tested numerically)
      ystar = flow(xstar,parr.alpha_Y,parr = parr, chh = chh)[1]/parr.delta_Y+parr.alpha_Y

      stable,ls=stabilityJacobian(xstar,ystar,parr = parr, chh = chh) # stability is assessed by diagonalization of the Jacobian
      if stable == True:
        stable = 'stable'
      elif stable == False:
        stable = 'unstable'

      ss.append({"X":xstar,"Y":ystar,"stab":stable,"lambdas":ls}) 
    idx=idx+1

  borders = [0,-1] # possibility that the zero is at the edge of the vector
  for b in borders:
    if abs(zi[b]) < eps2 : 
      xstar = xi[b]
      ystar = yi[b]
      stable,ls=stabilityJacobian(xstar,ystar,parr = parr, chh = chh)
      if stable == True:
        stable = 'stable'
      elif stable == False:
        stable = 'unstable'
      ss.append({"X":xstar,"Y":ystar,"stab":stable,"lambdas":ls}) 

  return ss

def loglikelihood(dataset, paramvector):
  ''' Calculate the loglikelihood of the dataset given a vector of parameters
  '''

  # Casting from list (useful for the use of PyDream) to the global parameter structure par
  par.alpha_X = paramvector[orderedpars['alpha_X']]
  par.alpha_Y = paramvector[orderedpars['alpha_Y']]
  par.beta_X = paramvector[orderedpars['beta_X']]
  par.beta_Y = paramvector[orderedpars['beta_Y']]
  par.K_IPTG = paramvector[orderedpars['K_IPTG']]
  par.K_AHL = paramvector[orderedpars['K_AHL']]
  par.K_LACI = paramvector[orderedpars['K_LACI']]
  par.K_TETR = paramvector[orderedpars['K_TETR']]
  par.K_ATC = paramvector[orderedpars['K_ATC']]
  par.n_LACI = paramvector[orderedpars['n_LACI']]
  par.n_AHL = paramvector[orderedpars['n_AHL']]
  par.n_TETR = paramvector[orderedpars['n_TETR']]
  #par.delta_X = paramvector[orderedpars['delta_X']]
  #par.delta_Y = paramvector[orderedpars['delta_Y']]
  par.delta_X = 1.0
  par.delta_Y = 1.0


  totalloglklh = 0

  # vector to look for values
  xii = np.logspace(np.log10(par.alpha_X+eps),np.log10(par.alpha_X+par.beta_X-eps),num=1000)

  for irow,row in dataset.iterrows():
    # for each point in the dataset set the inducer concentration
    chem.AHL = row['AHL']
    chem.IPTG = row['IPTG']
    chem.aTc = row['aTc']


    # search of steady states is limited to the range of possible values of fluorescence X
    ss = steady_states(np.log10(par.alpha_X+eps),np.log10(par.alpha_X+par.beta_X-eps),1000,parr = par, chh = chem, givenxi = xii)
    maxlklh = -1E10 # basal loglikelihood, to avoid outlayers 
    steadyStatesCount = 0
    if ss == []:
      print("Missed point!!", chem.AHL , chem.IPTG, chem.aTc)
    for state in ss:
      if state['stab'] == 'stable':
        steadyStatesCount += 1
        # normalizations of Gaussians might be dropped to give the same relevance to heights of coexistent peaks
        # of difference basins of attractions i.e. that a narrow peak in coexistence with a broad one does not want to 
        # absorb everything

        lklh = ((state['X']-row['RED_median_normed'])/(row['RED_std']))**2 # assuming gaussian 
        lklh += ((state['Y']-row['GREEN_median_normed'])/(row['GREEN_std']))**2 # assuming gaussian 

        # Alternative Likelihoods
        #lklh = ((state['X']-row['RED_median_normed'])/1000.0)**2 # assuming gaussian 
        #lklh += ((state['Y']-row['GREEN_median_normed'])/1000.0)**2 # assuming gaussian

        # Adding normalizations...
        #lklh = lklh + np.log(row['GREEN_std']*np.sqrt(2.0*np.pi)) # normalization of gaussian 
        #lklh = lklh + np.log(row['RED_std']*np.sqrt(2.0*np.pi)) # normalization of gaussian

        lklh = lklh*(-1.0)
        if lklh > maxlklh:
          maxlklh = lklh
    if ((steadyStatesCount>1) and (row['multistability'] ==1) or
        (steadyStatesCount==1) and (row['multistability'] ==2)):
      maxlklh -= 100  # penalizing not finding the right number of steady states
    # print chem.AHL, chem.IPTG, row['Red'], row['Green'], ss, mindistance
    
    totalloglklh += maxlklh

  print("vector: ", paramvector)
#  print("logvector: ", np.log10(paramvector))
  print("log likelihood: ", totalloglklh)
  return totalloglklh

def getbifurcationcurves_AHL(limAHL = [0.001,100], npoints = 1000, IPTG=0, ATC=0):
  ''' Return the bifurcation curves in a range of AHL by filling the three curves
  curve_stable1, curbe_stable2, and curve_unstable.
  Each curve is a list of AHL values and the corresponding position X,Y
  '''

  print()
  AHLarray = np.logspace(np.log10(limAHL[0]), np.log10(limAHL[1]), npoints)
  print(AHLarray)
  ch.ATC = ATC
  ch.IPTG = IPTG

  curve_stable_1 = np.ones((0,2))
  curve_unstable = np.ones((0,2))
  curve_stable_2 = np.ones((0,2))

  last_value_1 = 0 # This values will be used to choose curve to continue
  last_value_2 = np.inf
  for AHLv in AHLarray:
    # print('AHLv',AHLv)
    ch.AHL = AHLv
    ss = steady_states(np.log10(par.alpha_X+eps),np.log10(par.alpha_X+par.beta_X-eps),
               NX = 1000, parr = par, chh = ch, zerocorrection = True, givenxi = None)
    sss = [] # list of stable steady states found for this value of AHL
    for xss in ss:
      if xss['stab']=='stable':
        sss.append(xss['X'])
        # not assigned until we know all of them
      else: # unstable
        curve_unstable = np.vstack((curve_unstable,[AHLv,xss['X']]))

    if len(sss)==1: # Only one state then we decide which branch the it belongs to
      if (abs(sss[0]-last_value_1) < abs(sss[0]-last_value_2)):
        last_value_1 = sss[0]
        curve_stable_1 = np.vstack((curve_stable_1,[AHLv,sss[0]]))
        last_value_2 = np.inf
      else:
        last_value_2 = sss[0]
        curve_stable_2 = np.vstack((curve_stable_2,[AHLv,sss[0]]))
        last_value_1 = np.inf

    elif len(sss)==2: # bistable regime
        if (abs(sss[0]-last_value_1) < abs(sss[0]-last_value_2)):
          l1 = 0
          l2 = 1
        else:
          l1 = 1
          l2 = 0

        last_value_1 = sss[l1]
        last_value_2 = sss[l2]
        curve_stable_1 = np.vstack((curve_stable_1,[AHLv,sss[l1]]))
        curve_stable_2 = np.vstack((curve_stable_2,[AHLv,sss[l2]]))

    else:
      print("Error, too many stable states")

  return curve_stable_1,curve_stable_2,curve_unstable          


##################
# /////////////// AUXULIARY LINEAR STABILITY FUNCTIONS ////////////////////////////////////////////////////
##################

def stabilityJacobian(x,y,eps = 1E-10,parr = par, chh = ch):
  ''' Compute numerically the stability of a steady state 
  diagonalizing the numerical Jacobian
  '''

  fxx,fyx = flow(x+eps,y,parr = parr,chh = chh)
  fxy,fyy = flow(x,y+eps,parr = parr,chh = chh)
  lambdas,vecs=np.linalg.eig([[fxx,fyx],[fxy,fyy]])
  stable=all([l.real<0 for l in lambdas])
  return stable, lambdas

def getNumericalJacobian(x,y,eps = 1E-10,parr = par, chh = ch):
  ''' Compute the Jacobian numerically  at a point buy using finite differences
  '''
  print("Jac",x,y,eps,x+eps)

  fxxP,fyxP = flow(x+eps,y,parr = parr,chh = chh)
  fxxM,fyxM = flow(x-eps,y,parr = parr,chh = chh)
  fxyP,fyyP = flow(x,y+eps,parr = parr,chh = chh)
  fxyM,fyyM = flow(x,y-eps,parr = parr,chh = chh)
  fxx = (fxxP-fxxM)/(2*eps)
  fyy = (fyyP-fyyM)/(2*eps)
  fxy = (fxyP-fxyM)/(2*eps)
  fyx = (fyxP-fyxM)/(2*eps)
  return np.array([[fxx,fyx],[fxy,fyy]])

#####################################################################
# #//////////////////////////////////  PLOTS  ////////////////////////
######################################################################
# Plots used to show stability and integration of the flow equations without spatial diffusion of signals

def plotpargrid(chain = 'all', ordermethod = 'logps', condition = 40000, figsize = 20):
  
  '''
  Plot a grid with the results of the MCMC

  chain --- which chains to use. 'all' uses all the chains
  ordermethod --- There are two different ways of ordering the goodness of the results. 
    One of them is compute one by one the likelihoods for each parset
    This methods can be quite slow sometimes. The second option is to load the logps from the output of pyDream
  condition --- Which output file to read (condition refers to number of samples from the MCMC)
  figsize --- size of figure in inches
  '''

  ntrim = 10000 # trim the transient of the MCMC
  stride = 50 # ignore every 50 points to avoid close points and make the plot lighter
  condition = str(condition)
  chain = chain
  lklh_ci = 0.9 ## confidence interval threshold to plot 

  data1 = np.array([]).reshape(0,len(parnames))
  lklh_list = np.array([])

  if chain == 'all':
      chainlist = [0,1,2]
  else:
      chainlist = [chain]      
  for ch in chainlist:
      data1_temp = np.load('dream/bsfit_sampled_params_chain'+str(ch)+'_'+condition+'.npy')
      logpsdata1_temp = np.load('dream/bsfit_logps_chain'+str(ch)+'_'+condition+'.npy')
      lklh_list_temp = np.array([l[0] for l in logpsdata1_temp])
      data1_temp = data1_temp[ntrim::stride]
      lklh_list_temp = lklh_list_temp[ntrim::stride]
      data1 = np.vstack((data1,data1_temp))
      lklh_list = np.concatenate((lklh_list,lklh_list_temp))
  # Trim data with best likelihoods

  if ordermethod == 'logps':

    sorted_idxs = lklh_list.argsort()
    data1_sorted = data1[sorted_idxs]
    lklh_list_sorted = lklh_list[sorted_idxs]
    bestlklh = lklh_list[-1]
    worstlklh = lklh_list[0]
    bestpars = data1_sorted[-1]
    ibestlklh = -1

  # threshold_lklh = lklh_list_sorted[int(len(lklh_list_sorted)*lklh_ci)] # finding the lklh at a certain confidence interval

  print("ibestlklh: ", ibestlklh)
  print("bestlklh: ", bestlklh)
  print("bestpars:", bestpars)
  # print "teopars:", teopars
  print("worstlklh: ", worstlklh)
  # print "thresholdlklh: ", threshold_lklh

  print(data1_sorted)


  axislim = np.array([fp.lower_limits,fp.lower_limits+fp.scale_limits]).T
  
  axislim[-1]  = 10**axislim[-1]
  axislim[-2]  = 10**axislim[-2]
  axislim[-3]  = 10**axislim[-3]
  axislim[0]  = [0,550]
  axislim[1]  = [0,550]

  # plot_parnames = ['log($\\alpha_X$)','log($\\alpha_Y$)','log($\\beta_X$)',
  #             'log($\\beta_Y$)','log($K_{IPTG}$)','log($K_{AHL}$)',
  #             'log($K_{LACI}$)','log($K_{TETR}$)','log($K_{ATC}$)',
  #             '$n_{LACI}$','$n_{AHL}$','$n_{TETR}$']

  # Plot parnames excluding KaTc
  plot_parnames = ['$\\alpha_X$','$\\alpha_Y$','$\\mathrm{log}_{10}\\tilde{\\beta}_X$',
              '$\\mathrm{log}_{10}\\tilde{\\beta}_Y$','$\\mathrm{log}_{10}K_{IPTG}$','$\\mathrm{log}_{10}K_{AHL}$',
              '$\\mathrm{log}_{10}K_{LACI}$','$\\mathrm{log}_{10}K_{TETR}$',
              '$n_{LACI}$','$n_{AHL}$','$n_{TETR}$']
  data1_sorted = np.delete(data1_sorted,8,1) # deleting column for KATC
  axislim = np.delete(axislim,8,0) # deleting column for KATC


  data_df1 = pd.DataFrame(data = data1_sorted, columns = plot_parnames)

  data_df1['$\\alpha_X$'] = 10**data_df1['$\\alpha_X$']
  data_df1['$\\alpha_Y$'] = 10**data_df1['$\\alpha_Y$']
  
  data_df1['$n_{LACI}$'] = 10**data_df1['$n_{LACI}$']
  data_df1['$n_{AHL}$'] = 10**data_df1['$n_{AHL}$']
  data_df1['$n_{TETR}$'] = 10**data_df1['$n_{TETR}$']

  print("Array dimesions: ", data_df1.shape)
  cmap = cm.get_cmap('viridis')

  fig = plt.figure(figsize = (figsize,figsize))
  gs = gridspec.GridSpec(len(plot_parnames), len(plot_parnames))
  gaxes = []

  # Plot each subplot in coordinates irow,icol in a grid of plots
  for irow,row in enumerate(plot_parnames):
    grow = []
    for icol,col in enumerate(plot_parnames):
      grow.append(plt.subplot(gs[irow,icol]))
    gaxes.append(grow)  

  # Plot of offdiagonal scatter plots

  for irow,rowname in enumerate(plot_parnames):
    for icol,colname in enumerate(plot_parnames):
      ## scatter plots
      if icol != irow:
        gaxes[irow][icol].set_facecolor(cmap(0))
        gaxes[irow][icol].scatter(data_df1[colname],data_df1[rowname], c = lklh_list_sorted,
          s = 2, cmap = 'viridis', vmin = lklh_list_sorted[0], vmax = lklh_list_sorted[-1])
        gaxes[irow][icol].set_xlim([axislim[icol][0],axislim[icol][1]])
        gaxes[irow][icol].set_ylim([axislim[irow][0],axislim[irow][1]])
      ## density plots
      if icol == irow:
        gaxes[irow][icol].hist(data_df1[rowname],histtype = 'stepfilled', color = sbcolorcyclemuted[4], edgecolor = sbcolorcycledark[4])
        gaxes[irow][icol].set_xlim([axislim[irow][0],axislim[irow][1]])
      if irow==(len(plot_parnames)-1): # bottom row
        gaxes[irow][icol].set_xlabel(colname)
      else: # not bottom row
        gaxes[irow][icol].tick_params(labelbottom=False)
      if icol==0: # leftmost column
        gaxes[irow][icol].set_ylabel(rowname)
      else:
        gaxes[irow][icol].tick_params(labelleft=False)
      if (irow==0 and icol==0): # top left panel fix label
        gaxes[irow][icol].tick_params(labelleft=False)
        gaxes[irow][icol].yaxis.set_label_coords(-0.5,0.5)

  plt.gcf().set_size_inches(figsize, figsize)
  plt.savefig('pargrid_'+str(chain)+'_hill_'+str(ordermethod)+'.png',dpi=300)
  plt.show()


###############################################################################
def plot_testpars(chain = 1, parset = 'fit', figsize = (10,10), condition = 40000, bifurcation_line = False):
  '''
  Plot the bifurcation diagram on top of the input experimental data used to fit the model
  '''
  if parset == 'fit':
      condition = str(condition)
      chain = chain
      data1 = np.load('dream/bsfit_sampled_params_chain'+str(chain)+'_'+condition+'.npy')
      logpsdata1 = np.load('dream/bsfit_logps_chain'+str(chain)+'_'+condition+'.npy')

      lklh_list = np.array([l[0] for l in logpsdata1])

      sorted_idxs = lklh_list.argsort()
      data1_sorted = data1[sorted_idxs]
      bestpars = data1_sorted[-1]
      print(bestpars)
      #ibestlklh = -1

      par.alpha_X = 10**bestpars[orderedpars['alpha_X']]
      par.alpha_Y = 10**bestpars[orderedpars['alpha_Y']]
      par.beta_X = 10**bestpars[orderedpars['beta_X']]
      par.beta_Y = 10**bestpars[orderedpars['beta_Y']]
      par.K_IPTG = 10**bestpars[orderedpars['K_IPTG']]
      par.K_AHL = 10**bestpars[orderedpars['K_AHL']]
      par.K_LACI = 10**bestpars[orderedpars['K_LACI']]
      par.K_TETR = 10**bestpars[orderedpars['K_TETR']]
      par.K_ATC = 10**bestpars[orderedpars['K_ATC']]
      par.n_LACI = 10**bestpars[orderedpars['n_LACI']]
      par.n_AHL = 10**bestpars[orderedpars['n_AHL']]
      par.n_TETR = 10**bestpars[orderedpars['n_TETR']]
      #par.delta_X = bestpars[orderedpars['delta_X']]
      #par.delta_Y = bestpars[orderedpars['delta_Y']]
      par.delta_X = 1.0
      par.delta_Y = 1.0

  else:
      # Some example interesting parameters in case that a file is missing
      par.alpha_X = (10**2.65397418e+00)*1.67-404.74
      par.alpha_Y = 10**2.64892714e+00
      par.beta_X = (10**2.39798631e+00)*1.67
      par.beta_Y = 10**2.50684751e+00
      #par.K_IPTG = 10**-2.84705254e-01
      par.K_IPTG = 10**(0.0827)
      par.K_AHL = 10**2.16683127e+00
      par.K_LACI = 10**-1.628084364e+00 # strength of Y on X
      par.K_TETR = (10**-0.3461396e+00)/1.67 # strength of X on Y
      par.K_ATC = 10**-9.2435282e-01
      par.n_LACI = 10**0.58215
      par.n_AHL = 10**0.22496
      par.n_TETR = 10**0.14035
      # par.delta_X = bestpars[orderedpars['delta_X']]
      # par.delta_Y = bestpars[orderedpars['delta_Y']]
      par.delta_X = 1.0
      par.delta_Y = 1.0
      print("loglikelihood: ", loglikelihood(fll.df,[par.alpha_X, par.alpha_Y,
        par.beta_X, par.beta_Y, par.K_IPTG, par.K_AHL, par.K_LACI, par.K_TETR,
        par.K_ATC, par.n_LACI, par.n_AHL, par.n_TETR]))
      


  #IPTGconditions = [0,0.0625,0.09,0.125,0.16,0.5,1,10]
  IPTGconditions = [0,0.125,10]
  columns = ['RED_median_normed','GREEN_median_normed']
  Dcolumns = ['RED_var','GREEN_var']


  fig = plt.figure(constrained_layout = False, figsize = figsize)
  gs = gridspec.GridSpec(len(IPTGconditions), len(columns))

  for icond,cond in enumerate(IPTGconditions): # for each row of the plotgrid
    rowsdf = fll.df[fll.df['IPTG']==cond] # these is the subdataframe with the condition of the gridrow
    sslist = [] # list with the steady states found
    if bifurcation_line is False:
      for irow,row in rowsdf.iterrows(): # for each condition let's compute the steady states
          chem.AHL = row['AHL']
          chem.IPTG = row['IPTG']
          chem.aTc = row['aTc']
          steadystates = steady_states(np.log10(par.alpha_X+eps),np.log10(par.alpha_X+par.beta_X-eps),1000,par,chem)
          steadystates = [ s for s in steadystates if s['stab']=='stable'] # only plotting stable states
          for ss in steadystates:
            if (ss['X']>0 and ss['Y']>0):
              sslist.append([chem.AHL,ss['X'],ss['Y']])
      sslist = np.array(sslist)

    elif bifurcation_line is True:
      if cond > 9: # this regulation takes into account that IPTG seems to saturate the bacterial behaviour for large concentrations
        xcond = 7
      else:
        xcond = cond
      s1,s2,u =  getbifurcationcurves_AHL(limAHL = [1E-6,10], npoints = 1000, IPTG=xcond, ATC=0)
      if len(u)>0:
        u = np.array([cp for cp in u if cp[0]>1E-5]) # this line avoids some spurious unstable states detected for extreme values of IPTG
      print ('unstable line',len(u))

    for icol,column in enumerate(columns):
      axes = plt.subplot(gs[icond,icol])
      if sslist != []:
        for ss in sslist:
          if ss[0] > 0:
            plt.plot([np.log10(ss[0])],[ss[icol+1]],'s',markersize = 5,
              color =sbcolorcycledark[3-icol], zorder = 10)
          else: # if AHL = 0  
            plt.plot([-5],[ss[icol+1]],'s',markersize = 5,
              color =sbcolorcycledark[3-icol], zorder = 10)
      if bifurcation_line is True:
        if icol == 0:
          plt.plot(np.log10(s1[:,0]), s1[:,1],'-',color =sbcolorcycledark[3-icol])
          if len(s2)>0:
            plt.plot(np.log10(s2[:,0]), s2[:,1],'-',color =sbcolorcycledark[3-icol])
          if len(u)>0:
            plt.plot(np.log10(u[:,0]), u[:,1],':',color =sbcolorcycledark[3-icol])
        else:
          y1 = flow(s1,par.alpha_Y,parr = par, chh = chem)[1]/par.delta_Y+par.alpha_Y
          plt.plot(np.log10(s1[:,0]), y1[:,1],'-',color =sbcolorcycledark[3-icol])
          if len(s2)>0:
            y2 = flow(s2,par.alpha_Y,parr = par, chh = chem)[1]/par.delta_Y+par.alpha_Y
            plt.plot(np.log10(s2[:,0]), y2[:,1],'-',color =sbcolorcycledark[3-icol])
          if len(u)>0:
            yu = flow(u,par.alpha_Y,parr = par, chh = chem)[1]/par.delta_Y+par.alpha_Y
            plt.plot(np.log10(u[:,0]), yu[:,1],':',color =sbcolorcycledark[3-icol])

      for rrow in rowsdf.iterrows():
        if rrow[1]['AHL']>0: # iterrows returns a tuple, element 1 is the info  
          plt.errorbar([np.log10(rrow[1]['AHL'])],[rrow[1][column]],[np.sqrt(rrow[1][Dcolumns[icol]])],
               mfc =sbcolorcyclebright[3-icol], color =sbcolorcyclebright[3-icol],
               fmt = 'o', markersize = 5, zorder = -1, markeredgewidth = 0.5, mec = sbcolorcycledark[3-icol])
        else :  # if AHL = 0
          plt.errorbar([-6],[rrow[1][column]],[np.sqrt(rrow[1][Dcolumns[icol]])],
               mfc =sbcolorcyclebright[3-icol], color =sbcolorcyclebright[3-icol],
               fmt = '<', markersize = 5, zorder = -1, markeredgewidth = 0.5, mec = sbcolorcycledark[3-icol])

      if (icond == len(IPTGconditions)-1):
        plt.xlabel('$\\log_{10}([AHL])$ ($\\mu M$)')
      if (icol == 0 and icond == len(IPTGconditions)//2):
        axes.set_ylabel('Fluorescence mCherry')
        axes.yaxis.label.set_color('red')
      if (icol == 1 and icond == len(IPTGconditions)//2):
        axes.set_ylabel('Fluorescence GFP') 
        axes.yaxis.label.set_color('green')

      if (icol == 0):
        plt.text(-4,800,"[IPTG] = "+str(IPTGconditions[icond])+" mM", fontsize = 12)
        axes.set_ylim([220,1000])
      if (icol ==1 ):
        axes.yaxis.tick_right()
        axes.yaxis.set_label_position("right")
        axes.set_ylim([121,1200])

  plt.tight_layout()
  gs.update(wspace=0.1, hspace=0.1)
  plt.savefig('testpars.pdf')
  plt.show()
  return par

def plot_stochasticdynamics(chain=1):
 # Function to test and show indivudual trajectories and details of the dynamical system 

    condition = '60000'
    chain = chain
    data1 = np.load('dream/bsfit_sampled_params_chain_'+str(chain)+'_'+condition+'.npy')
    logpsdata1 = np.load('dream/bsfit_logps_chain_'+str(chain)+'_'+condition+'.npy')

    lklh_list = np.array([l[0] for l in logpsdata1])

    sorted_idxs = lklh_list.argsort()
    data1_sorted = data1[sorted_idxs]
    bestpars = data1_sorted[-1]
    print(bestpars)
    ibestlklh = -1

    par.alpha_X = 10**bestpars[orderedpars['alpha_X']]
    par.alpha_Y = 10**bestpars[orderedpars['alpha_Y']]
    par.beta_X = 10**bestpars[orderedpars['beta_X']]
    par.beta_Y = 10**bestpars[orderedpars['beta_Y']]
    par.K_IPTG = 10**bestpars[orderedpars['K_IPTG']]
    par.K_AHL = 10**bestpars[orderedpars['K_AHL']]
    par.K_LACI = 10**bestpars[orderedpars['K_LACI']]
    par.K_TETR = 10**bestpars[orderedpars['K_TETR']]
    par.K_ATC = 10**bestpars[orderedpars['K_ATC']]
    par.n_LACI = 10**bestpars[orderedpars['n_LACI']]
    par.n_AHL = 10**bestpars[orderedpars['n_AHL']]
    par.n_TETR = 10**bestpars[orderedpars['n_TETR']]
    #par.delta_X = bestpars[orderedpars['delta_X']]
    #par.delta_Y = bestpars[orderedpars['delta_Y']]
    par.delta_X = 1.0
    par.delta_Y = 1.0

    chem.AHL = 10
    chem.IPTG = 0.0625
    chem.aTc = 0.0


    # x-nullcline dotx = 0
    ylist = np.linspace(par.alpha_Y,par.alpha_Y+par.beta_Y,500)
    # xnull = flow(0,ylist)[0]

    xnull = flow(par.alpha_X,ylist,chh=chem,parr = par)[0]+par.alpha_X

    print('xnull', xnull)

    # y-nullcline doty = 0
    xlist = np.linspace(par.alpha_X,par.alpha_X+par.beta_X,500)
    # ynull = flow(xlist,0)[1]
    ynull = flow(xlist,par.alpha_Y,chh=chem,parr = par)[1]+par.alpha_Y

    plt.plot(xnull,ylist,'-b')
    plt.plot(xlist,ynull,'-r')

    # plot steady states
    sss = steady_states(np.log10(par.alpha_X+eps),np.log10(par.alpha_X+par.beta_X-eps),1000,par,chem)
    print(sss)
    for ss in sss:
      if ss['stab'] == "stable":
        plt.plot([ss['X']],[ss['Y']],'ok')
      if ss['stab'] == "unstable":
        plt.plot([ss['X']],[ss['Y']],'ow',markeredgecolor='k',markeredgewidth=2)

    plt.xlim(200,1000)
    plt.ylim(200,1000)

    # stochastic trajectory

    x0,y0 = 430,780
    timeIntegration = 100
    dt = 0.01 

    for j in range(10):
      traj = integration(totaltime = timeIntegration, dt = dt, initcond=[x0,y0], result = "traj",
                      stochastic = True, volume = 10, parr = par, chh = chem)

      plt.plot(traj[0],traj[1],'g',lw=1)

    plt.show()


#########################################################################
######################## DIFFUSION OF SIGNALS
##########################################################################

def bist_plot_2D(mode = "stab", varx = "AHL", vary = "IPTG", rangex = [-1,-4], rangey = [-2,2], nprec = 30): 
  '''
  Plot a 2d histogram with the endpoint of N*M  pairs of conditions that consists
  on logarithmically spaced rangex and rangy of varx and vary parameters
  each axis contains nprec points

  mode = "stab" then the program finds the steady state and plots 
  accordingly to the number and nature of the steady states

  if mode = "int" then the program integrates two high and low given conditions
  and plots accordingly to the number and nature of the steady states
  '''

  global par 
  par = getParametersFromMCMC()

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

  global pardict
  chem_temp = copy.deepcopy(ch) # copy of pardict to avoid its modification and restore it
  # at the end of the function call

  nprecx = nprec
  nprecy = nprec

  xarray = np.logspace(rangex[0],rangex[1],nprecx)
  yarray = np.logspace(rangey[0],rangey[1],nprecy)
  
  F1 = np.ndarray(shape=[nprecx,nprecy])# F1 will contain the values of the expression to plot
  F2 = np.ndarray(shape=[nprecx,nprecy])# F2 will contain the values of the expression to plot

  for ix,x in enumerate(xarray):
    for iy,y in enumerate(yarray):
      setattr(chem_temp,varx,x)
      setattr(chem_temp,vary,y)
      if mode == "stab":
        ss = steady_states(np.log10(par.alpha_X+eps),np.log10(par.alpha_X+par.beta_X-eps),1000,parr = par, chh = chem_temp)
        uniquess = []
        for state in ss:
          stateisunique = True
          for uniquestate in uniquess:
            if abs(state['X']-uniquestate['X'])<0.001:
              stateisunique = False
          if stateisunique:
            uniquess.append(state)
        ss = uniquess
        if len(ss) == 1:
          colorcoord = np.log(ss[0]['Y'])-np.log(ss[0]['X']) # negative -> red, positive -> green
          colorcoord = 0.5*(colorcoord+0.5) # linear transformation so neutral is at 0.5 of the colorscale
          colorcoord = max(colorcoord,0) # set limits of the color
          colorcoord = min(colorcoord,1) # set limits of the color
          # print "state",np.log(ss[0]['X']),np.log(ss[0]['Y']),np.log(ss[0]['X'])-np.log(ss[0]['Y'])
          F1[iy,ix] = colorcoord 
          F2[iy,ix] = 0
        elif len(ss) >1:

          # Choosing among the possible steady states
          colorcoord = -np.inf
          isstate = 100
          for isss,sss in enumerate(ss):
            if sss['Y'] > colorcoord: 
              colorcoord = sss['Y']
              isstate = isss
          print(varx, x, vary, y)
          colorcoord = np.log(ss[isstate]['Y'])-np.log(ss[isstate]['X']) # negative -> red, positive -> green
          colorcoord = 0.5*(colorcoord+0.5) # linear transformation so neutral is at 0.5 of the colorscale
          colorcoord = max(colorcoord,0) # set limits of the color
          colorcoord = min(colorcoord,1) # set limits of the color
          # print "state",np.log(ss[0]['X']),np.log(ss[0]['Y']),np.log(ss[0]['X'])-np.log(ss[0]['Y'])
          F1[iy,ix] = colorcoord 
          F2[iy,ix] = 1

          # Differnt color for multistability
          # F1[iy,ix] = 0 # whatever
          # F2[iy,ix] = 1

        else:
          print("No stability states found!")
      elif mode == "int":
        # Setting initial condition with a preintegration (overnight culture)
        setattr(chem_temp,'AHL',0)
        setattr(chem_temp,'aTc',0)
        setattr(chem_temp,'IPTG',1)
        [Xini,Yini,t] = integration(totaltime = integdict['totaltime'], dt=integdict['dt'],
                                    chh = chem_temp, result = 'end')
        setattr(chem_temp,varx,x)
        setattr(chem_temp,vary,y)
  
        [X,Y,t]=integration(totaltime = integdict['totaltime'], initcond = [Xini,Yini],
                            dt=integdict['dt'], chh = chem_temp, result = 'end')
        colorcoord = np.log(Y)-np.log(X) # negative -> red, positive -> green
        colorcoord = 0.5*(colorcoord+0.5) # linear transformation so neutral is at 0.5 of the colorscale
        colorcoord = max(colorcoord,0) # set limits of the color
        colorcoord = min(colorcoord,1) # set limits of the color
        F1[iy,ix] = colorcoord 
        F2[iy,ix] = 0

  cmap = cm.get_cmap('RedtoGreen')
  K1 = ax.pcolor(F1, cmap = 'RedtoGreen',rasterized=True)
  K1 = ax.pcolor(F2, cmap = 'AlphatoBlue',rasterized=True)
  K1.set_edgecolor('face')
  
  leftvalue = int(np.ceil(rangex[0]))
  rightvalue = int(np.floor(rangex[1]))
  if leftvalue>rightvalue:
    step = -1
    rightvalue = rightvalue - 1
  else: 
    step = 1
    rightvalue = rightvalue + 1
  labelvectorx = np.array(range(leftvalue,rightvalue,step))
  tickvectorx = (labelvectorx-labelvectorx[0])*nprecx/(labelvectorx[-1]-labelvectorx[0])

  leftvalue = int(np.ceil(rangey[0]))
  rightvalue = int(np.floor(rangey[1]))
  if leftvalue>rightvalue:
    step = -1
    rightvalue = rightvalue - 1
  else: 
    step = 1
    rightvalue = rightvalue + 1
  labelvectory = np.array(range(leftvalue,rightvalue,step))
  tickvectory = (labelvectory-labelvectory[0])*nprecy/(labelvectory[-1]-labelvectory[0])

  ax.set_xticks(tickvectorx)
  ax.set_yticks(tickvectory)
  ax.set_xticklabels(labelvectorx)
  ax.set_yticklabels(labelvectory)

  ax.set_xlabel("$\\mathrm{log}_{10}["+varx+"$] ({})".format(fll.units_dict[varx]),fontsize=20)
  ax.set_ylabel("$\\mathrm{log}_{10}["+vary+"$] ({})".format(fll.units_dict[vary]),fontsize=20)

  plt.tight_layout()
  plt.savefig("SS_GREEN_Expression.pdf")
  plt.show()


def create_diffusion_matrix(totaltime = 32, dt = 0.01, deg = 0.0, result = "traj", timelapse = 0.1,
                          Diff = 0.0025, xlength = 20, ylength = 20, name = 'IPTG', consource = 1000,  neighdist = 0.125,
                          initialconcentration = 0, consink = 0):
  '''
  Returns the matrix of gradients and also saves it in an external file for a given set of conditions. 
  This way avoids to simulate the diffusion every time. 

  The results is 2 files: a *.in file with the parameters of the simulation and a *.npy that contains the result of the simulations
  the matrix is a 2 dimension matrix (N x N, timepoints). In order to read the info from the NxN matrix a latticediff instance can be created, 
  it should also be accessible from a reshaping of the matrix. The .in file also contains the time datapoints

  Units are measured in hours, cm
  diffusion of 0.018 cm^2/h corresponds with 5E-6 cm^2/s as was measured for lactose Miyamoto 2018 Chem. Pharm. Bull.
  Based on the resemblance to Sucrose (molecular weight 342) and Glucose (m.w. 180) with IPTG (m.w. 238) and AHL (m.w. 297)
  (Note that there is a typo in Table 1 of the paper)

  diffusion of 0.0006 cm^2/h   corresponds with 1E-3 mm^2/min as measured in Basu 2005 Nature
  
  degradation of AHL and IPTG is set to 0.1 following Politi et al (2014) J. Bio. Eng. 8:5

  consource is the concentration at the border that can be kept constant to simulate sources and sinks
  '''

  foldername = 'diff_matrix/'

  dt = min(neighdist*neighdist/Diff/100.0,0.05) # to ensure a maximum change in concentration that is a tenth of the maximum concentration

  diffusiondata = {}
  diffusiondata['totaltime'] = totaltime
  diffusiondata['dt'] = dt
  diffusiondata['timelapse'] = timelapse
  diffusiondata['xlength'] = xlength
  diffusiondata['ylength'] = ylength
  diffusiondata['Diff'] = Diff
  diffusiondata['neighdist'] = neighdist
  diffusiondata['initialconcentration'] = initialconcentration
  diffusiondata['degradation'] = deg
  diffusiondata['time'] = []

  agarplate = lat.lattice(N=xlength+2,Ny = ylength+2)
  gradient = np.ones([(xlength+2)*(ylength+2)])*initialconcentration
 
  gradient_time = gradient # 3D array of the plate in time
 
  time = 0
  recordtime = 0
  consource_t = consource  # consource possibily changing
  consink_t = consink  # consink possibily changing

  while time<totaltime:

      # update of sink and source concentrations to degradation
      #consource_t -= consource_t*dt*deg
      #consink_t -= consink_t*dt*deg

      # consource_t = consource*(1-np.exp(-1.0e-4*time)) ## works for the diffusion of AHL
      # consource_t = consource*(1-np.exp(-1.0e-3*time)) ## works for the diffusion of AHL
      # consource_t = consource*(1-np.exp(-1.0e-2*time)) ## works for the diffusion of AHL

      diffusiontovoid = Diff*dt/neighdist/neighdist

      for cell in agarplate:
        # first update exterior. Condition is left column source, rest sink
        if agarplate[cell]['mpos'][1] == 0: # left column
          gradient[cell] = consource_t
        elif agarplate[cell]['mpos'][1] == xlength+1: # left column
          # gradient[cell] = consink_t
          gradient[cell] = gradient[cell] - gradient[cell]*Diff/neighdist/neighdist*dt
          # gradient[cell] = consource_t
        if agarplate[cell]['mpos'][0] == 0: # top column
          gradient[cell] = gradient[cell] - gradient[cell]*Diff/neighdist/neighdist*dt
          # gradient[cell] = consource_t
        elif agarplate[cell]['mpos'][0] == ylength+1: # bottom column
          gradient[cell] = gradient[cell] - gradient[cell]*Diff/neighdist/neighdist*dt
          # gradient[cell] = consource_t
      gradient = lat.diffuse(agarplate,gradient,Diff,deg,dt,neighbourdist = neighdist)
 
      time = time + dt

      if time>recordtime:

        diffusiondata['time'].append(time)
        gradient_time = np.vstack((gradient_time, gradient))

        print("timepoint: ", time, "  dimension:", gradient_time.shape)

        recordtime += timelapse

  id = random.randint(1,10000)
  string = '{}{}_{}_{}t{}'.format(foldername,name,consource,consink,totaltime)

  with open(string+".in", 'w') as outfile:  
    json.dump(diffusiondata,outfile)

  np.save(string+".npy",gradient_time)
  
  return gradient_time
  

#/////////////////////////////////////////////////////////////////////////
def integration_diffusion(totaltime = 18, dt = 0.1, recordtime = 4, initcond=[500,500,0], result = "traj",
                          Diff = 0.018, xlength = 20, ylength = 20,
                          sourceAHL = -10, sinkAHL=1E-10, sourceIPTG = -10, sinkIPTG = 1E-10,
                          timelapse = 0.1,
                          diffAHL = 'AHL100tt40id5327', diffIPTG ='IPTG10tt40id5714', diffaTc ='IPTG10tt40id5714',
                          directionAHL = "right", directionIPTG = 'down', directionaTc = 'right',
                          savematrix = True, section = False, stationarytime = 5,
                          makemovie = True):
  '''
  Integration of the bistable switch on a diffusing grid. Returns an array of vectors of X,Y,Z,t
  the diffusion grid contains is an array of length+2 elements allocating the sink and source at bouth sides
  the gradients are left to right by default, to rotate a gradient change values directionAHL, directionIPTG, dirctionaTc

  if a section = slice() is selected, then the result is the dynamics of the pattern profile for that given slice of the grid
  the precise details of the slide are incorporated in the condition
  '''

  if diffAHL:
    diffAHLdata, diffAHLmat = getDiffusionFromFile(diffAHL)
    diffAHLmat = rotategradient(diffAHLmat,directionAHL)
    CheckEnoughDiffusionTime(diffAHLdata,totaltime)
    if sourceAHL > 0: # scale AHL scales the gradient to the determined condition at the boundary
      indexsource = int(diffAHLdata['xlength'])+2 # skipping first row that is a sink
      # it contains xlength elements plus 2 sink columns
      factor = (sourceAHL-sinkAHL)/diffAHLmat[0][indexsource]
      print('factor',factor)
      diffAHLmat = diffAHLmat*factor + sinkAHL
      # print("After scale", diffAHLvec)

  if diffIPTG:
    diffIPTGdata, diffIPTGmat = getDiffusionFromFile(diffIPTG)
    CheckEnoughDiffusionTime(diffIPTGdata,totaltime)
    if sourceIPTG > 0: # scale IPTG scales the gradient to the determined condition at the boundary
      indexsource = int(diffIPTGdata['xlength'])+2 # skipping first row that is a sink
      print('idxsource', indexsource)
      factor = (sourceIPTG-sinkIPTG)/diffIPTGmat[0][indexsource]
      print('factor',factor)
      diffIPTGmat = diffIPTGmat*factor + sinkIPTG
    diffIPTGmat = rotategradient(diffIPTGmat,directionIPTG) # rotating after scaling to find indexsource easier


  if diffaTc:
    diffaTcdata, diffaTcmat = getDiffusionFromFile(diffaTc)
    diffaTcmat = np.zeros_like(rotategradient(diffaTcmat,directionaTc))
    CheckEnoughDiffusionTime(diffaTcdata,totaltime)

  print('IPTGt=0',diffIPTGmat[42],diffIPTGmat[43],diffIPTGmat[44])
  print('IPTGt=1',diffIPTGmat[84],diffIPTGmat[85],diffIPTGmat[46])

  # Units are measured in hours, cm
  # distance between squares is
  #neighdist = 0.125 Agar mask used
  neighdist = diffAHLdata['neighdist']

  degprot = 8.316 # this sets the relationship between flow (protein dynamical system) and time
  # since time is measured in hours, degprot corresponds with degradation rate of the protein in hours^{-1}
  # 8.316 h^{-1} corresponds with the 0.1386 min^{-1} reported in CHing-Hung Wu et al Bioinformatics 2011

  # loading fitted parameters of the GRN
  global par
  par = getParametersFromMCMC()

  # creating a plate to work with. It will not be used to diffuse because the diffusion
  # is calculated directly in the function create_diffusion to speed up computation
  agarplate = lat.lattice(N=xlength+2,Ny = ylength+2)
  # +2 to accomodate boundary conditions
  AHLgradient = np.zeros([(xlength+2)*(ylength+2)])
  IPTGgradient = np.ones_like(AHLgradient)*0.0
  aTcgradient = np.ones_like(AHLgradient)*0.0

  # Initializing cell expression (ovenright conditions) GREEN
  # chem.AHL = 0.0
  # chem.IPTG = 1.0
  # chem.aTc = 0.0

 # Initializing cell expression (ovenright conditions) RED
  chem.AHL = 10.0
  chem.IPTG = 1.0
  chem.ATC = 0.0

  Xini,Yini,t = integration(totaltime = 1000.0, dt = 0.01,initcond=[500,500],
    result = "final", parr = par, chh = chem)

  print("Xini, Yini", Xini, Yini)

  framenamelist = []
  if timelapse == None:
    timelapse = 0.1
  time = 0
  Xarray = np.ones_like(AHLgradient)*Xini
  Yarray = np.ones_like(AHLgradient)*Yini

  # Setting expression in borders to zero (since they have no cells)
  for cell in agarplate:
    if agarplate[cell]['mpos'][1] == 0 or agarplate[cell]['mpos'][1] == xlength+1:
        Xarray[cell] = 0
        Yarray[cell] = 0
    if agarplate[cell]['mpos'][0] == 0 or agarplate[cell]['mpos'][0] == ylength+1:
        Xarray[cell] = 0
        Yarray[cell] = 0

  difftimecount = 1 # this is the time counter to get times from the loaded diffusion matrices
  nextdiffusetime = diffAHLdata['time'][difftimecount] # next time to update morphogen values
  AHLgradient = diffAHLmat[0]
  IPTGgradient = diffIPTGmat[0]
  aTcgradient = diffaTcmat[0]

  while time<totaltime:
      print(time)
      if time>nextdiffusetime:
        AHLgradient = diffAHLmat[difftimecount]
        IPTGgradient = diffIPTGmat[difftimecount]
        aTcgradient = diffaTcmat[difftimecount]
        difftimecount += 1

      for cell in agarplate:
        # for each interior cell
        if agarplate[cell]['mpos'][1] != 0 and agarplate[cell]['mpos'][1] != xlength+1:
          if agarplate[cell]['mpos'][0] != 0 and agarplate[cell]['mpos'][0] != ylength+1:
            chem.AHL = AHLgradient[cell]
            chem.IPTG = IPTGgradient[cell]
            chem.aTc = aTcgradient[cell] 
            flow_X,flow_Y = flow(Xarray[cell],Yarray[cell],chem, par) 
            if (time>stationarytime):
              ## change in protein number from the model using Euler integration
              Xarray[cell] = Xarray[cell]+dt*flow_X*degprot
              Yarray[cell] = Yarray[cell]+dt*flow_Y*degprot

      if time>recordtime:
        # print 'Xarray', Xarray  
        # print 'Yarray', Yarray
        if not section:
            lat.printlattices(agarplate,[np.log10(AHLgradient),np.log10(IPTGgradient),Xarray,Yarray],
                limcolors = [[-3,0],[-2,1],[400,800],[250,1000]],
                cmaps = ['viridis','viridis','Reds','Greens'],labels = ['log AHL','log IPTG','mCherry','GFP'],
                cols = 2, save = 'diffusionimages/diffusion{0:.2f}.png'.format(time), timestamp = str(time), show = False)
            if makemovie:
              framenamelist.append('diffusionimages/diffusion{0:.2f}.png'.format(time))
        else: # section = True
            idxcell = 4 
            sl = slice(idxcell,-1,xlength) #
            # sl = slice((idxcell-1)*xlength,idxcell*xlength)
            plt.plot(neighdist*np.arange(xlength),Xarray[sl])
        # lat.printlattice(agarplate, )
        # lat.printlattice(agarplate, Xarray, limcolor = [400,800], cmap = 'Reds')
        # lat.printlattice(agarplate, Yarray, limcolor = [250,1000], cmap = 'Greens')
        recordtime += timelapse
      time = time + dt

  if not section:
    lat.printlattices(agarplate,[np.log10(AHLgradient),np.log10(IPTGgradient),Xarray,Yarray],
            limcolors = [[-3,0],[-2,1],[400,800],[250,1000]],
            cmaps = ['viridis','viridis','Reds','Greens'],labels = ['log AHL','log IPTG','mCherry','GFP'],
            cols = 2, save = 'diffusionimages/diffusion{0:.2f}.png'.format(time), timestamp = str(time))
  else:
    plt.xlabel('distance (cm)')
    plt.ylabel('mCherry')

  if makemovie:
    frame0 = cv2.imread(framenamelist[0])
    height, width, layers = frame0.shape
    print('checpoint 1')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('moviediffusion.mp4', fourcc, 10,(width,height))


    for frame in framenamelist:
      print('checpoint 2 ', frame)
      imag = cv2.imread(frame)
      video.write(imag)

    cv2.destroyAllWindows()
    video.release()


#/////////////////////////////////////////////////////////////////////////
def integration_diffusion_comparison(totaltime = 18, dt = 0.1, recordtime = 4, initcond=[500,500,0], result = "traj",
                          Diff = 0.01, xlength = 20, ylength = 20, sourceAHL = -1, sourceIPTG = -1, 
                          sinkAHL = 1E-10, sinkIPTG = 1E-10,
                          timelapse = 1.0,
                          diffAHL = 'IPTG10tt32id924deg0.1.npy', diffIPTG ='IPTG10tt40id5714', diffaTc ='IPTG10tt40id5714',
                          constantIPTG = -1, constantAHL = -1, 
                          directionAHL = "right", directionIPTG = 'down', directionaTc = 'right',
                          savematrix = True, section = False, stationarytime = 5,
                          makemovie = False):
  '''
  Same function as integration_diffusion but using two different initial conditions and comparing resulting profile
  stationarytime freezes the cells to respond for a given time (hours)
  '''

  if diffAHL:
    diffAHLdata, diffAHLmat = getDiffusionFromFile(diffAHL)
    diffAHLmat = rotategradient(diffAHLmat,directionAHL)
    CheckEnoughDiffusionTime(diffAHLdata,totaltime)
    if sourceAHL > 0: # scale AHL scales the gradient to the determined condition at the boundary
      indexsource = int(diffAHLdata['xlength'])+2 # skipping first row that is a sink
      # it contains xlength elements plus 2 sink columns
      factor = (sourceAHL-sinkAHL)/diffAHLmat[0][indexsource]
      print('factor',factor)
      diffAHLmat = diffAHLmat*factor + sinkAHL
      # print("After scale", diffAHLvec)

  if diffIPTG:
    diffIPTGdata, diffIPTGmat = getDiffusionFromFile(diffIPTG)
    CheckEnoughDiffusionTime(diffIPTGdata,totaltime)
    if sourceIPTG > 0: # scale IPTG scales the gradient to the determined condition at the boundary
      indexsource = int(diffIPTGdata['xlength'])+2 # skipping first row that is a sink
      print('idxsource', indexsource)
      factor = (sourceIPTG-sinkIPTG)/diffIPTGmat[0][indexsource]
      print('factor',factor)
      diffIPTGmat = diffIPTGmat*factor + sinkIPTG
    diffIPTGmat = rotategradient(diffIPTGmat,directionIPTG) # rotating after scaling to find indexsource easier


  if diffaTc:
    diffaTcdata, diffaTcmat = getDiffusionFromFile(diffaTc)
    diffaTcmat = np.zeros_like(rotategradient(diffaTcmat,directionaTc))
    CheckEnoughDiffusionTime(diffaTcdata,totaltime)

# Units are measured in hours, cm
# distance between squares is
#neighdist = 0.125 mask used in the experiment
  neighdist = diffAHLdata['neighdist']

# degradation rate of the gradient is omited and degradation rate of the proteins of the switch

  degprot = 8.316 # this sets the relationship between flow (protein dynamical system) and time
  #degprot = 4.0 # this sets the relationship between flow (protein dynamical system) and time
  # since time is measured in hours, degprot corresponds with degradation rate of the protein in hours^{-1}
  # 8.316 h^{-1} corresponds with the 0.1386 min^{-1} reported in CHing-Hung Wu et al Bioinformatics 2011

  global par
  par = getParametersFromMCMC()
  print("K_IPTG", par.K_IPTG)

  # creating a plate to work with. It will not be used to diffuse because the diffusion
  # is calculated directly in the function create_diffusion to speed up computation
  agarplate = lat.lattice(N=xlength+2,Ny = ylength+2)
  # +2 to accomodate boundary conditions
  AHLgradient = np.zeros([(xlength+2)*(ylength+2)])
  IPTGgradient = np.ones_like(AHLgradient)*0.0
  aTcgradient = np.ones_like(AHLgradient)*0.0

  # AHLgradient[-1] = 0

  Xarrays = [[],[]]# the different components correspond to different conditions we want to compare in the ends
  Yarrays = [[],[]]

  # Initializing cell expression (ovenright conditions) GREEN
  chem.AHL = 0.0
  chem.IPTG = 1.0
  chem.aTc = 0.0

  Xini,Yini,t = integration(totaltime = 1000.0, dt = 0.01,initcond=[500,500],
    result = "final", parr = par, chh = chem)

  print("Xini, Yini", Xini, Yini)

  if timelapse == None:
    timelapse = 0.1
  time = 0
  Xarrays[0] = np.ones_like(AHLgradient)*Xini
  Yarrays[0] = np.ones_like(AHLgradient)*Yini

 # Initializing cell expression (ovenright conditions) RED
  chem.AHL = 10.0
  chem.IPTG = 1.0
  chem.ATC = 0.0

  Xini,Yini,t = integration(totaltime = 1000.0, dt = 0.01,initcond=[500,500],
    result = "final", parr = par, chh = chem)

  print("Xini, Yini", Xini, Yini)

  if timelapse == None:
    timelapse = 0.1
  time = 0
  Xarrays[1] = np.ones_like(AHLgradient)*Xini
  Yarrays[1] = np.ones_like(AHLgradient)*Yini

  print("Xiniag, Yiniag", Xarrays[1][100], Yarrays[1][100])

  # Setting expression in borders to zero (since they have no cells)
  for cell in agarplate:
    if agarplate[cell]['mpos'][1] == 0 or agarplate[cell]['mpos'][1] == xlength+1:
        Xarrays[0][cell] = 0
        Yarrays[0][cell] = 0
        Xarrays[1][cell] = 0
        Yarrays[1][cell] = 0
    if agarplate[cell]['mpos'][0] == 0 or agarplate[cell]['mpos'][0] == ylength+1:
        Xarrays[0][cell] = 0
        Yarrays[0][cell] = 0
        Xarrays[1][cell] = 0
        Yarrays[1][cell] = 0
  
  difftimecount = 1 # this is the time counter to get times from the loaded diffusion matrices
  nextdiffusetime = diffAHLdata['time'][difftimecount] # next time to update morphogen values
  AHLgradient = diffAHLmat[0]
  IPTGgradient = diffIPTGmat[0]
  aTcgradient = diffaTcmat[0]

  if makemovie:
    framenamelist = []

  while time<totaltime:
    print(time)
    if time>nextdiffusetime:
        AHLgradient = diffAHLmat[difftimecount]
        IPTGgradient = diffIPTGmat[difftimecount]
        aTcgradient = diffaTcmat[difftimecount]
        difftimecount += 1

    if time>=recordtime:
        # print 'Xarrays[0]', Xarrays[0].tolist()  
        # print 'Yarrays[0]', Yarrays[0].tolist()
        # print 'Xarrays[1]', Xarrays[1].tolist()  
        # print 'Yarrays[1]', Yarrays[1].tolist()
        if not section:
            # cond1color_Red = xtocolor(Xarrays[0],'Reds',[400,800])
            cond1color_Red = xtocustomcolor(Xarrays[0],[1,0,0],[400,700])
            #cond1color_Green = xtocolor(Yarrays[0],'Greens',[250,1000])
            cond1color_Green = xtocustomcolor(Yarrays[0],[0,0.7,0],[270,500])
            cond1color = mixcolor(cond1color_Red,cond1color_Green,'additive')
            # cond2color_Red = xtocolor(Xarrays[1],'Reds',[400,800])
            cond2color_Red = xtocustomcolor(Xarrays[1],[1,0,0],[400,700])
            #cond2color_Green = xtocolor(Yarrays[1],'Greens',[250,1000])
            cond2color_Green = xtocustomcolor(Yarrays[1],[0,0.7,0],[270,500])
            cond2color = mixcolor(cond2color_Red,cond2color_Green,'additive')
            condmixcolor = mixcolor(cond1color,cond2color,'additive')
            print('XY1', Xarrays[0][100], Yarrays[0][100])
            print('cond1color_Red',cond1color_Red[100])
            print('cond1color_Green',cond1color_Green[100])
            print('c1',cond1color[100])
            print('XY2', Xarrays[1][100], Yarrays[1][100])
            print('cond2color_Red',cond2color_Red[100])
            print('cond2color_Green',cond2color_Green[100])
            print('c2',cond1color[100])

            lat.printlattices_comp(agarplate,[np.log10(AHLgradient),np.log10(IPTGgradient),cond1color,cond2color, condmixcolor],limcolors = [[-3,0],[-3,0],[300,600],[280,700],[290,700]],
                cmaps = ['viridis','viridis','custom','custom','custom'],
                labels = ['log AHL','log IPTG','Inital Green','Inital Red','Inital Mixed'],
                cols = 2, save = 'diffusionimages/diffusion{0:.2f}.png'.format(time), timestamp = str(time))
            if makemovie:
              framenamelist.append('diffusionimages/diffusion{0:.2f}.png'.format(time))

        else: # section = True
            idxcell = 4 
            sl = slice(idxcell,-1,xlength) #
            # sl = slice((idxcell-1)*xlength,idxcell*xlength)
            print('Xarrays',Xarrays[0])
            plt.plot(neighdist*np.arange(xlength+2),getCenterSliceMatrix(Xarrays[0],rotate = True))
        # lat.printlattice(agarplate, )
        # lat.printlattice(agarplate, Xarray, limcolor = [400,800], cmap = 'Reds')
        # lat.printlattice(agarplate, Yarray, limcolor = [250,1000], cmap = 'Greens')
        recordtime += timelapse

    for cell in agarplate:
        # for each interior cell
      if agarplate[cell]['mpos'][1] != 0 and agarplate[cell]['mpos'][1] != xlength+1:
       if agarplate[cell]['mpos'][0] != 0 and agarplate[cell]['mpos'][0] != ylength+1:
        chem.AHL = AHLgradient[cell]
        chem.IPTG = IPTGgradient[cell]
        chem.aTc = aTcgradient[cell]
        for ii in range(len(Xarrays)): # for each initial condition we want to compare 
            flow_X,flow_Y = flow(Xarrays[ii][cell],Yarrays[ii][cell],chem,par)              
        ## change in protein number from the model
            if (time>stationarytime):
                Xarrays[ii][cell] = Xarrays[ii][cell]+dt*flow_X*degprot
                Yarrays[ii][cell] = Yarrays[ii][cell]+dt*flow_Y*degprot

    time = time + dt

  if not section:
            lat.printlattices_comp(agarplate,[np.log10(AHLgradient),np.log10(IPTGgradient),cond1color,cond2color, condmixcolor],limcolors = [[-1,2],[-2,1],[400,800],[250,1000],[250,1000]],
                cmaps = ['viridis','viridis','custom','custom','custom'],labels = ['log AHL','log IPTG','Init Green','Init Red','Init Mix'], cols = 2, save = 'diffusion'+str(time)+'.png', timestamp = str(time))
  else:
    plt.xlabel('distance (cm)')
    plt.ylabel('mCherry')
    plt.savefig('slice.png')
    plt.show()

  if makemovie:
    frame0 = cv2.imread(framenamelist[0])
    height, width, layers = frame0.shape
    print('checkpoint 1')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('moviediffusion.mp4', fourcc, 10,(width,height))


    for frame in framenamelist:
      print('checkpoint 2 ', frame)
      imag = cv2.imread(frame)
      video.write(imag)

    cv2.destroyAllWindows()
    video.release()

##########################################

def integration_diffusion_1D(totaltime = 20, dt = 0.1, recordtimelist = [3,4,5,6,8,9,10,11,12], initcond=[500,500,0], result = "traj",
                          Diff = 0.018, xlength = 20, ylength = 20, sourceAHL = -1, sourceIPTG = -1,
                          sinkAHL = 1E-10, sinkIPTG = 1E-10,
                          timelapse = 1.0,
                          diffAHL = 'IPTG10tt32id924deg0.1.npy', diffIPTG ='IPTG10tt40id5714', diffaTc =False,
                          constantIPTG = -1, constantAHL = -1, 
                          fixIPTG = 0, fixAHL = 0,
                          directionAHL = "right", directionIPTG = 'right', directionaTc = 'right',
                          savematrix = True, section = False, stationarytime = 1,
                          parameterset = '40000', # parameter set to use from the MCMC result
                          ):
  ''' Analogous to the 2D grid integrators but reduced to 1D to to study velocity of patterning and boundary formation
  the scale function scales the concentration of the morphogen to capture the fact that the 
  boundary concentration may be different.

  since the diffusion equation is linear, any linear transformation of the gradient is also a solution of the equation
  therefore we can scale and shift the gradient without the need of redoing the simulation. In particular if sourceAHL is set > 0, then a linear transformation to
  fix the source of the gradient at sourceAHL and the sinks at sinkAHL.
  '''

  if diffAHL:
    diffAHLdata, diffAHLmat = getDiffusionFromFile(diffAHL)
    gendiffdata, gendiffmat = diffAHLdata, diffAHLmat  # This is a dummy container
    diffAHLmat = rotategradient(diffAHLmat,directionAHL)
    CheckEnoughDiffusionTime(diffAHLdata,totaltime)
    print("Before Scale", getCenterSlice(diffAHLmat))
    diffAHLvec = getCenterSlice(diffAHLmat)
    if sourceAHL > 0: # scale AHL scales the gradient to the determined condition at the boundary

      indexsource = int(diffAHLdata['xlength'])+2 # skipping first row that is a sink
      # it contains xlength elements plus 2 sink columns
      factor = (sourceAHL-sinkAHL)/diffAHLmat[0][indexsource]
      print('factor',factor)
      diffAHLvec = diffAHLvec*factor + sinkAHL
      print("After scale", diffAHLvec)


  if diffIPTG:
    diffIPTGdata, diffIPTGmat = getDiffusionFromFile(diffIPTG)
    CheckEnoughDiffusionTime(diffIPTGdata,totaltime)
    if sourceIPTG > 0: # scale IPTG scales the gradient to the determined condition at the boundary
      indexsource = int(diffIPTGdata['xlength'])+2 # skipping first row that is a sink
      print('idxsource', indexsource)
      factor = (sourceIPTG-sinkIPTG)/diffIPTGmat[0][indexsource]
      print('factor',factor)
      diffIPTGmat = diffIPTGmat*factor + sinkIPTG
    diffIPTGmat = rotategradient(diffIPTGmat,directionIPTG) # rotating after scaling to find indexsource easier
    diffIPTGvec = getCenterSlice(diffIPTGmat)
    # diffIPTGvec = np.ones_like(diffIPTGvec)*0.125


  if diffaTc:
    diffaTcdata, diffaTcmat = getDiffusionFromFile(diffaTc)
    diffaTcmat = rotategradient(diffaTcmat,directionaTc)
    CheckEnoughDiffusionTime(diffaTcdata,totaltime)
    diffaTcvec = getCenterSlice(diffaTcmat)
  else:
    diffaTcvec = np.zeros_like(diffAHLvec)  



# Units are measured in hours, cm
# distance between squares is
  neighdist = 0.125 # Experimental masking in the agar gel

# degradation rate of the gradient is omited and degradation rate of the proteins of the switch
# is measured to be around 5 hours so 

  degprot = 8.316 # this sets the relationship between flow (protein dynamical system) and time
  # since time is measured in hours, degprot corresponds with degradation rate of the protein in hours^{-1}
  # 8.316 h^{-1} corresponds with the 0.1386 min^{-1} reported in CHing-Hung Wu et al Bioinformatics 2011

  global par
  par = getParametersFromMCMC(condition = parameterset)

  # creating a plate to work with. It will not be used to diffuse because the diffusion
  # is calculated directly in the function create_diffusion to speed up computation
  # AHLgradient[-1] = 0

  Xarray = np.zeros_like(diffAHLvec[0])# the different components correspond to different conditions we want to compare in the ends
  Yarray = np.zeros_like(diffIPTGvec[0])# the different components correspond to different conditions we want to compare in the ends

  # Initializing cell expression (ovenright conditions) GREEN
  # Initializing cell expression (ovenright conditions) GREEN
  chem.AHL = 0.0
  chem.IPTG = 1.0
  chem.aTc = 0.0

 # Initializing cell expression (ovenright conditions) RED
  # chem.AHL = 10.0
  # chem.IPTG = 1.0
  # chem.ATC = 0.0  

  print ('init AHL', chem.AHL)
  print ('init IPTG', chem.IPTG)
  print ('init ATC', chem.aTc)

  Xini,Yini,t = integration(totaltime = 1000.0, dt = 0.01, initcond = [500,500],
    result = "final", parr = par, chh = chem)

  print("Xini, Yini", Xini, Yini)

  if timelapse == None:
    timelapse = 0.1
  time = 0
  Xarray = np.ones_like(diffAHLvec[0])*Xini
  Yarray = np.ones_like(diffIPTGvec[0])*Yini

  time = 0
  inextrecordtime = 0
  Xarrays = []
  Yarrays = []



  difftimecount = 1 # this is the time counter to get times from the loaded diffusion matrices
  nextdiffusetime = gendiffdata['time'][difftimecount] # next time to update morphogen values
  AHLgradient = diffAHLvec[0]
  IPTGgradient = diffIPTGvec[0]
  aTcgradient = diffaTcvec[0]

  while time<totaltime:
      if time>nextdiffusetime:
        AHLgradient = diffAHLvec[difftimecount]
        IPTGgradient = diffIPTGvec[difftimecount]
        aTcgradient = diffaTcvec[difftimecount]
        difftimecount += 1

      for icell in range(len(Xarray)):        
        chem.AHL = AHLgradient[icell]
        chem.IPTG = IPTGgradient[icell]
        chem.aTc = aTcgradient[icell] 
        flow_X,flow_Y = flow(Xarray[icell],Yarray[icell],chem, par) 
        ## change in protein number from the model
        if (time>stationarytime):
          Xarray[icell] = Xarray[icell] + dt*flow_X*degprot
          Yarray[icell] = Yarray[icell] + dt*flow_Y*degprot

      if (inextrecordtime<(len(recordtimelist))):
        if time>recordtimelist[inextrecordtime]:
          inextrecordtime += 1  
          print('time',time)
          Xarrays.append(np.array(Xarray))
          Yarrays.append(np.array(Yarray))

          # plt.plot(np.arange(len(Xarray))*neighdist,Xarray,'ro-')
          # plt.plot(np.arange(len(Yarray))*neighdist,Yarray,'go-')

          # plt.plot(np.arange(len(Yarray))*neighdist,AHLgradient,'go-')
          # plt.plot(np.arange(len(Yarray))*neighdist,IPTGgradient,'go-')

          #plt.show(block=True)
          # lat.printlattice(agarplate, )
          # lat.printlattice(agarplate, Xarray, limcolor = [400,800], cmap = 'Reds')
          # lat.printlattice(agarplate, Yarray, limcolor = [250,1000], cmap = 'Greens')
          #recordtime += timelapse
      time = time + dt
  # plt.xlim([0,5])
  # plt.ylim([250,850])
  # plt.savefig('boundary_{}.pdf.'.format(fixAHL))
  # plt.clf()
  return Xarrays,Yarrays

def comparePattern1D(**kwargs):

  '''
  Function to plot several 1-D patterns by calling integration_diffusion_1D to a set of conditions
  '''

  timeintegrations = np.linspace(3,14,12)
  pixeltocm = 0.125 # relationship in the agar plate
  # removetimeIPTGs = [8,10,12]

  conditions = [{'diffAHL' : 'doubleIPTG_1_0t20', 'diffIPTG' : 'constantIPTG_0.15_0.15t32', 'condition':0.15},
                  {'diffAHL' : 'doubleIPTG_1_0t20', 'diffIPTG' : 'constantIPTG_1_1t32', 'condition':1 }]

  for icond, cond in enumerate(conditions):

    # finalAHL = np.log10(initAHLs[iIPTG]) + length*np.cos(np.arctan(slope))
    # finalAHL = np.power(10,finalAHL)
    # finalIPTG = np.log10(initIPTGs[iIPTG]) + length*np.sin(np.arctan(slope))
    # finalIPTG = np.power(10,finalIPTG)

    positionArray = np.array([])
    widthArray = np.array([])
    timesArray = np.array([])

    # print "Initial point: ", np.log10(initAHLs[iIPTG]), np.log10(initIPTGs[iIPTG])
    # print "Final point: ", np.log10(finalAHL), np.log10(finalIPTG)


    Xarrays, Yarrays = integration_diffusion_1D(totaltime = timeintegrations[-1]+0.2, 
                                recordtimelist = timeintegrations,
                                # IPTGzero = 0, AHLzero = 0, sinkAHL = 0,
                                # sourceIPTG = finalIPTG, sourceAHL = finalAHL,
                                # sinkIPTG = initIPTGs[iIPTG], sinkAHL = initAHLs[iIPTG],
                                sourceIPTG = -1,sourceAHL = -1,
                                #fixAHL = fixAHL, # diffAHL = False,#IPTGzero = fixIPTG,
                                # removetimeIPTG = removetimeIPTG,
                                diffAHL = cond['diffAHL'],diffIPTG = cond['diffIPTG'],
                                **kwargs)
    for iY,Yarray in enumerate(Yarrays): # for each time point spatial pattern
      pos,sigma = getPositionBoundary(Yarray)
      #print('Yarray',Yarray)
      #print('pos',pos)

      if pos>0: # if there is a bounary, add it to the boundaries to plot, otherwise ignore it
        positionArray = np.append(positionArray,pos)
        widthArray = np.append(widthArray,sigma)
        timesArray = np.append(timesArray,timeintegrations[iY])

    print('posarray', positionArray)
    print('timesArray', timesArray)
    plt.plot(timesArray, positionArray,'o-' , color = sbcolorcycle[icond], label = '$IPTG$ = '+str(cond['condition'])+' $\\mu$M')
    B = np.array([timesArray,positionArray])
    np.savetxt('array'+str(cond['condition']),B.T)

  plt.xlabel('time (h)')
  plt.ylabel('boundary position (squares)')
  plt.legend()
  plt.tight_layout()
  plt.savefig('timewindow_fixIPTG'+'sourceAHL'+str(1)+'.pdf')
  plt.show() 

  return 

def getPositionBoundary(xarray,method='square'):
  '''
  Compute the position and width of the boundary in a xarray
  method decides if there is a linear interpolation or the absolute squares are given 
  '''

  leftLimit = xarray[0]
  rightLimit = xarray[-1]
  if np.abs(leftLimit-rightLimit)<200: # completely arbitrary limit to remove small changes in concentrations that do not constitute a pattern
    return -2,-2 # There is no boundary because there is not a large difference between activated an inactivated

  middleconc = (xarray[0]+xarray[-1])/2.0

  if method == 'linear':
    idxmiddle = (np.abs(xarray - middleconc)).argmin() 
    if ((idxmiddle == 0) or (idxmiddle == len(xarray-1))):
      return -1,-1 # there is no reliable boundary because it is too close to the sides of the integration zone

    idxarray = [idxmiddle-1,idxmiddle,idxmiddle+1]
    intensityarray = [xarray[idxmiddle-1],xarray[idxmiddle],xarray[idxmiddle+1]]
    slope,intercept,a,b,c = stats.linregress(idxarray,intensityarray)
    # print 'Slope, intercept', slope, intercept
    boundarypos = (middleconc - intercept)/slope
    boundaryerror = np.abs((leftLimit-rightLimit))/slope
    # print 'dist, width', np.abs((leftLimit-rightLimit)), boundaryerror
    return boundarypos, boundaryerror

  else: 
    idxmiddle = np.argmax(xarray>middleconc)
    print('xarray', xarray)
    print('middleconc', middleconc)
    print ('idxmiddle',idxmiddle)
    if ((idxmiddle == 0) or (idxmiddle == len(xarray-1))):
      return -1,-1 # there is no reliable boundary because it is too close to the sides of the integration zone    
    return idxmiddle,1  


def rotategradient(A,direction):
    ''' Rotate a grid A created by integration_diffusion.
    Direction indicated the diffusing direction 
    can be 'left','up','down' and assumes that the original array diffuses from left to right
    '''
    print("rotating gradient...")
    if direction == "left":
            N = int(np.sqrt(len(A[0])))
            rotated_A = np.zeros_like(A)
            for ig,gradient in enumerate(A): # for each timepoint
                gradient = np.reshape(gradient,(N,N)) 
                for irow,row in enumerate(gradient):
                    gradient[irow,:] = row[::-1] 
                rotated_A[ig] = gradient.flatten()
            return rotated_A

    if direction == "up":
            N = int(np.sqrt(len(A[0])))
            rotated_A = np.zeros_like(A)
            for ig,gradient in enumerate(A): # for each timepoint
                gradient = np.reshape(gradient,(N,N)) 
                gradient = gradient.T
                rotated_A[ig] = gradient.flatten()
            return rotated_A

    if direction == "down":
            N = int(np.sqrt(len(A[0])))
            rotated_A = np.zeros_like(A)
            for ig,gradient in enumerate(A): # for each timepoint
                gradient = np.reshape(gradient,(N,N)) 
                for irow,row in enumerate(gradient):
                    gradient[irow,:] = row[::-1] 
                gradient = gradient.T
                rotated_A[ig] = gradient.flatten()
            return rotated_A

    if direction == "right":
        return A
    else:
        print("Unknown direction!!")
        return 0


##############################################################################
### OTHER AUXILIAR FUNCTIONS
###############################################################################

def getCovarianceFromLangevin(Jacobian, Diffusion):
  '''
  given a Drift vector and a Diffusion diffusion matrix, returns the covariance
  predicted for the equilibrium Gaussian Cov in the case of a linearized equation around a steady state
  In the steady state the Drift is zero and the curvature is given by the Jacobian
  ignoring the rest of the terms we can write the linear multivariate F-P equation
  which has a covariance (van Kampen book eq.6.9) Diffusion = -Jacobian*Cov - Cov*Jacobian.T
  I restrucutred by hand this equation to transform it into a linear problem Diff_vec = K Cov_vec
  where Diff_vec is the 4d vector (flattened) form of Diff, and the same for Cov_vec
  K is a 4x4 matrix with elements
  '''

  J = -Jacobian

  print('J', J)
  Diff_vec = [Diffusion[0][0], Diffusion[0][1], Diffusion[1][0], Diffusion[1][1]]

  K=np.array([[2*J[0][0], J[0][1], J[0][1], 0],
    [J[1][0], J[1][1]-J[0][0], 0, J[0][1]],
    [J[1][0], 0, J[1][1]-J[0][0], J[0][1]],
    [0, J[1][0] , J[1][0], 2*J[1,1] ]])

  print('K', K)

  Cov_vec = np.dot(np.linalg.inv(K),Diff_vec)

  Cov = [[Cov_vec[0],Cov_vec[1]],
        [Cov_vec[2], Cov_vec[3]]]

  print('C', Cov)

  return Cov


def getMultivariateGaussianFromLangevin(Point, parr = par, chh = ch):
  ''' Return the equiblirium linear approximation Gaussian at the steady state Point.
  It has the potential use to compare stochastic models with flow data
  '''

  Jacobian = getNumericalJacobian(Point[0],Point[1],chh=ch,parr = par)
  fx, fy, gx, gy = flowWithNoiseIntensity(Point[0],Point[1],chh = ch, parr = par)
  Diffusion = np.zeros([2,2])
  Diffusion[0,0] = gx*gx
  Diffusion[1,1] = gy*gy
  Covariance = getCovarianceFromLangevin(Jacobian, Diffusion)
  return stats.multivariate_normal(mean=Point, cov=Covariance)


def getParametersFromMCMC(condition = '30000', verbose = False):
  ''' Load parameters from PyDream output
  verbose prints the list of parameters with its 95% CI
  '''
    

  localpar = param()  
  # subfolder = 'fitsApril2019/'
  subfolder = ''
  chain = 0
  data1 = np.load('dream/'+subfolder+'bsfit_sampled_params_chain'+str(chain)+'_'+condition+'.npy')
  logpsdata1 = np.load('dream/bsfit_logps_chain'+str(chain)+'_'+condition+'.npy')

  lklh_list = np.array([l[0] for l in logpsdata1])

  sorted_idxs = lklh_list.argsort()
  data1_sorted = data1[sorted_idxs]
  bestpars = data1_sorted[-1]
  ibestlklh = -1


  localpar.alpha_X = 10**bestpars[orderedpars['alpha_X']]
  localpar.alpha_Y = 10**bestpars[orderedpars['alpha_Y']]
  localpar.beta_X = 10**bestpars[orderedpars['beta_X']]
  localpar.beta_Y = 10**bestpars[orderedpars['beta_Y']]
  localpar.K_IPTG = 10**bestpars[orderedpars['K_IPTG']]
  localpar.K_AHL = 10**bestpars[orderedpars['K_AHL']]
  localpar.K_LACI = 10**bestpars[orderedpars['K_LACI']]
  localpar.K_TETR = 10**bestpars[orderedpars['K_TETR']]
  localpar.K_ATC = 10**bestpars[orderedpars['K_ATC']]
  localpar.n_LACI = 10**bestpars[orderedpars['n_LACI']]
  localpar.n_AHL = 10**bestpars[orderedpars['n_AHL']]
  localpar.n_TETR = 10**bestpars[orderedpars['n_TETR']]

  localpar.delta_X = 1.0
  localpar.delta_Y = 1.0

  if verbose:
    trim = 10000
    data1_sorted = data1_sorted[trim:]
    N = len(data1_sorted)
    N_95 = int(N/95)
    for p in orderedpars:
      try:
        ordered_values = data1_sorted[:,orderedpars[p]]
        ordered_values_95 = ordered_values[N_95:]
        print("Par: {}  Median:{}  CI = [{},{}]".format(p, 10**np.median(ordered_values_95), 10**np.min(ordered_values_95), 10**np.max(ordered_values_95)))
      except:
        print("Parameter {} not analyzed".format(p))
        pass

  return localpar


def getDiffusionFromFile(filename):
  ''' Load the diffusion matrix previously created by integration_diffusion
  '''

  foldername = 'diff_matrix/'
  with open(foldername+filename+'.in') as file:
    diffdata = json.load(file)
    diffmat = np.load(foldername+filename+'.npy')
    return diffdata,diffmat

def CheckEnoughDiffusionTime(diffdata,totaltime):
  ''' Check that load data from a previous diffusion integration contains 
  time points beyond a threshold totaltime
  '''

  if diffdata['totaltime']<totaltime:
    raise ValueError('Total time of morphogen gradient is not long enough')

def getCenterSlice(diffmat):
  ''' Returns a time series of the central slice (central row) of a diffusion matrix time series
  '''

  NumberOfRows = int(np.sqrt(len(diffmat[0]))) # assuming diffusion always occurs in a square lattice
  CenterIndex = int(NumberOfRows/2) # middle row
  sliceVectorTimeSeries = []

  for ig,gradient in enumerate(diffmat): # for each timepoint
    gradient = np.reshape(gradient,(NumberOfRows,NumberOfRows)) 
    #print("CenterIndex",CenterIndex)
    sliceVectorTimeSeries.append(gradient[CenterIndex])
  return np.array(sliceVectorTimeSeries)    

for N in range(1):
  print("loglikelihood: ", loglikelihood(fll.df,[par.alpha_X, par.alpha_Y,
        par.beta_X, par.beta_Y, par.K_IPTG, par.K_AHL, par.K_LACI, par.K_TETR,
        par.K_ATC, par.n_LACI, par.n_AHL, par.n_TETR]))


def getCenterSliceMatrix(diffmat, rotate):
  ''' Same as getCenterSlice by takes as imput a single matrix instead of a time series of matrices
  '''

  NumberOfRows = int(np.sqrt(len(diffmat))) # assuming diffusion always occurs in a square lattice
  CenterIndex = int(NumberOfRows/2) # middle row

  gradient = np.reshape(diffmat,(NumberOfRows,NumberOfRows))
  if rotate:
    gradient = gradient.T
    #print("CenterIndex",CenterIndex)
  return np.array(gradient[CenterIndex])  
  

for N in range(1):
  print("loglikelihood: ", loglikelihood(fll.df,[par.alpha_X, par.alpha_Y,
        par.beta_X, par.beta_Y, par.K_IPTG, par.K_AHL, par.K_LACI, par.K_TETR,
        par.K_ATC, par.n_LACI, par.n_AHL, par.n_TETR]))

##################################################################
######## COLOR MAPPING FUNCTIONS #################################
#################################################################

def xtocolor(Xarray,colormap = 'Reds', lim = [400.0,800.0]):
  ''' maps an array of floats into an array of colors
  for the colormap and boundaries lim chosen
  '''
  return getattr(cm, colormap)((Xarray - lim[0])/(lim[1]-lim[0]))

def xtocustomcolor(Xarray,color = [1,0,0], lim = [400.0,800.0]):
  ''' maps an array of floats into an array of colors
  that determine the intensity of the given color
  '''

  Xarray_rescaled = ((Xarray - lim[0])/(lim[1]-lim[0]))
  Xarray_rescaled = np.minimum(Xarray_rescaled,1)
  Xarray_rescaled = np.maximum(Xarray_rescaled,0)
  Xcolor = np.array([np.array(color)*X for X in Xarray_rescaled])
  Xcolor = np.column_stack((Xcolor,np.ones_like(Xarray)))
  return Xcolor

def mixcolor(Xarray,Yarray, mode = 'additive'):
  ''' function used to mix colors
  '''
  if mode == 'additive':
      C = Xarray + Yarray
      for iel,el in enumerate(C):
          C[iel] = np.minimum(C[iel],1.0) # if one of the coordinates is bigger than 1, then return 1
      return C

  if mode == 'additive_correct':
      C = Xarray + Yarray - np.ones_like(Xarray)
      for iel,el in enumerate(C):
          C[iel] = np.minimum(C[iel],1.0) # if one of the coordinates is bigger than 1, then return 1
      return C

  if mode == 'multiplicative':
      C = Xarray * Yarray
      return C

  elif mode == 'average':
      return (Xarray + Yarray)/2.0




