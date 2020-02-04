import integration_bs as bsint
import flow_load as fll
import scipy.special as sp
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
from pydream.core import run_dream
from pydream.parameters import SampledParam
from scipy.stats import norm, uniform
from pydream.convergence import Gelman_Rubin
from scipy.optimize import minimize


# fitting of parameters of the bistable switch defined by the models in integration_bs to the experimental data that is formatted by flow_load.py

parnames = ['$\\alpha_X$','$\\alpha_Y$','$\\beta_Y$','$\\beta_Y$','$K_{IPTG}$','$K_{AHL}$',   '$K_{LACI}$','$K_{TETR}$','$K_{ATC}$',   '$n_{LACI}$','$n_{AHL}$','$n_{TETR}$' ]


# lower_limits = np.array([2.55,2.35,2.0,2.0,  0.0,1.0 ,  -2.0,-1.0,-1,   0.0,0.0,0.0])
# scale_limits = np.array([0.1,0.1,1.0,1.0,  2.0,2.0,   2.0,2.0,2.0,    0.7,0.7,0.7])

# Limits of the parameter search in the order given by parnames
lower_limits = np.array([2.0,2.0,1.5,1.5,  -0.5,1.5 ,  -2.5,-1.5,-2.5,   0.0,0.0,0.0])
scale_limits = np.array([1.0,1.0,2.0,2.0,  2.0,2.0,   2.0,2.0,2.0,    0.7,0.7,0.7])

# lower_limits+np.random.random(len(scale_limits))*scale_limits

parameters_to_sample = [SampledParam(uniform, loc=lower_limits, scale=scale_limits)]

niterations = 40000
converged = False
total_iterations = niterations
nchains = 3 # number of MCMC chains
GRlim = 1.2 # GR Convergence limit

startchains = []
for ch in range(nchains):
	startchains.append(lower_limits+np.random.random(len(scale_limits))*scale_limits) # Initial point can be random or fixed close to a known solution
	# startchains.append(np.array([ 2.65928628 , 2.65969698 , 2.56514661 , 2.72445355 , 0.08293827 , 2.17268301
 # ,-1.50328667 ,-0.24599039 ,-0.5103528  , 0.39328967 , 0.32894559 , 0.32731401]))

def likelihood(param_vector):
	return bsint.loglikelihood(fll.df,10**param_vector)

def minuslikelihood(param_vector):
	return -1.0*likelihood(param_vector)

optimization_method = 'MCMC' # If a name different from MCMC is given then scipy is invoked with optimization_method routine

if optimization_method == 'MCMC':

	if __name__ == '__main__':

		#Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
		sampled_params, log_ps = run_dream(parameters_to_sample, likelihood, niterations=niterations,
		                nchains=nchains, multitry=True, gamma_levels=4, adapt_gamma=True, parallel = True,
		                history_thin=200, model_name='bsfit_nopysb_dreamzs_5chain', verbose=True,
		                start = startchains)
		
		#Save sampling output (sampled parameter values and their corresponding logps).
		for chain in range(len(sampled_params)):
			np.save('dream/bsfit_sampled_params_chain'+str(chain)+'_'+str(total_iterations), sampled_params[chain])
			np.save('dream/bsfit_logps_chain'+str(chain)+'_'+str(total_iterations), log_ps[chain])

		# Check convergence and continue sampling if not converged

		GR = Gelman_Rubin(sampled_params)
		print('At iteration: ', total_iterations, ' GR = ', GR)
		np.savetxt('dream/bsfit_GelmanRubin_iteration_' + str(total_iterations) + '.out', GR)

		old_samples = sampled_params
		if np.any(GR > GRlim):
			starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
			while not converged:
				total_iterations += niterations

				sampled_params, log_ps = run_dream(parameters_to_sample, likelihood, niterations=niterations,
												   nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True,
												   history_thin=1, model_name='bsfit_nopysb_dreamzs_5chain', verbose=True, restart=False)

				for chain in range(len(sampled_params)):
					np.save('dream/bsfit_sampled_params_chain' + str(chain) + '_' + str(total_iterations),
								sampled_params[chain])
					np.save('dream/bsfit_logps_chain' + str(chain) + '_' + str(total_iterations),
								log_ps[chain])

				old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
				GR = Gelman_Rubin(old_samples)
				print('At iteration: ', total_iterations, ' GR = ', GR)
				np.savetxt('dream/bsfit_GelmanRubin_iteration_' + str(total_iterations)+'.txt', GR)

				if np.all(GR < 1.2):
					converged = True

else: 
	# if the inference optimization_method is not 'MCMC' then scipy.optimize is called with the optimization_method chosen
	for ichain,chain in enumerate(startchains):
		res = minimize(minuslikelihood, x0 = chain, method = optimization_method)
		print('Result from optimization ',ichain, res.x, '\n with loglikelihood ',likelihood(res.x)) 