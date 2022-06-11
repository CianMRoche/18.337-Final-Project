#######################################
# run_MCMC.py: file for the likelihood function of the emcee and do the 1-d fit fo the velocity distribution
# Lina Necib, July 11, 2018, Caltech
#######################################


import matplotlib
import numpy as np
import pandas as pd
import matplotlib as mpl

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erf, erfc
import corner
from scipy.integrate import quad, simps
from scipy import integrate
from pylab import *
import functions_MCMC_cy
from functools import partial
from scipy.interpolate import interp1d
import read_inputs
from tqdm import tqdm
import emcee
from schwimmbad import MPIPool
import configparser
from matplotlib import gridspec
import os, sys

twopi = 2 * np.pi
sqrt2 = np.sqrt(2)
sqrt2pi = np.sqrt(twopi)

from scipy.stats import poisson

# print poisson.pmf(1.0,1.0)
poissonpdf = poisson.pmf


#######################################

def making_cuts(dataset, z_cut, v_cutoff=2000, rgcmin=0, rgcmax=20, z_max=10):
    """
    Takes the full dataset, applies cuts on [Fe/H] and |z|, and returns the cut dataset
    input:
    :dataset: pandas file, should have "x", "y", "z" as well as the velocity components
    :z_cut: float, cut on |z| in kpc
    :v_cutoff: float, cut on |v| to get rid of stars with |v| with crazy velocities. (and these are really crazy!!!)
    :rgcmin: R_gc > rgcmin cut
    :rgcmax: R_gc < rgcmax cut
    :z_max: float, cut max on |z|
    """

    # zlist = dataset["z"]
    # z_list_error = dataset["z_err"]
    # zval_2sigma_cut = 0.95

    xyz = dataset[["x", "y", "z"]].values
    rgc = np.linalg.norm(xyz, axis=1)

    x, y, z = xyz.T

    # Commented out code uses the errors on the distance, but that does create a bias
    # TODO: Check if that needs to be included later
    # x_err, y_err, z_err = dataset[["x_err", "y_err", "z_err"]].values.T

    # rgc_err = ( abs(x)*x_err + abs(y)*y_err + abs(z)*z_err )/rgc

    # rgc_val95 = np.array([ 0.5* ( - erf( (rgcmin - rgc[i])/(np.sqrt(2)*rgc_err[i]) )  +  erf( (rgcmax - rgc[i])/(np.sqrt(2)*rgc_err[i]) )  ) for i in range(len(rgc))])

    # rgc_val95 = integrated_gaussian(rgc, rgc_err, rgcmin, rgcmax)

    distance_cut = (rgc > rgcmin) & (rgc < rgcmax)
    # distance_cut = (rgc_val95 > zval_2sigma_cut)

    # Making a cut on crazy stars
    # Using stored v_abs and error velocities

    if z_max < 10:
        # zval95 = integrated_gaussian(abs(zlist), z_list_error, z_cut, z_max)
        z_cut_condition = (abs(z) < z_max) & (abs(z) > z_cut)

    else:
        # zval95 = np.array([1 - 0.5*(erf( (z_cut - zlist[i])/(np.sqrt(2)*z_list_error[i])  ) + erf( (z_cut + zlist[i])/(np.sqrt(2)*z_list_error[i])) ) for i in range(len(zlist)) ] )
        z_cut_condition = (abs(z) > z_cut)

    all_halo_cuts = z_cut_condition & distance_cut

    print("\n---Numbers within making_cuts---")
    print("Number of stars after z and rgc cuts = {}".format(np.count_nonzero(all_halo_cuts)) )

    all_v = dataset[["vabs"]].values

    v_norm = np.linalg.norm(all_v, axis=1)
    all_halo_cuts = all_halo_cuts & (v_norm < v_cutoff)

    print("Number of stars after unphysical outlier cut = {}".format(np.count_nonzero(all_halo_cuts)) )

    cut_df = dataset[all_halo_cuts]

    return cut_df


########################## Functions for binned data fit  #############################

# Updated way to calculate the smeared velocity distribution
def prob_i_in_bin_j(two_bin_edges, x_list, xerror_list, y_list):
    """
    Takes in the bin_edges for a bin (j), and caluclates
    the probability of a star with true velocity v_i to be in that bin
    :two_bin_edges: array of length 2
    :x_list:        list of v_i values of truth model
    :xerror_list:   average velocity error for that x value
    :y_list:        truth model pdf for that x value
    """

    l = two_bin_edges[0]
    u = two_bin_edges[1]
    sqrt2 = np.sqrt(2)
    dx = x_list[1] - x_list[0]

    val = np.zeros_like(x_list)

    for i in range(len(x_list)):
        if (x_list[i] > 0.0):
            val[i] = dx / (u - l) * y_list[i] * 0.5 * (erf((x_list[i] - l) / (sqrt2 * xerror_list[i]))
                                                       - erf((x_list[i] - u) / (sqrt2 * xerror_list[i])))
        else:
            val[i] = 0.0

    return sum(val)


# Number of counts in each bin, normalized to 1 (without outlier model)
def number_counts(bin_edges, x_list, xerror_list, y_list):
    """
    Number of counts per bin
    :bins_edges: array of length N+1: where N number of bins
    :x_list:        list of x values of truth model
    :xerror_list:   average velocity error for that x value
    :y_list:        truth model pdf for that x value
    """

    val_bins = np.array([prob_i_in_bin_j(bin_edges[j:j + 2], x_list, xerror_list, y_list)
                         for j in range(len(bin_edges) - 1)])

    return val_bins


# integrate truth distribution w/ gaussian
# def likeint(v0, v0err, vesc, k, vmin_data, vmin_model):
#     """
#     Code from Tongyan to evaluate the integrals
#     """
#     xmin = max(min(v0-4.0*v0err,0.9*vesc), vmin_model)
#     xmax = min(vesc,v0 + 3.0*v0err)
#     xx = np.linspace(xmin, xmax, 51)
#     yy = np.exp(-0.5*(v0-xx)**2/(v0err)**2)*(vesc-xx)**k

#     return np.log(simps(yy)*(xx[1]-xx[0])) - (k+1.)*np.log((vesc - vmin_model)) #- k*np.log((vesc - vmin_data)) #


# likelihood_eval = np.vectorize(likeint)

# def lnlike(theta, v, verr, vmin = 300):
#     """
#     Log likelihood of the function evaluated, later we will add outlier model
#     :theta: values to fit for, currently [vesc, k, norm] #adding the normalization to float
#     :v: velocity of stars
#     :verr: errors on the velocity of the stars
#     :vmin: fixed to 300 km/s for now, but can be updated
#     """
#     vesc, k = theta
#     model = np.sum(likelihood_eval(v, verr, vesc, k, vmin, vmin) )  #C*(vesc - x)**k
#     # inv_sigma2 = 1.0/(yerr**2 + model**2)
#     # return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2/(twopi))))
#     return model


def model_function(v, vesc, k, vmin=300):
    """
    Evaluating the function with the requirement that v>vesc -> f = 0
    :v: velocity norm
    :vesc: escape velocity
    :k: Coefficient
    :norm: normalization factor, in log space
    :vmin: fixed to 300 km/s for now, but can be updated
    """
    C = (k + 1.) / (vesc - vmin) ** (k + 1)
    if v < vesc:
        return C * (vesc - v) ** k
    else:
        return 0


model_eval = np.vectorize(model_function)


################################################################
##  Functions for binned likelihood analysis to follow

def lnlike_binned(theta, xbins, y, xerr=0.05, vmin=300, errortype='percent'):
    """
    Log likelihood of the function evaluated
    :theta: values to fit for, currently [vesc, k, norm] + [frac, sigma_out] for outlier model
    :xbin: histogram bin edges
    :y: number of stars in histogram (not normalized)
    :vmin: fixed to 300 km/s for now, but can be updated
    """
    vesc, k, norm, frac, sigma_out = theta

    # take a smooth rep of the distribution
    xx = np.linspace(vmin - 100, 1000, 250)
    model = model_eval(xx, vesc, k, vmin=vmin)

    # now smear the middle values of the bins
    if (errortype == 'percent'):
        model_smeared = number_counts(xbins, xx, xx * xerr, model)
    elif (errortype == 'polyfit'):
        model_smeared = number_counts(xbins, xx, xx * (xx * xerr[0] + xerr[1]), model)
    else:
        model_smeared = number_counts(xbins, xx, (xx * 0.0 + xerr), model)

    xmid = (xbins[0:-1] + xbins[1:]) / 2.0
    dx = (xbins[1:] - xbins[0:-1])
    model = norm * sum(y) * dx * (model_smeared * (1 - exp(frac)) + \
                                  exp(frac) * exp(-(xmid) ** 2 / (2 * sigma_out ** 2)) * 2.0 / sqrt(
                2 * pi * sigma_out ** 2) / (1.0 - erf(vmin / (np.sqrt(2) * sigma_out))))

    return sum([np.log(poissonpdf(y[i], model[i]) + 1e-100) for i in range(len(y))])
    # return -0.5*(np.sum((y*norm-model_smeared)**2*inv_sigma2 - np.log(inv_sigma2/(twopi))))


def lnprob_binned(theta, xbins, y, xerr, errortype, vmin=300, vescprior=False,
                  kpriormin=1, kpriormax=15):
    """
    Combining the probablity with the log likelihood
    :xbins: histogram bin means [length N + 1]
    :y: number of stars in histogram bins [length N]
    :xerr: average error on the velocities in bin i
    :vmin: fixed to 300 km/s for now, but can be updated
    """
    lp = lnprior_binned(theta, vmin=vmin, vescprior=vescprior, kpriormin=kpriormin, kpriormax=kpriormax)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_binned(theta, xbins, y, vmin=vmin, xerr=xerr, errortype=errortype)


def lnprior_binned(theta, vmin=300, vescprior=False, kpriormin=1, kpriormax=15):
    """
    Adding the priors of the different values
    :theta: [vesc, k] to be fit for; n is overall normalization relative to observed number of events
             f, sigma_out are for the outlier model
    """
    vesc, k, n, frac, sigma_out = theta
    if vmin < vesc < 1000 and kpriormin < k < kpriormax and 0.1 < n < 10 and np.log(1e-6) < frac < np.log(
            1) and 500. < sigma_out < 2000:
        if (vescprior):
            return - np.log(vesc)
        else:
            return 0.0
    return -np.inf


################################################################
##  Functions for unbinned likelihood analysis to follow


def constant(vesc, k, sigma, vmin=300):
    """
    Constant that multiplies the whole function. Needs to be properly normalized to 1
    :theta: values to fit for, [vesc, k] # added for code consistency
    :sigma: dispersion for that particular value of the error
    :vmin: fixed for now
    """

    integral = integrate.quad(lambda x: 0.5 * (vesc - x) ** k * (erf((x - vmin) / (sqrt2 * sigma)) \
                                                                 - erf((x - vesc) / (sqrt2 * sigma))), vmin, vesc)

    return 1. / integral[0]


def fitting_function(vesc, k, v, sigma, vmin=300):
    """
    Evaluating the function with the requirement that v>vesc -> f = 0
    :v: velocity norm
    :theta: values to fit for, [vesc, k, norm] # added for code consistency
    :sigma: dispersion for that particular value of the error
    :sigma_int: intrinsic dispersion that we fit for
    :vmin: fixed for now
    """

    vmax = vesc
    C = constant(vmax, k, sigma, vmin=vmin)

    # if (sigma/v < 1e-5):
    #     integral = (vesc  - v)**k

    # elif (sigma/v > 0.2):
    #     integral = integrate.quad(lambda x: np.exp( - (v - x)*(v - x)/(2*sigma*sigma)) * (vesc  - x)**k/ (sqrt2pi*sigma), vmin, vesc )[0]

    if (v < vesc) & (v > vmin):
        n_samples = 1000
        min_integral = vmin
        max_integral = vmax  # vesc
        x_integrals = np.linspace(min_integral, max_integral, n_samples)

        # print("integral interval", min_integral, max_integral, v, sigma)

        y_integral = np.exp(- (v - x_integrals) * (v - x_integrals) / (2 * sigma * sigma)) * (
                vesc - x_integrals) ** k / (sqrt2pi * sigma)
        integral = integrate.simps(y_integral, x_integrals)

        return C * integral
    else:
        return 0


function_vectorized = np.vectorize(fitting_function)


def probability_no_error(vesc, k, v, verr, vmin=300):
    """
    Evaluating the probability function without taking into account any errors
    :v: velocity norm
    :theta: values to fit for, [vesc, k] # added for code consistency
    :vmin: fixed for now
    """

    if vesc <= vmin:
        return 0

    constant = (k + 1) / (vesc - vmin) ** (k + 1)
    if (v <= vesc) & (v > vmin):
        func = (vesc - v) ** k
    else:
        func = 0

    return constant * func


probability_no_error_vectorized = np.vectorize(probability_no_error)


def outliers(v, vesc, k, sigma=1000, vmin=300.):
    """
    Evaluating the outlier model where it is a Gaussian with a certain dispersion.
    TODO: think about including the errors on the measurements here
    :v: velocity norm
    :vesc: escape velocity
    :k: Coefficient
    :vmin: fixed to 300 km/s for now, but can be updated
    :sigma: dispersion of the outlier model. We will vary it later
    """

    a = 2. / (erfc(vmin / (sqrt2 * sigma)))
    # print("a", a)
    # print("exp", np.exp( -(v*v*1.0) /(2.*sigma*sigma)))
    # print("sqrt ( 2 pi sigma^2)",  np.sqrt(twopi*sigma*sigma) )
    prob = a * np.exp(-(v * v * 1.0) / (2. * sigma * sigma)) / np.sqrt(twopi * sigma * sigma)
    return prob


outlier_vectorized = np.vectorize(outliers)


def lnlike(theta, v, verr, vmin=300):
    """
    Log likelihood of the function evaluated, later we will add outlier model
    :theta: values to fit for, [vesc, k, norm] # added for code consistency
    :v: velocity norm
    :verr: dispersion for that particular value of the error
    :vmin: fixed to 300 km/s for now, but can be updated
    """

    vesc, k, frac, sigma_out = theta

    # model = function_vectorized(vesc, k, v, verr, vmin = vmin)

    model = probability_no_error_vectorized(vesc, k, v, verr, vmin=vmin)

    outlier_likelihood = outlier_vectorized(v, vesc, k, sigma=sigma_out, vmin=vmin)

    return np.sum(np.log((1. - np.exp(frac)) * model + np.exp(frac) * outlier_likelihood))


def lnprior(theta, vmin=300):
    """
    Adding the priors of the different values
    :theta: [vesc, k] to be fit for
    """

    vesc, k, frac, sigma_out = theta
    if vmin < vesc < 1000 and 1.0 < k < 10.0 and np.log(1e-6) < frac < np.log(1) and 500 < sigma_out < 2000:
        return 0.0
    return -np.inf


def lnprob(theta, v, verr, vmin=300, outlier=1):  # sigma = 1000,
    """
    Combining the probablity with the log likelihood
    :v: velocity of stars
    :verr: errors on the stars
    :vmin: fixed to 300 km/s for now, but can be updated
    """

    lp = lnprior(theta, vmin=vmin)

    if not np.isfinite(lp):
        return -np.inf

    lk = lnlike(theta, v, verr, vmin=vmin)
    if not (np.isfinite(lp + lk)):
        return -np.inf
    return lp + lk


################################################################
##  Unbinned Analysis without outlier model

def lnlike_no_outlier(theta, v, verr, sigma=1000., vmin=300):
    """
    Log likelihood of the function evaluated, later we will add outlier model
    :theta: values to fit for, [vesc, k] # added for code consistency
    :v: velocity norm
    :verr: dispersion for that particular value of the error
    :vmin: fixed to 300 km/s for now, but can be updated
    """

    vesc, k = theta
    model = probability_no_error_vectorized(vesc, k, v, verr, vmin=vmin)
    return np.sum(np.log(model))


def lnprior_no_outlier(theta, vmin=300):
    """
    Adding the priors of the different values
    :theta: [vesc, k] to be fit for
    """

    vesc, k = theta
    if vmin < vesc < 1000 and 1.0 < k < 10.0:
        return 0.0
    return -np.inf


def lnprob_no_outlier(theta, v, verr, vmin=300):  # sigma = 1000,
    """
    Combining the probablity with the log likelihood
    :v: velocity of stars
    :verr: errors on the stars
    :vmin: fixed to 300 km/s for now, but can be updated
    """

    lp = lnprior_no_outlier(theta, vmin=vmin)

    if not np.isfinite(lp):
        return -np.inf

    lk = lnlike_no_outlier(theta, v, verr, vmin=vmin)
    if not (np.isfinite(lp + lk)):
        return -np.inf
    return lp + lk


################################################################
##  Plotting Functions


def make_corner_plot(samples, filename, plots_dir='../plots/', true_values=[], label_list=[], inverse_vesc_prior=False,
                     three_functions=False, two_vesc=False, levels=[0.68, 0.95, 0.99]):
    """
    Makes the corner plot given the output of the MCMC
    :samples: array of shape (nwalkers*nsteps, ndim)
    :filename: string for the file name
    :plots_dir: location where the files will be saved
    :true_values: array of real values
    """

    truths = None
    if len(true_values) > 0:
        truths = true_values

    if inverse_vesc_prior:
        samples[:, 0] = 1. / samples[:, 0]
    if two_vesc:
        samples[:, 6] = 1. / samples[:, 6]

    if len(label_list) > 0:
        fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=label_list, show_titles=True, truths=truths,
                            levels=levels)
    else:
        if three_functions:
            plotlabels = [r"$v_{\rm{esc}}$", r'$k$', r'$\log(f)$', r'$\log(\sigma_{\rm{out}})$', \
                          r'$k_{\rm{Subs}}$', r'$f_{\rm{Subs}}$', r'$k_3$', r'$f_3$']
        elif two_vesc:
            plotlabels = [r"$v_{\rm{esc}}$", r'$k$', r'$\log(f)$', r'$\log(\sigma_{\rm{out}})$', \
                          r'$k_{\rm{Subs}}$', r'$f_{\rm{Subs}}$', r"$v_{\rm{esc, S}}$"]
        else:
            plotlabels = [r"$v_{\rm{esc}}$", r'$k$', r'$\log(f)$', r'$\log(\sigma_{\rm{out}})$', \
                          r'$k_{\rm{Subs}}$', r'$f_{\rm{Subs}}$', r'$v_{\rm{esc}, \rm{Subs}}$']
        plotlabels = plotlabels[0:len(samples[0])]
        print("Length of samples", np.shape(samples))
        print("Length of plotlabels", len(plotlabels))
        print("Truths", truths)
        fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=plotlabels, plotrange=0.95, show_titles=True,
                            truths=truths, levels=levels)
    fig.savefig(plots_dir + filename + '.pdf')
    plt.close()


def make_corner_plot_vesc_k(samples, filename, plots_dir='../plots/', true_values=[], title='', ndata=0, vmin=300,
                            color='b', plotrange=0.95):
    """
    Makes the corner plot given the output of the MCMC
    :samples: array of shape (nwalkers*nsteps, ndim)
    :filename: string for the file name
    :plots_dir: location where the files will be saved
    :true_values: array of real values
    """

    truths = None
    if len(true_values) > 0:
        truths = true_values

    print("Samples shape", np.shape(samples))

    fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=[r"$v_{\rm{esc}}$", r'$k$'], show_titles=True,
                        truths=truths, range=plotrange, plot_density=True, color=color, plot_datapoints=False,
                        levels=[0.68, 0.95, 0.99], truth_color='mediumblue')

    if title != '':
        fig.text(0.6, 0.95,
                 title + '\n' + r'$N=$' + str(ndata) + '\n' + r'$v_{\rm{min}} = $' + str(int(vmin)) + ' km/s',
                 verticalalignment='top', fontsize=14)

    fig.savefig(plots_dir + 'k_vesc' + filename + '.pdf')
    plt.close()


def evaluate_best_fits(samples):
    """
    Returns the confidence intervals of the best fits
    :samples: array of shape (nwakers*nsteps, ndim)
    """
    vesc_mcmc, k_mcmc, frac_mcmc = np.array(
        list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))))
    return vesc_mcmc, k_mcmc, frac_mcmc


def walker_plot(sampler_chain, nwalkers, param_index, extratext, plots_dir='../plots/'):
    """
    Evolution of the parameters as the scan progresses
    :sampler_chain: result of the MCMC
    :nwalkers: number of walkers used
    :param_index: parameter to plot
    :extratext: string to be added to the filename
    :plots_dir: string for the file location
    """

    mpl.rcParams.update({'font.size': 18, 'font.family': 'serif'})

    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['patch.linewidth'] = 1
    mpl.rcParams['figure.figsize'] = 6, 5

    minorticks_on()
    tick_params('both', length=4, width=1, which='major', direction='in', labelsize=14)
    tick_params('both', length=2, width=1, which='minor', direction='in', labelsize=14)

    for i in range(0, nwalkers):
        plt.plot(sampler_chain[i, :, param_index], color='grey', alpha=0.5)

    label_list = [r"$v_{\rm{esc}}$", r'$k$', r'$\log(f)$', r'$\log(\sigma_{\rm{out}})$', r'$k_S$', r'$\log(g_S)$']
    plt.ylabel(label_list[param_index])
    plt.xlabel('Step Number')
    plt.tight_layout()
    plt.savefig(plots_dir + 'walker_plot' + extratext + '.pdf')
    plt.close()


def normal_distribution(mean, sigma, x):
    """
    Returns the evaluation of a normal distribution at point x
    """
    factor = 1. / np.sqrt(2 * np.pi * sigma * sigma)
    return factor * np.exp(- (x - mean) * (x - mean) / (2 * sigma * sigma))


def plot_kde(speed, speed_errors, x_data):
    """
    Returns x and y's of the kde distribution
    :speed: values of the speed, length n
    :speed_errors: values of the errors, length n
    """
    y_data = np.zeros_like(x_data)
    for k in range(len(x_data)):
        y_data[k] += np.sum(normal_distribution(speed, speed_errors, x_data[k]))

    bin_size = x_data[1] - x_data[0]

    normalization = np.sum(y_data)
    return y_data / normalization / bin_size


def plot_data_fit(
        speed,
        speed_errors,
        filename,
        samples,
        plots_dir='../plots/',
        vmin=300,
        true_values=[],
        full_fit=0,
        inverse_vesc_prior=False,
        mock_run=False,
        plot_log=True,
        kde=False,
        chi=False,
        title_text='',
        error_text='',
        component_names=['Halo', 'Sausage', 'Disk'],
        color_list=['red', 'darkgreen', 'violet'],
        substructure=0,
        sub_mean=370,
        sub_dispersion=20,
        frac_substructure=0):
    """
    Function to plot the histogram of the speed distribution. This will plot the fit as well
    :speed: values of the speed, length n
    :speed_errors: values of the errors, length n
    :filename: string to save the file
    :samples: posterior distribution output
    :plots_dir: string, directory for plotting
    :vmin: default minimum cutoff where the fit starts
    :vesc_true: real value for vesc
    :k_true: real value for k
    :true_values: for the MCMC, true distributions
    :full_fit: comparison with the distributions I had before
    :plot_log: making the plot log
    :kde: boolean, if true, instead of histograms we have kdes
    :chi: Booleanm if true, adds the number of chi^2/dof to the plot
    :title_text: string for thing to add to title
    :error_text: string to add about the error inside the plot.
    :component_names: list of names to customize the labels
    :param: substructure, boolean, if true, it includes another gaussian in the generation for an extra structure.
    :param: sub_mean: double for the mean of the new Gaussian
    :param: sub_dispersion: double for the dispersion of the new Gaussian
    :param: frac_substructure: double for the fraction of the new Gaussian
    """

    data_dir = '../data/chi_squared/'

    mpl.rcParams.update({'font.size': 18, 'font.family': 'serif'})

    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['patch.linewidth'] = 1
    mpl.rcParams['figure.figsize'] = 5, 4

    fontsize = 16

    ax = plt.subplot(1, 1, 1)
    ax.minorticks_on()
    ax.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
    ax.tick_params('both', length=4, width=1, which='minor', direction='in', labelsize=fontsize)

    vmax = 800
    v_step = 25
    factor = 1

    print("Making the data plot here for vmin", vmin, "!!!!!")
    # Now the different components

    speed_array = np.arange(vmin, vmax, v_step)
    avgerror = np.mean(speed_errors / speed)
    print('Average error', avgerror)
    speed_error_array = speed_array * avgerror

    check_no_errors = int(np.all(np.logical_not(speed_errors)))

    # Data
    print("Number of stars", len(speed))

    if kde:
        y_data = plot_kde(speed, speed_errors, speed_array)
        ax.plot(speed_array, factor * y_data, color='black', label='Data KDE')
        bin_list = (speed_array[:-1] + speed_array[1:]) / 2.
    else:
        hist, bin_edges = np.histogram(speed, range=[vmin, vmax], bins=speed_array, density=True)
        bin_list = (bin_edges[:-1] + bin_edges[1:]) / 2.
        ax.step(bin_list, factor * hist, where='mid', label='Data', lw=2.0, color='black')

    bin_list_error_array = np.array(bin_list) * avgerror

    # Last n terms of samples

    n_test = 5000
    samples = samples[-n_test:, :]
    sausage = False
    two_vesc = False
    three_functions = False

    # if inverse_vesc_prior:
    #     samples[:,0] = 1./samples[:,0]

    # Read the posteriors
    if (len(samples[0]) == 4):  # Outlier + Data
        vesc, k, log_f_out, log_sigma_out = samples.T
        f_out = np.exp(log_f_out)
        sigma_out = np.exp(log_sigma_out)
        data_normalization = (1 - f_out)


    elif (len(samples[0]) == 6):  # Outlier + data + sausage
        sausage = True
        vesc, k, log_f_out, log_sigma_out, kS, fS = samples.T
        f_out = np.exp(log_f_out)
        sigma_out = np.exp(log_sigma_out)
        data_normalization = (1 - f_out) * (1 - fS)
        sausage_normalization = (1 - f_out) * fS

    elif (len(samples[0]) == 7):  # Outlier + data + sausage (separate vesc)
        sausage = True
        two_vesc = True
        vesc, k, log_f_out, log_sigma_out, kS, fS, vescS = samples.T
        if inverse_vesc_prior:
            vescS = 1. / vescS
            print("Mean of vescS", np.mean(vescS))
        f_out = np.exp(log_f_out)
        sigma_out = np.exp(log_sigma_out)
        data_normalization = (1 - f_out) * (1 - fS)
        sausage_normalization = (1 - f_out) * fS

    elif (len(samples[0]) == 8):  # Outlier + data + sausage + third params
        sausage = True
        two_vesc = False
        three_functions = True
        vesc, k, log_f_out, log_sigma_out, kS, fS, k_three, f_three = samples.T
        f_out = np.exp(log_f_out)
        sigma_out = np.exp(log_sigma_out)
        data_normalization = (1 - f_out) * (1 - fS - f_three)
        sausage_normalization = (1 - f_out) * fS
        third_normalization = (1 - f_out) * f_three

    n_params = 2  # vesc, and k
    # data_func = np.exp(np.array(map(partial(functions_MCMC_cy.function_vectorized, v = speed_array, sigma = speed_error_array, vmin = vmin, relative_error = 0, kin_energy = 0), vesc, k ) ) ) # shape (n_samples * n_speed_array)

    if check_no_errors:
        data_func = np.exp(np.array(list(
            map(partial(functions_MCMC_cy.function_no_err_vectorized, v=bin_list, sigma=bin_list_error_array, vmin=vmin,
                        relative_error=0, kin_energy=0), vesc, k))))

    else:
        data_func = np.exp(np.array(list(
            map(partial(functions_MCMC_cy.function_vectorized, v=bin_list, sigma=bin_list_error_array, vmin=vmin,
                        relative_error=0, kin_energy=0), vesc, k))))

    data_func_normalized = np.array([data_normalization * data_func[:, k] for k in
                                     range(len(data_func[0]))])  # Shape flipped (n_speed_array, n_samples)

    data_mean = np.mean(data_func_normalized, axis=1)
    data_sigma = np.std(data_func_normalized, axis=1)

    ## Making Plots

    ax.fill_between(bin_list, factor * (data_mean + data_sigma), factor * (data_mean - data_sigma), alpha=0.4,
                    color=color_list[0], label=component_names[0])

    ax.plot(bin_list, factor * data_mean, color=color_list[0])

    ## Outliers
    n_params += 2  # sigma, and fraction
    # outlier_func = np.exp(np.array(map(partial(functions_MCMC_cy.outliers_normalized_vectorized, speed_array, verr = speed_error_array,  vmin = vmin), sigma_out)))

    outlier_func = np.exp(np.array(list(
        map(partial(functions_MCMC_cy.outliers_normalized_vectorized, v=bin_list, verr=bin_list_error_array, vmin=vmin),
            sigma_out))))

    outlier_func_normalized = np.array([f_out * outlier_func[:, k] for k in range(len(outlier_func[0]))])

    outlier_mean = np.mean(outlier_func_normalized, axis=1)
    outlier_sigma = np.std(outlier_func_normalized, axis=1)

    ## Plotting
    ax.fill_between(bin_list, factor * (outlier_mean + outlier_sigma), factor * (outlier_mean - outlier_sigma),
                    alpha=0.4, color='cyan', label='Outliers')

    ax.plot(bin_list, factor * outlier_mean, color='cyan')

    summed_distribution = data_func_normalized + outlier_func_normalized

    if sausage:
        if three_functions:
            n_params += 4
            if check_no_errors:
                sausage_func = np.exp(np.array(list(
                    map(partial(functions_MCMC_cy.function_no_err_vectorized, v=bin_list, sigma=bin_list_error_array,
                                vmin=vmin, relative_error=0, kin_energy=0), vesc, kS))))

                third_func = np.exp(np.array(list(
                    map(partial(functions_MCMC_cy.function_no_err_vectorized, v=bin_list, sigma=bin_list_error_array,
                                vmin=vmin, relative_error=0, kin_energy=0), vesc, k_three))))
            else:
                sausage_func = np.exp(np.array(list(
                    map(partial(functions_MCMC_cy.function_vectorized, v=bin_list, sigma=bin_list_error_array,
                                vmin=vmin, relative_error=0, kin_energy=0), vesc, kS))))

                third_func = np.exp(np.array(list(
                    map(partial(functions_MCMC_cy.function_vectorized, v=bin_list, sigma=bin_list_error_array,
                                vmin=vmin, relative_error=0, kin_energy=0), vesc, k_three))))

            third_func_normalized = np.array(
                [third_normalization * third_func[:, k] for k in range(len(third_func[0]))])

            third_mean = np.mean(third_func_normalized, axis=1)
            third_sigma = np.std(third_func_normalized, axis=1)
            print("shape of third_func_normalized", np.shape(third_func_normalized))

            ## Plotting
            print("Length check, bin list, means and sigmas", len(bin_list), len(third_mean), len(third_sigma))

            ax.fill_between(bin_list, factor * (third_mean + third_sigma), factor * (third_mean - third_sigma),
                            alpha=0.4, color=color_list[2], label=component_names[2])
            ax.plot(bin_list, factor * third_mean, color=color_list[2])

            summed_distribution += third_func_normalized


        elif two_vesc:
            n_params += 3  # vescS, k, fraction
            if check_no_errors:
                sausage_func = np.exp(np.array(list(
                    map(partial(functions_MCMC_cy.function_no_err_vectorized, v=bin_list, sigma=bin_list_error_array,
                                vmin=vmin, relative_error=0, kin_energy=0), vescS, kS))))
            else:
                sausage_func = np.exp(np.array(list(
                    map(partial(functions_MCMC_cy.function_vectorized, v=bin_list, sigma=bin_list_error_array,
                                vmin=vmin, relative_error=0, kin_energy=0), vescS, kS))))
        else:
            n_params += 2  # k, fraction
            # sausage_func = np.exp(np.array(map(partial(functions_MCMC_cy.function_vectorized, v = speed_array, sigma = speed_error_array, vmin = vmin, relative_error = 0, kin_energy = 0), vesc, kS ) ) )
            if check_no_errors:
                sausage_func = np.exp(np.array(list(
                    map(partial(functions_MCMC_cy.function_no_err_vectorized, v=bin_list, sigma=bin_list_error_array,
                                vmin=vmin, relative_error=0, kin_energy=0), vesc, kS))))
            else:
                sausage_func = np.exp(np.array(list(
                    map(partial(functions_MCMC_cy.function_vectorized, v=bin_list, sigma=bin_list_error_array,
                                vmin=vmin, relative_error=0, kin_energy=0), vesc, kS))))

        sausage_func_normalized = np.array(
            [sausage_normalization * sausage_func[:, k] for k in range(len(sausage_func[0]))])

        sausage_mean = np.mean(sausage_func_normalized, axis=1)
        sausage_sigma = np.std(sausage_func_normalized, axis=1)

        ## Plotting

        ax.fill_between(bin_list, factor * (sausage_mean + sausage_sigma), factor * (sausage_mean - sausage_sigma),
                        alpha=0.4, color=color_list[1], label=component_names[1])
        ax.plot(bin_list, factor * sausage_mean, color=color_list[1])

        summed_distribution += sausage_func_normalized

    sum_mean = np.mean(summed_distribution, axis=1)
    sum_sigma = np.std(summed_distribution, axis=1)

    ## Chi Squared Evaluation
    if chi:
        chi_squared = np.sum((sum_mean - hist) ** 2 / (sum_sigma) ** 2)

        print("Total chi squared", chi_squared)
        degrees_of_freedom = len(bin_list) - n_params
        print("Number of degrees of freedom", degrees_of_freedom)
        chi_squared_dof = chi_squared * 1.0 / degrees_of_freedom
        print("Chi / d.o.f", chi_squared_dof)

        np.save(data_dir + 'chi_squared_vmin_' + filename + '.npy', np.array([vmin, chi_squared_dof]))

    ax.fill_between(bin_list, factor * (sum_mean + sum_sigma), factor * (sum_mean - sum_sigma), alpha=0.4, color='blue',
                    label='Sum')
    ax.plot(bin_list, factor * sum_mean, color='blue', linewidth=1.4)

    if full_fit:
        bins, fv_subs = np.loadtxt('../data/f_v_substructure_normalized_galactic.txt').T
        bins, fv_halo = np.loadtxt('../data/f_v_halo_normalized_galactic.txt').T
        fraction = 0.76

        subs_interpolation = interp1d(bins, fv_subs)
        halo_interpolation = interp1d(bins, fv_halo)

        norm_subs = 1. / quad(subs_interpolation, vmin, vmax)[0]
        norm_halo = 1. / quad(halo_interpolation, vmin, vmax)[0]

        print("normalizations, subs, halo", norm_subs, norm_halo)

        limit_plot = np.where(bins < vmin)[0][-1]

        ax.plot(bins[limit_plot:], factor * (1 - fraction) * norm_halo * fv_halo[limit_plot:], ls='--', lw=2,
                color=color_list[0])
        ax.plot(bins[limit_plot:], factor * fraction * norm_subs * fv_subs[limit_plot:], ls='--', lw=2,
                color=color_list[1])

        ax.plot(bins[limit_plot:], factor * (
                (1 - fraction) * norm_halo * fv_halo[limit_plot:] + fraction * norm_subs * fv_subs[limit_plot:]),
                ls='--', lw=2, color=color_list[0])

    if len(true_values) == 4:
        vesc_true, k_true, f_out_true, sigma_out_true = true_values

        if check_no_errors:
            data_true = (1 - f_out_true) * np.exp(np.array(
                functions_MCMC_cy.function_no_err_vectorized(vesc=vesc_true, k=k_true, v=speed_array,
                                                             sigma=speed_error_array, vmin=vmin, relative_error=0,
                                                             kin_energy=0)))
        else:
            data_true = (1 - f_out_true) * np.exp(np.array(
                functions_MCMC_cy.function_vectorized(vesc=vesc_true, k=k_true, v=speed_array, sigma=speed_error_array,
                                                      vmin=vmin, relative_error=0, kin_energy=0)))

            #     outlier_func = np.exp(np.array(map(partial(functions_MCMC_cy.outliers_normalized_vectorized, speed_array, verr = speed_error_array,  vmin = vmin), sigma_out)))
        outlier_true = f_out_true * np.exp(np.array(
            functions_MCMC_cy.outliers_normalized_vectorized(sigma_out_true, speed_array, speed_error_array, vmin)))

        ax.plot(speed_array, factor * data_true, ls='--', lw=2, color=color_list[0])
        ax.plot(speed_array, factor * outlier_true, ls='--', lw=2, color='cyan')

        ax.plot(speed_array, factor * (data_true + outlier_true), ls='--', lw=2, color='blue')

    elif len(true_values) == 6:

        vesc_true, k_true, f_out_true, sigma_out_true, ks_true, fs_true = true_values

        if check_no_errors:
            data_true = (1 - f_out_true) * (1 - fs_true) * np.exp(np.array(
                functions_MCMC_cy.function_no_err_vectorized(vesc=vesc_true, k=k_true, v=speed_array,
                                                             sigma=speed_error_array, vmin=vmin, relative_error=0,
                                                             kin_energy=0)))

            sausage_true = (1 - f_out_true) * (fs_true) * np.exp(np.array(
                functions_MCMC_cy.function_no_err_vectorized(vesc=vesc_true, k=ks_true, v=speed_array,
                                                             sigma=speed_error_array, vmin=vmin, relative_error=0,
                                                             kin_energy=0)))
        else:
            data_true = (1 - f_out_true) * (1 - fs_true - frac_substructure) * np.exp(np.array(
                functions_MCMC_cy.function_vectorized(vesc=vesc_true, k=k_true, v=speed_array, sigma=speed_error_array,
                                                      vmin=vmin, relative_error=0, kin_energy=0)))

            sausage_true = (1 - f_out_true) * (fs_true) * np.exp(np.array(
                functions_MCMC_cy.function_vectorized(vesc=vesc_true, k=ks_true, v=speed_array, sigma=speed_error_array,
                                                      vmin=vmin, relative_error=0, kin_energy=0)))

        if substructure:
            substructure_vel = (1 - f_out_true) * frac_substructure * np.exp(
                functions_MCMC_cy.substructure_normalized_vectorized(sigma=sub_dispersion, v=speed_array, mu=sub_mean,
                                                                     verr=speed_error_array, vmin=vmin))

            ax.plot(speed_array, factor * substructure_vel, ls='--', color='purple', label='Subs')

        outlier_true = f_out_true * np.exp(np.array(
            functions_MCMC_cy.outliers_normalized_vectorized(sigma_out_true, speed_array, speed_error_array, vmin)))

        ax.plot(speed_array, factor * data_true, ls='--', lw=2, color=color_list[0])
        ax.plot(speed_array, factor * outlier_true, ls='--', lw=2, color='cyan')
        ax.plot(speed_array, factor * sausage_true, ls='--', lw=2, color=color_list[1])

        if substructure:
            ax.plot(speed_array, factor * (data_true + outlier_true + sausage_true + substructure_vel), ls='--', lw=2,
                    color='blue')
        else:
            ax.plot(speed_array, factor * (data_true + outlier_true + sausage_true), ls='--', lw=2, color='blue')

    elif len(true_values) == 8:

        vesc_true, k_true, f_out_true, sigma_out_true, ks_true, fs_true, k_three_true, f_three_true = true_values

        if check_no_errors:
            data_true = (1 - f_out_true) * (1 - fs_true - f_three_true) * np.exp(np.array(
                functions_MCMC_cy.function_no_err_vectorized(vesc=vesc_true, k=k_true, v=speed_array,
                                                             sigma=speed_error_array, vmin=vmin, relative_error=0,
                                                             kin_energy=0)))

            sausage_true = (1 - f_out_true) * (fs_true) * np.exp(np.array(
                functions_MCMC_cy.function_no_err_vectorized(vesc=vesc_true, k=ks_true, v=speed_array,
                                                             sigma=speed_error_array, vmin=vmin, relative_error=0,
                                                             kin_energy=0)))

            third_true = (1 - f_out_true) * f_three_true * np.exp(np.array(
                functions_MCMC_cy.function_no_err_vectorized(vesc=vesc_true, k=ks_true, v=speed_array,
                                                             sigma=speed_error_array, vmin=vmin, relative_error=0,
                                                             kin_energy=0)))
        else:
            data_true = (1 - f_out_true) * (1 - fs_true - f_three_true) * np.exp(np.array(
                functions_MCMC_cy.function_vectorized(vesc=vesc_true, k=k_true, v=speed_array, sigma=speed_error_array,
                                                      vmin=vmin, relative_error=0, kin_energy=0)))

            sausage_true = (1 - f_out_true) * (fs_true) * np.exp(np.array(
                functions_MCMC_cy.function_vectorized(vesc=vesc_true, k=ks_true, v=speed_array, sigma=speed_error_array,
                                                      vmin=vmin, relative_error=0, kin_energy=0)))

            third_true = (1 - f_out_true) * f_three_true * np.exp(np.array(
                functions_MCMC_cy.function_vectorized(vesc=vesc_true, k=ks_true, v=speed_array, sigma=speed_error_array,
                                                      vmin=vmin, relative_error=0, kin_energy=0)))

        outlier_true = f_out_true * np.exp(np.array(
            functions_MCMC_cy.outliers_normalized_vectorized(sigma_out_true, speed_array, speed_error_array, vmin)))

        ax.plot(speed_array, factor * data_true, ls='--', lw=2, color=color_list[0])
        ax.plot(speed_array, factor * outlier_true, ls='--', lw=2, color='cyan')
        ax.plot(speed_array, factor * sausage_true, ls='--', lw=2, color=color_list[1])
        ax.plot(speed_array, factor * third_true, ls='--', lw=2, color=color_list[2])

        ax.plot(speed_array, factor * (data_true + outlier_true + sausage_true + third_true), ls='--', lw=2,
                color='blue')

    if chi:
        ax.text(0.05, 0.95, r'$N=$' + str(len(speed)) + '\n' + r' $\chi^2/\rm{d.o.f} = %.2f$' % chi_squared_dof,
                ha='left', va='top', transform=ax.transAxes, fontsize=14)
    else:
        if check_no_errors:
            ax.text(0.05, 0.95,
                    r'$N=$' + str(len(speed)) + ', No Err' + '\n' + r'$v_{\rm{min}}=$' + str(int(vmin)) + ' km/s',
                    ha='left', va='top', transform=ax.transAxes, fontsize=14)
        else:
            ax.text(0.05, 0.95,
                    r'$N=$' + str(len(speed)) + error_text + '\n' + r'$v_{\rm{min}}=$' + str(int(vmin)) + ' km/s',
                    ha='left', va='top', transform=ax.transAxes, fontsize=14)

    ax.set_xlabel(r'$|\vec{v}|$' + ' [km/s]', fontsize=14)
    ax.set_xlim([vmin, vmax])
    ax.set_ylabel(r'$g(|\vec{v}|)$' + ' [km/s]' + r'$^{-1}$', fontsize=14)
    ax.legend(fontsize=12, frameon=False)

    if plot_log:
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim([1e-6 * factor, 0.1 * factor])
    plt.tight_layout()
    # if mock_run:
    #     plt.title(r'\textbf{Simulated Data}', fontsize = 16)
    # else:
    #     plt.title(r'\textbf{Gaia DR2}' + title_text , fontsize = 16)
    plt.savefig(plots_dir + "data_plot" + filename + ".pdf", bbox_inches='tight')
    plt.close()


def plot_binned_data_fit(speed_hist, bin_edges, hist_errors, filename, speed_err=0,
                         plots_dir='../plots/', vesc_fit=0, k_fit=0, n_fit=1.0, vmin=300):
    """
    Function to plot the histogram of the speed distribution. This will plot the fit as well
    :speed_hist: values of the speeds, length n
    :bin_edges: edges of the bins to plot, length n+1
    :speed_errors: values of the errors, length n
    :vesc_fit: best fit value of the escape velocity
    :k_fit: best fit value for k
    :vmin: default minimum cutoff where the fit starts
    """

    mpl.rcParams.update({'font.size': 18, 'font.family': 'serif'})

    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['patch.linewidth'] = 1
    mpl.rcParams['figure.figsize'] = 5, 4

    bin_mean = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = (-bin_edges[:-1] + bin_edges[1:]) / 2.0

    fontsize = 16

    plt.minorticks_on()
    plt.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
    plt.tick_params('both', length=4, width=1, which='minor', direction='in', labelsize=fontsize)

    plt.errorbar(bin_mean, speed_hist, xerr=bin_width, yerr=np.sqrt(speed_hist), label='Data', lw=2, color='SteelBlue')

    if vesc_fit > 0:
        # plot truth model
        model = model_eval(bin_mean, vesc_fit, k_fit, vmin=vmin)
        plt.plot(bin_mean, model, label='Fit (not smeared)', color='k', linewidth=1.3)
        # plot smeared truth model
        xx = np.linspace(100, 1000, 50)
        model = model_eval(xx, vesc_fit, k_fit, vmin)
        model_smeared = number_counts(bin_edges, xx, xx * (xx * speed_err[0] + speed_err[1]), model) * n_fit
        plt.plot(bin_mean, model_smeared, label='Fit (smeared)', color='Crimson', linewidth=1.3)

    plt.xlabel(r'$|v|$')
    plt.xlim([vmin, 800])
    plt.ylabel(r'$N$')
    plt.legend(loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir + "data_plot" + filename + ".pdf", bbox_inches='tight')
    plt.close()


def extract_extratext_data(rgcmin, rgcmax, zmin, zmax, verrcut, vmin, sausage=1, vphicut=1, inverse_vesc_prior=0,
                           limited_prior=0, kpriormin=0, kpriormax=15):
    """
    Takes in the input parameters and returns the extratext to call the right file
    :rgcmin: R_min in spherical coordinates
    :rgcmax: R_max in spherical coordinates
    :zmin: minimum vertical cut
    :zmax: maximum vertical cut
    :verrcut: cut on the error in the velocities
    :sausage: boolean, whether or not the analysis is sausage
    :inverse_vesc_prior: boolean, whether or not 1/ve prior
    """

    jobs_dir = '../jobs/'
    chains_dir = '../chains/'
    data_dir = '../data/'
    plots_dir = '../plots/'

    if limited_prior:
        if int(vphicut):
            filename = jobs_dir + 'gaia_' + 'rgc_' + str(rgcmin) + '_' + str(rgcmax) + 'z_' + str(zmin) + '_' + str(
                zmax) + 'verrcut_' + str(verrcut) + '_vmin_' + str(vmin) + 'k_min' + str(kpriormin) + 'k_max' + str(
                kpriormax) + '.ini'
        else:
            filename = jobs_dir + 'gaia_' + 'rgc_' + str(rgcmin) + '_' + str(rgcmax) + 'z_' + str(zmin) + '_' + str(
                zmax) + 'verrcut_' + str(verrcut) + '_vmin_' + str(vmin) + 'k_min' + str(kpriormin) + 'k_max' + str(
                kpriormax) + 'no_vphi' + '.ini'

    else:
        if int(vphicut):
            filename = jobs_dir + 'gaia_' + 'rgc_' + str(rgcmin) + '_' + str(rgcmax) + 'z_' + str(zmin) + '_' + str(
                zmax) + 'verrcut_' + str(verrcut) + '_vmin_' + str(vmin) + '.ini'
        else:
            filename = jobs_dir + 'gaia_' + 'rgc_' + str(rgcmin) + '_' + str(rgcmax) + 'z_' + str(zmin) + '_' + str(
                zmax) + 'verrcut_' + str(verrcut) + '_vmin_' + str(vmin) + 'no_vphi' + '.ini'

    print("filename is", filename)

    parameters = read_inputs.setup(filename)

    if len(parameters) == 19:
        rgcmin, rgcmax, z_cut, z_max, verrcut, vmin, vphicut, vescprior, sigmapriormin, kpriormin, kpriormax, fs_sausage_min, fs_sausage_max, vesc_mock, k_mock, sigma_mock, fraction_mock, ksausage_mock, fsausage_mock = parameters
    elif len(parameters) == 17:
        rgcmin, rgcmax, z_cut, z_max, verrcut, vmin, vphicut, vescprior, sigmapriormin, kpriormin, kpriormax, vesc_mock, k_mock, sigma_mock, fraction_mock, ksausage_mock, fsausage_mock = parameters
    elif len(parameters) == 18:
        rgcmin, rgcmax, z_cut, z_max, verrcut, vmin, vphicut, vescprior, sigmapriormin, kpriormin, kpriormax, fs_sausage_min, fs_sausage_max, sixd_propermotion, propermotion, dr2, ananke, mock = parameters
    else:
        print("Length of parameters", len(parameters))
        raise ValueError('Wrong length of the parameter list!')

    extratext = 'unbinned_'
    if (sixd_propermotion):
        extratext = extratext + 'pmonly_'
    if (propermotion):
        extratext = extratext + 'pmfull_'

    extratext += 'r_' + str(rgcmin) + '_' + str(rgcmax) + '_z_' + str(z_cut) + "_" + str(z_max) + "_vmin" + str(vmin)
    extratext += '_verrcut' + str(verrcut)

    if limited_prior:
        extratext += 'k_min' + str(kpriormin) + 'k_max' + str(kpriormax)

    if vphicut:
        extratext += '_vphicut_'

    if not (sausage):
        extratext += '_1func'

    if inverse_vesc_prior:
        extratext += '_inverse_vesc_'

    return extratext, vmin


def extract_extratext_mock(vmin_fit, sausage=1, inverse_vesc_prior=0, limited_prior=0, kpriormin=0, kpriormax=15,
                           k_mock=3.5, k_sausage=0.5, frac_sausage=0.5, fs_sausage_min=0.0, fs_sausage_max=1.0,
                           kpriormin_input=0, kpriormax_input=15, limited_priors=0):
    """
    Takes in the input parameters and returns the extratext to call the right file
    :rgcmin: R_min in spherical coordinates
    :rgcmax: R_max in spherical coordinates
    :zmin: minimum vertical cut
    :zmax: maximum vertical cut
    :verrcut: cut on the error in the velocities
    :sausage: boolean, whether or not the analysis is sausage
    :inverse_vesc_prior: boolean, whether or not 1/ve prior
    """

    jobs_dir = '../jobs/'
    chains_dir = '../chains/'
    data_dir = '../data/'
    plots_dir = '../plots/'

    filename = jobs_dir + 'k_sausage_mock_k_mock' + str(k_mock) + '_ks_' + str(k_sausage) + '_fs_' + str(
        frac_sausage) + '_min_' + str(fs_sausage_min) + '_max_' + str(fs_sausage_max) + '.ini'

    print("Loading filename", filename)

    parameters = read_inputs.setup(filename)

    if len(parameters) == 19:
        rgcmin, rgcmax, z_cut, z_max, verrcut, vmin, vphicut, vescprior, sigmapriormin, kpriormin, kpriormax, fs_sausage_min, fs_sausage_max, vesc_mock, k_mock, sigma_mock, fraction_mock, ksausage_mock, fsausage_mock = parameters
    elif len(parameters) == 17:
        rgcmin, rgcmax, z_cut, z_max, verrcut, vmin, vphicut, vescprior, sigmapriormin, kpriormin, kpriormax, vesc_mock, k_mock, sigma_mock, fraction_mock, ksausage_mock, fsausage_mock = parameters
    else:
        print("Length of parameters", len(parameters))
        raise ValueError('Wrong length of the parameter list!')

    extratext = "mock_" + "vmin_" + str(vmin) + "_vesc_" + str(vesc_mock) + "_k_" + str(k_mock) + "_err_" + str(
        verrcut) + "_sigma_" + str(sigma_mock) + "_frac_" + str(fraction_mock)
    extratext += '_sausage_k_' + str(ksausage_mock) + '_frac_' + str(fsausage_mock)

    # if limited_prior:
    #     extratext += 'k_min' + str(kpriormin) + 'k_max' + str(kpriormax)

    if not (sausage):
        extratext += '_not_fit_S_'

    extratext += '_fs_' + str(fs_sausage_min) + '_' + str(fs_sausage_max)

    if limited_priors:
        extratext += '_limited_'
        kpriormin = kpriormin_input
        kpriormax = kpriormax_input

    extratext += 'k_prior_' + str(kpriormin) + '_' + str(kpriormax)
    extratext += '_vmin_fit_' + str(float(vmin_fit))

    if inverse_vesc_prior:
        extratext += '_over_ve_'

    return extratext, vmin_fit


def calculate_bic(number_parameters, number_data_points, max_log_likelihood):
    """
    Returns the BIC https://en.wikipedia.org/wiki/Bayesian_information_criterion
    :param: number_parameters: int, number of fitting parameters. When fitting with 2 functions, this number increases
    :param: number_data_points: int, number of stars in the same
    :param: max_log_likelihood: float, max log likelihood of that fit
    """
    return number_parameters * np.log(number_data_points) - 2 * max_log_likelihood


def calculate_aic(number_parameters, max_log_likelihood):
    """
    Returns the AIC https://en.wikipedia.org/wiki/Akaike_information_criterion
    :param: number_parameters: int, number of fitting parameters. When fitting with 2 functions, this number increases
    :param: max_log_likelihood: float, max log likelihood of that fit
    """
    return 2 * number_parameters - 2 * max_log_likelihood


def calculate_delta_bic(number_parameters_a, number_parameters_b, number_data_points, max_log_likelihood_a,
                        max_log_likelihood_b):
    """
    Returns the Delta BIC of two hypotheses
    :param: number_parameters_a: int, number of fitting parameters of the first hypothesis. When fitting with 2 functions, this number increases
    :param: number_parameters_b: int, number of fitting parameters of the first hypothesis. When fitting with 2 functions, this number increases
    :param: number_data_points: int, number of stars in the same
    :param: max_log_likelihood_a: float, max log likelihood of the fit of hypothesis a
    :param: max_log_likelihood_b: float, max log likelihood of the fit of hypothesis b
    """

    return calculate_bic(number_parameters_a, number_data_points, max_log_likelihood_a) - calculate_bic(
        number_parameters_b, number_data_points, max_log_likelihood_b)


def extract_vesc_mock(
        vmin_fit,
        sausage=1,
        two_vesc=0,
        inverse_vesc_prior=0,
        limited_prior=0,
        kpriormin=0,
        kpriormax=15,
        make_plot=0,
        return_k=0,
        return_k_fs=0,
        k_mock=3.5,
        k_sausage=0.5,
        frac_sausage=0.5,
        fs_sausage_min=0.0,
        fs_sausage_max=1.0,
        kpriormin_input=0,
        kpriormax_input=15,
        return_vesc_k=0,
        mean_values=1,
        return_n_data=0,
        return_speed=0,
        kin_energy=0,
        sausage_generation=1,
        vesc_mock=500,
        error_mag=0.1,
        error_type='percent',
        error_range=0,
        return_bic=0,
        return_aic=0,
        f_outlier=0.01,
        substructure=0,
        sub_mean=370,
        sub_dispersion=20,
        frac_substructure=0.2,
        jobs_dir='../jobs/',
        chains_dir='../chains/',
        data_dir='../data/',
        plots_dir='../plots/',
        n_mock=-1):
    """
    Takes in the input parameters, loads the samples file and gets the vesc values with the error bars
    :rgcmin: R_min in spherical coordinates
    :rgcmax: R_max in spherical coordinates
    :zmin: minimum vertical cut
    :zmax: maximum vertical cut
    :verrcut: cut on the error in the velocities
    :sausage: boolean, whether or not the analysis is sausage
    :inverse_vesc_prior: boolean, whether we are using the inverse priors
    :limited_prior: boolean, whether or not we are limiting the priors
    :make_plot: boolean, whether or not to make the data plot
    :return_k: boolean, to return the value of k instead of vesc
    :return_k_fs: boolean, to return the value of k and fs
    :return_vesc_k: boolean, to return the value of k and vesc
    :mean_values: boolean, if true it just returns the mean and +/- 1 sigma, else returns the full posterior
    :return_n_data: boolean, returns the number of data points
    :return_speed: boolean, returns the speed
    :kin_energy: boolean, fits with the kinematic energy formula
    :sausage_generation: boolean, if true, means that we generated a sausage in the beginning. It should be a default
    :vesc_mock: float, this is the true escape velocity
    :error_mag: float, magnitude of the errors
    :error_type: string, type of error, should be percent, absolute, or 'no_errors'
    :error_range: boolean, if true, then the error span different values up to a cap
    :return_bic: boolean, if true, it returns the bic of the dataset
    :return_aic: boolean, if true, it returns the aic of the dataset
    :param: f_outlier: float, fraction nof the outlier
    :param: substructure, boolean, if true, it includes another gaussian in the generation for an extra structure.
    :param: sub_mean: double for the mean of the new Gaussian
    :param: sub_dispersion: double for the dispersion of the new Gaussian
    :param: frac_substructure: double for the fraction of the new Gaussian
    :param: data_dir: string, data directory
    :param: plots_dir: string, plots directory
    :param: chains_dir: string, chains directory
    :param: jobs_dir: string, jobs directory
    :param: n_mock: int, if -1, it means we're not assigning a number to this mock, else it enters in the file name
    """

    mock_text = ''
    if n_mock > -1:
        mock_text = ' ' + str(n_mock)

    if sausage_generation:
        filename = jobs_dir + 'mock_run_k_' + str(k_mock) + '_vesc_' + str(vesc_mock) + '_ks_' + str(
            k_sausage) + '_fs_' + str(frac_sausage) + '_err_' + str(
            error_mag) + '_type_' + error_type + mock_text + '.ini'
    else:
        filename = jobs_dir + 'single_mock_run_k_' + str(k_mock) + '_vesc_' + str(vesc_mock) + '_err_' + str(
            error_mag) + '_type_' + error_type + mock_text + '.ini'

    print("Reading the inialization file", filename)

    ##############################################################################
    # Reading the input file

    if sausage_generation:
        config = configparser.ConfigParser()
        config.read(filename)
        error_type = config['Cuts']['error_type']
        verrcut, vmin = [float(config['Cuts'][key]) for key in ['verrcut', 'vmin']]
        vesc_mock, k_mock, sigma_mock, fraction_mock, k_sausage, frac_sausage = [float(config['Mocks'][key]) for key in
                                                                                 ['vesc_mock', 'k_mock', 'sigma_mock',
                                                                                  'fraction_mock', 'ksausage_mock',
                                                                                  'fsausage_mock']]
        kpriormin, kpriormax, sigmapriormin = [float(config['Priors'][key]) for key in
                                               ['kpriormin', 'kpriormax', 'sigmapriormin']]
        fs_sausage_min, fs_sausage_max = [0, 1]
    else:
        config = configparser.ConfigParser()
        config.read(filename)
        error_type = config['Cuts']['error_type']
        verrcut, vmin = [float(config['Cuts'][key]) for key in ['verrcut', 'vmin']]
        vesc_mock, k_mock, sigma_mock, fraction_mock = [float(config['Mocks'][key]) for key in
                                                        ['vesc_mock', 'k_mock', 'sigma_mock', 'fraction_mock']]
        kpriormin, kpriormax, sigmapriormin = [float(config['Priors'][key]) for key in
                                               ['kpriormin', 'kpriormax', 'sigmapriormin']]
        fs_sausage_min, fs_sausage_max = [0, 1]

    if error_type == 'no_errors':
        verrcut = 0.0

    vmin_text = str(int(vmin))
    vesc_text = str(int(vesc_mock))
    sigma_text = str(int(sigma_mock))

    fraction_mock = f_outlier  # To overwrite this as needed.

    # if return_n_data:
    #     vmin_text = str(int(vmin))
    #     vesc_text = str(int(vesc_mock))
    #     sigma_text = str(int(sigma_mock))

    if kin_energy:
        extratext = "mock_fitkin_" + "vmin_" + vmin_text + "_vesc_" + vesc_text + "_k_" + str(k_mock) + "_err_" + str(
            verrcut) + "_sigma_" + sigma_text + "_frac_" + str(fraction_mock)
    else:
        extratext = "mock_" + "vmin_" + vmin_text + "_vesc_" + vesc_text + "_k_" + str(k_mock) + "_err_" + str(
            verrcut) + "_sigma_" + sigma_text + "_frac_" + str(fraction_mock)

    if sausage_generation:
        extratext += '_sausage_k_' + str(k_sausage) + '_frac_' + str(frac_sausage)

    if error_range:
        extratext += 'err_range'

    if substructure:
        extratext += '_subs_' + str(sub_mean) + '_dis_' + str(sub_dispersion) + '_frac_' + str(frac_substructure)

    if n_mock > -1:
        extratext += '_mock_' + str(n_mock)

    if make_plot or return_n_data or return_speed or return_bic:
        n_plot = 3000
        print("Loading " + data_dir + 'mocks/speed_' + extratext + '.npy')
        speed = np.load(data_dir + 'mocks/speed_' + extratext + '.npy')
        speed_error = np.load(data_dir + 'mocks/speed_error_' + extratext + '.npy')

        # if verrcut > 0:
        #     speed_cut_error = (speed_error/speed < verrcut)
        #     speed = speed[ speed_cut_error ]
        #     speed_error = speed_error[ speed_cut_error ]

        # Cutting the data below vmin
        speed_cut = speed > vmin_fit
        speed = speed[speed_cut]
        speed_error = speed_error[speed_cut]

        # Reducing the dataset for initial testing purposes

        speed = speed[:n_plot]
        speed_error = speed_error[:n_plot]

        print("post velocity error shape is: ", np.shape(speed))

        if return_speed:
            return speed, speed_error

        ndata = len(speed)
        if return_n_data:
            return ndata

    # if limited_prior:
    #     extratext += 'k_min' + str(kpriormin) + 'k_max' + str(kpriormax)

    if not (sausage):
        extratext += '_not_fit_S_'

    if inverse_vesc_prior:
        extratext += '_inverse_vesc_'

    if sausage_generation and sausage:
        extratext += '_fs_' + str(fs_sausage_min) + '_' + str(fs_sausage_max)

    if limited_prior:
        extratext += '_limited_'
        kpriormin = kpriormin_input
        kpriormax = kpriormax_input

    extratext += 'k_prior_' + str(kpriormin) + '_' + str(kpriormax)
    extratext += '_vmin_fit_' + str(float(vmin_fit))

    print("Loading file", chains_dir + extratext + '.npy')

    # print("Loading file", chains_dir + extratext + '.npy' )
    ##############################################################################
    ####### Reading Samples

    if return_bic:
        log_list = np.loadtxt(chains_dir + extratext + '.txt')
        max_log_likelihood = np.max(log_list[:, -2])
        number_parameters = len(log_list[0]) - 2
        return calculate_bic(number_parameters, ndata, max_log_likelihood)

    if return_aic:
        log_list = np.loadtxt(chains_dir + extratext + '.txt')
        max_log_likelihood = np.max(log_list[:, -2])
        number_parameters = len(log_list[0]) - 2
        return calculate_aic(number_parameters, max_log_likelihood)

    samples = np.load(chains_dir + extratext + '.npy')

    if make_plot:
        plot_data_fit(speed, speed_error, extratext, samples, plots_dir=plots_dir, vmin=vmin_fit,
                      inverse_vesc_prior=inverse_vesc_prior, mock_run=True)

    # if inverse_vesc_prior:
    #     samples[:,0] = 1./samples[:,0]

    if sausage:
        if two_vesc:
            vesc_mcmc, k_mcmc, frac_mcmc, sigma_mcmc, ks, fs, vescS = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                                                          zip(*np.percentile(samples, [16, 50, 84],
                                                                                             axis=0)))

            if return_k:
                return k_mcmc, ks
            return vesc_mcmc, vescS

        else:
            vesc_mcmc, k_mcmc, frac_mcmc, sigma_mcmc, kS, fS = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                                                   zip(*np.percentile(samples, [16, 50, 84], axis=0)))

            if return_k_fs:
                if mean_values:
                    return kS, fS
                else:
                    return samples[:, 4], samples[:, 5]

            if return_k:
                if mean_values:
                    return k_mcmc, kS
                else:
                    return samples[:, 1], samples[:, 4]

            if return_vesc_k:
                if mean_values:
                    return vesc_mcmc, kS
                else:
                    return samples[:, 0], samples[:, 4]

            return vesc_mcmc
    else:

        vesc_mcmc, k_mcmc, frac_mcmc, sigma_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                                       zip(*np.percentile(samples, [16, 50, 84], axis=0)))

        if return_vesc_k:
            if mean_values:
                return vesc_mcmc, k_mcmc
            else:
                return samples[:, 0], samples[:, 1]

        if return_k:
            if mean_values:
                return k_mcmc
            else:
                return samples[:, 1]
        return vesc_mcmc


def extract_vesc(rgcmin, rgcmax, zmin, zmax, verrcut, vmin, sausage=1, vphicut=0, two_vesc=0, inverse_vesc_prior=0,
                 limited_prior=0, kpriormin=0, kpriormax=15, make_plot=0, return_k=0, return_k_fs=0, return_ndata=0,
                 return_extratext=0, return_vesc_k=0, mean_values=1, kin_energy=0, error_type='percent', return_bic=0,
                 return_aic=0, accreted=0, cutoff=0.95, three_functions=0, chains_dir='../chains/', jobs_dir='../jobs/',
                 data_dir='../data/', plots_dir='../plots/', fire=0, lsr=0, simulation='m12i', edr3=0):
    """
    Takes in the input parameters, loads the samples file and gets the vesc values with the error bars
    :rgcmin: R_min in spherical coordinates
    :rgcmax: R_max in spherical coordinates
    :zmin: minimum vertical cut
    :zmax: maximum vertical cut
    :verrcut: cut on the error in the velocities
    :sausage: boolean, whether or not the analysis is sausage
    :inverse_vesc_prior: boolean, whether we are using the inverse priors
    :limited_prior: boolean, whether or not we are limiting the priors
    :make_plot: boolean, whether or not to make the data plot
    :return_k: boolean, to return the value of k instead of vesc
    :return_k_fs: boolean, to return the value of k and fs
    :return_ndata: boolean, to return the number of data points
    :return_extratext: boolean, to return the filename
    :return_vesc_k: boolean, to return vesc and k
    :mean_values: boolean, if true it just returns the mean and +/- 1 sigma, else returns the full posterior
    :error_type: string, either 'no_errors', 'percent', 'absolute'. Type of error considered here
    :return_bic: boolean, if true, it returns the bic of the dataset
    :return_aic: boolean, if true, it returns the aic of the dataset
    :accreted: boolean, if true, loads up the accreted dataset
    :cutoff: float, the cutoff on the score for the accreted stars
    :three_functions: boolean, if true, the fit uses 3 functions.
    :chains_dir: directory for the chains
    :plots_dir: directory for the plots
    :data_dir: directory for the data
    :jobs_dir: directory for the jobs
    :param: fire: Boolean, fi true, it gets the FIRE sim not Gaia
    :param: lsr: int, location of the Sun in the simulation, 0,1,2
    :param: simulation 'm12i' string for the simulation
    :param: edr3, boolean, if so analyzes edr3 data
    """

    sim_data_text = ''
    assert simulation == 'm12i' or simulation == 'm12f'

    if edr3:
        print("Reading Gaia EDR3")
        sim_data_text = '_edr3_'
    elif fire:
        sim_data_text = '_fire_' + simulation + '_' + str(lsr) + '_'

    if limited_prior:
        if int(vphicut):
            filename = jobs_dir + 'gaia_' + 'rgc_' + str(rgcmin) + '_' + str(rgcmax) + 'z_' + str(zmin) + '_' + str(
                zmax) + 'verrcut_' + str(verrcut) + '_vmin_' + str(vmin) + 'k_min' + str(kpriormin) + 'k_max' + str(
                kpriormax) + '.ini'
        else:
            filename = jobs_dir + 'gaia_' + 'rgc_' + str(rgcmin) + '_' + str(rgcmax) + 'z_' + str(zmin) + '_' + str(
                zmax) + 'verrcut_' + str(verrcut) + '_vmin_' + str(vmin) + 'k_min' + str(kpriormin) + 'k_max' + str(
                kpriormax) + 'no_vphi' + '.ini'

    else:
        if int(vphicut):
            filename = jobs_dir + 'gaia_' + sim_data_text + 'rgc_' + str(rgcmin) + '_' + str(rgcmax) + 'z_' + str(
                zmin) + '_' + str(
                zmax) + 'verrcut_' + str(verrcut) + error_type + '_vmin_' + str(vmin) + '.ini'
        else:
            filename = jobs_dir + 'gaia_' + sim_data_text + 'rgc_' + str(rgcmin) + '_' + str(rgcmax) + 'z_' + str(
                zmin) + '_' + str(
                zmax) + 'verrcut_' + str(verrcut) + error_type + '_vmin_' + str(vmin) + 'no_vphi' + '.ini'

    ##############################################################################
    # Reading the input file

    print("Reading the input", filename)

    config = configparser.ConfigParser()
    config.read(filename)

    error_type = config['Cuts']['error_type']
    rgcmin, rgcmax, z_cut, z_max, verrcut, vmin = [float(config['Cuts'][key]) for key in
                                                   ['rgcmin', 'rgcmax', 'z_min', 'z_max', 'verrcut', 'vmin']]

    vphicut = int(config['Cuts']['vphicut'])

    kpriormin, kpriormax, sigmapriormin = [float(config['Priors'][key]) for key in
                                           ['kpriormin', 'kpriormax', 'sigmapriormin']]
    fs_sausage_min, fs_sausage_max = [0, 1]

    sixdpropermotion, propermotion, dr2, edr3, fire, lsr = [int(config['Dataset'][key]) for key in
                                                            ['sixdpropermotion', 'propermotion', 'dr2', 'edr3', 'fire',
                                                             'lsr']]

    if accreted:
        extratext = 'accreted_' + str(int(100 * cutoff))
        data_dir += 'accreted/'
        extratext += 'r_' + str(rgcmin) + '_' + str(rgcmax) + '_z_' + str(z_cut) + "_" + str(z_max) + "_vmin" + str(
            vmin)
        extratext += '_verrcut' + str(verrcut)

    else:
        extratext = ''
        if dr2:
            data_dir += 'dr2/'
        elif edr3:
            data_dir += 'edr3/'
        elif fire:
            data_dir += 'simulations/' + simulation + '/'
            extratext += 'lsr_' + str(lsr) + '_'

        extratext += 'r_' + str(rgcmin) + '_' + str(rgcmax) + '_z_' + str(z_cut) + "_" + str(z_max) + "_vmin" + str(
            int(vmin))
        extratext += '_verrcut' + str(verrcut) + error_type

    if make_plot or return_ndata:
        n_plot = 2000
        print("Loading " + data_dir + 'speed_' + extratext + '.npy')
        speed = np.load(data_dir + 'speed_' + extratext + '.npy')
        speed_error = np.load(data_dir + 'speed_error_' + extratext + '.npy')
        vphi = np.load(data_dir + 'vphi_' + extratext + '.npy')

        if (vphicut):
            # vphi convention here appears to be opposite Monari convention, notice sign.
            # print " vphi negative,", sum([vphi < 0])
            # print " vphi positive,", sum([vphi > 0])
            vphi_vel_cut = 0
            speed = speed[(vphi < vphi_vel_cut)]
            speed_error = speed_error[(vphi < vphi_vel_cut)]
            print("post vphicut data size is: ", len(speed), len(vphi))

        if error_type == 'percent':
            data_cut = (speed_error / speed < verrcut)
        elif error_type == 'absolute':
            data_cut = (speed_error < verrcut)
        elif error_type == 'no_error':
            data_cut = np.array(np.ones_like(speed_error), dtype=bool)
        else:
            ValueError("The type of error you're looking for is not implemented!")

        speed = speed[data_cut]
        speed_error = speed_error[data_cut]

        #
        # Cutting the data below vmin
        speed_cut = speed > vmin
        speed = speed[speed_cut]
        speed_error = speed_error[speed_cut]

        # Reducing the dataset for initial testing purposes

        speed = speed[:n_plot]
        speed_error = speed_error[:n_plot]

        print("post velocity error shape is: ", np.shape(speed))
        #
        ndata = len(speed)
        if return_ndata:
            return ndata

    if dr2:
        if accreted:
            title_text = ', Accreted Data' + r'$S>$' + str(cutoff)
            corner_text = 'Accreted Data' + r'$S>$' + str(cutoff)
        else:
            title_text = ', All Data'
            corner_text = 'All Data'
    elif edr3:
        title_text = ', Gaia EDR3'
        corner_text = 'Gaia EDR3'
    elif fire:
        title_text = ', FIRE, LSR=' + str(lsr)
        corner_text = 'FIRE, LSR=' + str(lsr)

    if vphicut:
        extratext += '_vphicut_'
        title_text = ', Retrograde Data'
        corner_text = 'Retrograde Data'

    if limited_prior:
        extratext += 'k_min' + str(kpriormin) + 'k_max' + str(kpriormax)
        title_text += r'$k \in [%.1f, %.1f]$' % (kpriormin, kpriormax)

    if not (sausage):
        extratext += '_1func'

    if kin_energy:
        extratext += '_kin'

    if three_functions:
        extratext += '_3func'

    if inverse_vesc_prior:
        extratext += '_inverse_vesc_'

    if two_vesc:
        extratext += '_two_vesc'

    print("Loading file", chains_dir + extratext + '.npy')
    ##############################################################################
    ####### Reading Samples

    if return_bic:
        log_list = np.loadtxt(chains_dir + extratext + '.txt')
        max_log_likelihood = np.max(log_list[:, -2])
        number_parameters = len(log_list[0]) - 2
        return calculate_bic(number_parameters, ndata, max_log_likelihood)

    if return_aic:
        log_list = np.loadtxt(chains_dir + sim_data_text + extratext + '.txt')
        max_log_likelihood = np.max(log_list[:, -2])
        number_parameters = len(log_list[0]) - 2
        return calculate_aic(number_parameters, max_log_likelihood)

    if return_extratext:
        return extratext

    samples = np.load(chains_dir + extratext + '.npy')

    if make_plot:

        # functions_MCMC.plot_data_fit(speed, speed_error, extratext, samples, plots_dir=plots_dir, vmin = vmin, inverse_vesc_prior = inverse_vesc_prior, title_text = title_text)

        # if inverse_vesc_prior:
        #     samples[:,0] = 1./samples[:,0]

        if sausage:
            make_corner_plot_vesc_k(np.array([samples[:, 0], samples[:, 4]]).T,
                                    filename='vmin' + str(vmin) + '_S_' + str(sausage) + 'v_phi' + str(
                                        vphicut) + '_inv_' + str(inverse_vesc_prior), plots_dir=plots_dir, color='red',
                                    ndata=ndata, vmin=vmin, title=r'\textbf{Gaia DR2}' + '\n' + corner_text,
                                    plotrange=[(400., 900.), (0., 5)])
        else:

            make_corner_plot_vesc_k(np.array([samples[:, 0], samples[:, 1]]).T,
                                    filename='vmin' + str(vmin) + '_S_' + str(sausage) + 'v_phi' + str(
                                        vphicut) + '_inv_' + str(inverse_vesc_prior), plots_dir=plots_dir, color='blue',
                                    ndata=ndata, vmin=vmin, title=r'\textbf{Gaia DR2}' + '\n' + corner_text,
                                    plotrange=[(400., 900.), (0., 5)])

    if sausage:
        if two_vesc:
            vesc_mcmc, k_mcmc, frac_mcmc, sigma_mcmc, ks, fs, vescS = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                                                          zip(*np.percentile(samples, [16, 50, 84],
                                                                                             axis=0)))

            if return_k:
                return k_mcmc, ks
            return vesc_mcmc, vescS

        elif three_functions:
            vesc_mcmc, k_mcmc, frac_mcmc, sigma_mcmc, kS, fS, k_three, f_three = map(
                lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

            if return_k_fs:
                return kS, fS

            if return_k:
                if mean_values:
                    return k_mcmc, kS, k_three
                else:
                    return samples[:, 1], samples[:, 4], samples[:, 6]

            if return_vesc_k:
                if mean_values:
                    return vesc_mcmc, kS
                else:
                    return samples[:, 0], samples[:, 4]

            return vesc_mcmc

        else:
            vesc_mcmc, k_mcmc, frac_mcmc, sigma_mcmc, kS, fS = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                                                   zip(*np.percentile(samples, [16, 50, 84], axis=0)))

            if return_k_fs:
                return kS, fS

            if return_k:
                if mean_values:
                    return k_mcmc, kS
                else:
                    return samples[:, 1], samples[:, 4]

            if return_vesc_k:
                if mean_values:
                    return vesc_mcmc, kS
                else:
                    return samples[:, 0], samples[:, 4]

            return vesc_mcmc
    else:
        # if inverse_vesc_prior:
        #     samples[:,0] = 1./samples[:,0]

        vesc_mcmc, k_mcmc, frac_mcmc, sigma_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                                       zip(*np.percentile(samples, [16, 50, 84], axis=0)))

        if return_vesc_k:
            if mean_values:
                return vesc_mcmc, k_mcmc
            else:
                return samples[:, 0], samples[:, 1]

        if return_k:
            if mean_values:
                return k_mcmc
            else:
                return samples[:, 1]
        return vesc_mcmc


def make_run(sampler, nsamples, p0, ndim, save_progress=False, burn_file_path='', test_plotting=False,
             n_steps_to_plot=50, plot_walkers=False, nwalkers=200, extratext='', plots_dir='../plots/', true_values=[],
             sausage=False, save_samples=True, chains_dir='../chains_dir/', cross_correlation=1,
             inverse_vesc_prior=False, three_functions=False, two_vesc=False):
    """
    Sets up the burn in or full runs, just gets rid of the mess in the run files
    """

    f = open(burn_file_path + '.txt', "w")
    f.close()

    for i, result in tqdm(enumerate(sampler.sample(p0, iterations=nsamples, store=False))):         #USED TO BE STORECHAIN, NOW CALLED STORE
        # for i, result in enumerate(sampler.sample(p0, iterations=nsamples_burnin, store=False)):
        pos, prob, state = result
        f = open(burn_file_path + '.txt', "a")
        for k in range(pos.shape[0]):
            f.write(" ".join(map(str, np.append(pos[k], [prob[k], k]))))
            f.write("\n")
        f.close()
        print(i, "Currently performing burn-in")

        if test_plotting:
            if (i + 1) % n_steps_to_plot == 0:
                print("{0:5.1%}".format(float(i) / nsamples))
                chain_total = np.loadtxt(burn_file_path + '.txt')

                nsteps = int(len(chain_total) / nwalkers)
                nparams = chain_total.shape[-1] - 2

                chain_by_walker = []

                for k in range(nwalkers):
                    chain_by_walker.append(chain_total[chain_total[:, -1] == k])

                chain_by_walker = np.array(chain_by_walker)
                if plot_walkers:
                    walker_plot(chain_by_walker, nwalkers, 0, "vesc_" + extratext, plots_dir=plots_dir)
                    walker_plot(chain_by_walker, nwalkers, 1, "k_" + extratext, plots_dir=plots_dir)
                    walker_plot(chain_by_walker, nwalkers, 2, "frac_" + extratext, plots_dir=plots_dir)
                    walker_plot(chain_by_walker, nwalkers, 3, "sigma_" + extratext, plots_dir=plots_dir)

                    if sausage:
                        walker_plot(chain_by_walker, nwalkers, 4, "k_sausage_" + extratext, plots_dir=plots_dir)
                        walker_plot(chain_by_walker, nwalkers, 5, "frac_sausage_" + extratext, plots_dir=plots_dir)

                samples = chain_by_walker[:, :, :-2]
                samples = samples.reshape((-1, ndim))
                make_corner_plot(samples, 'corner_plot_' + extratext, plots_dir=plots_dir, true_values=true_values,
                                 inverse_vesc_prior=inverse_vesc_prior, three_functions=three_functions,
                                 two_vesc=two_vesc)

                samples = samples.reshape((-1, ndim))
                np.save(chains_dir + extratext + '.npy', samples)

    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

    walker_plot(chain_by_walker, nwalkers, 0, "vesc_" + extratext, plots_dir=plots_dir)
    walker_plot(chain_by_walker, nwalkers, 1, "k_" + extratext, plots_dir=plots_dir)

    if cross_correlation > 1:
        print("Getting rid of cross correlations")
        samples = sampler.chain[:, ::cross_correlation]

    samples = samples.reshape((-1, ndim))
    print("Shape of samples", samples.shape)

    np.save(chains_dir + extratext + '.npy', samples)
    return pos


def make_resulting_vesc_k_plot(
        vmin_list=[300, 325, 350, 375, 400],
        make_plot=False,
        k_mock=3.5,
        k_sausage=1.0,
        frac_sausage=0.6,
        fs_sausage_min=0.0,
        fs_sausage_max=1.0,
        error_range=0,
        sausage_generation=1,
        vesc_mock=500,
        error_mag=0.1,
        error_type='percent',
        mock=1,
        rgcmin=7.0,
        rgcmax=9.0,
        zmin=0.0,
        zmax=15.0,
        vphicut=1,
        limited_prior=0,
        v_min_plot=400,
        v_max_plot=700,
        k_min_plot=0,
        k_max_plot=9,
        accreted=0,
        cutoff=0.95,
        include_aic=0,
        kpriormin_input=0.0,
        kpriormax_input=15.0,
        f_outlier=0.01):
    """
    Returns a figure with 5 panels, that summarizes the results with a single and multiple function fit.
    :param: vmin_list: list of vmins to go through
    :param: make_plot: boolean, if true, saves the figure plot as well. That might not really work because of multiple figures open at the same time... #TODO Rethink
    :param: k_mock: the true mock value of k
    :param: k_sausage: the true mock value of k sausage
    :param: frac_sausage: the true sausage fraction
    :param: fs_sausage_min: minimum fraction prior
    :param: fs_sausage_max: maximum fraction prior
    :param: error_range: if the errors are spread out across a range that's capped, instead of fixed.
    :param: sausage_generation: whether this is a single or 2 functions generated
    :param: vesc_mock: true escape velocity
    :param: error_mag: magnitude of the errors
    :param: error_type: type of errors
    :param: mock: boolean, if true it runs the mocks, if false, it runs the data
    :param: rgcmin = 7, minimum distance cut for Gaia runs
    :param: rgcmax = 9, maximum distance cut for Gaia runs
    :param: zmin = 0, minimum vertical cut for Gaia runs
    :param: zmax = 15, maximum vertical cut for Gaia runs
    :param: vphicut = 1, retrograde cut
    :param: two_vesc = 0, making 2 vesc for the fit. I don't really use this
    :param: limited_prior = 0, boolean, if true, limits the priors on k.
    :param: v_min_plot = 400, vmin of the plots
    :param: v_max_plot = 700, vmax for the plots
    :param: k_min_plot = 0, k min for the plots
    :param: k_max_plot = 9, k max for the plots
    :param: accreted = 0, boolean, if true, pulls up the accreted dataset
    :param: cutoff = 0.95, float, accreted score cut
    :param: include_aic: boolean, if true, includes the aic plot as the last panel
    :param: kmin: float, minimum k for the prior on k
    :param: kmax: float, maximum k for the prior on k
    :param: f_outlier: float, fraction of the outlier, to make sure it's reading the right file
    """

    jobs_dir = '../jobs/'
    chains_dir = '../chains/'
    data_dir = '../data/'
    plots_dir = '../plots/'

    fontsize = 14

    mpl.rcParams.update({'font.size': fontsize, 'font.family': 'serif'})

    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['patch.linewidth'] = 1
    # mpl.rcParams['figure.figsize'] = 5, 3

    fig = plt.figure(figsize=(9, 6))

    filename = ''
    if mock:
        filename = '_inv_k_mock' + str(k_mock) + '_frac_' + str(frac_sausage)
    if error_range:
        filename += 'err_range'

    if limited_prior:
        filename += '_k_prior_' + str(kpriormin_input) + '_' + str(kpriormax_input)

    if include_aic:
        return_bic = False
        return_aic = True
        delta_bic_list = np.zeros_like(vmin_list)

    for s, vmin in enumerate(vmin_list):

        ax = fig.add_subplot(2, 3, s + 1)

        ax.minorticks_on()
        ax.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
        ax.tick_params('both', length=4, width=1, which='minor', direction='in', labelsize=fontsize)

        print("vmin", vmin)
        sausage = False
        two_vesc = False
        inverse_vesc_prior = True
        if mock:
            ndata = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot, k_mock=k_mock,
                                      k_sausage=k_sausage, frac_sausage=frac_sausage, fs_sausage_min=fs_sausage_min,
                                      fs_sausage_max=fs_sausage_max, return_n_data=True,
                                      inverse_vesc_prior=inverse_vesc_prior, sausage_generation=sausage_generation,
                                      vesc_mock=vesc_mock, error_mag=error_mag, error_type=error_type,
                                      error_range=error_range, limited_prior=limited_prior,
                                      kpriormin_input=kpriormin_input, kpriormax_input=kpriormax_input,
                                      f_outlier=f_outlier)
            # try:
            vesc, k = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot, k_mock=k_mock,
                                        k_sausage=k_sausage, frac_sausage=frac_sausage, fs_sausage_min=fs_sausage_min,
                                        fs_sausage_max=fs_sausage_max, mean_values=False, return_vesc_k=True,
                                        inverse_vesc_prior=inverse_vesc_prior, sausage_generation=sausage_generation,
                                        vesc_mock=vesc_mock, error_mag=error_mag, error_type=error_type,
                                        error_range=error_range, limited_prior=limited_prior,
                                        kpriormin_input=kpriormin_input, kpriormax_input=kpriormax_input,
                                        f_outlier=f_outlier)

            if include_aic:
                aic_a = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot, k_mock=k_mock,
                                          k_sausage=k_sausage, frac_sausage=frac_sausage, fs_sausage_min=fs_sausage_min,
                                          fs_sausage_max=fs_sausage_max, return_n_data=False,
                                          inverse_vesc_prior=inverse_vesc_prior, sausage_generation=sausage_generation,
                                          vesc_mock=vesc_mock, error_mag=error_mag, error_type=error_type,
                                          error_range=error_range, return_bic=return_bic, return_aic=return_aic,
                                          limited_prior=limited_prior, kpriormin_input=kpriormin_input,
                                          kpriormax_input=kpriormax_input, f_outlier=f_outlier)

        else:
            ndata = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                 two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot, return_ndata=True,
                                 inverse_vesc_prior=inverse_vesc_prior, error_type=error_type, accreted=accreted,
                                 cutoff=cutoff)
            # try:
            vesc, k = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                   two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                   return_ndata=False, inverse_vesc_prior=inverse_vesc_prior, error_type=error_type,
                                   return_vesc_k=True, mean_values=False, accreted=accreted, cutoff=cutoff)

            if include_aic:
                aic_a = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                     two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                     return_ndata=False, error_type=error_type, return_vesc_k=False, mean_values=False,
                                     inverse_vesc_prior=inverse_vesc_prior, return_bic=return_bic,
                                     return_aic=return_aic, accreted=accreted, cutoff=cutoff)

        corner.hist2d(vesc, k, bins=20, ax=ax, labels=[r'$v_{\rm{esc}}$', r'$k$'], plot_datapoints=False, color='blue',
                      alpha=0.1, levels=[0.68], plot_density=True)  # [0.68, 0.95]
        # except:
        # None

        if mock:
            if sausage_generation:
                sausage = True
                two_vesc = False
                inverse_vesc_prior = True

                vesc, k = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot,
                                            k_mock=k_mock, k_sausage=k_sausage, frac_sausage=frac_sausage,
                                            fs_sausage_min=fs_sausage_min, fs_sausage_max=fs_sausage_max,
                                            mean_values=False, return_vesc_k=True,
                                            inverse_vesc_prior=inverse_vesc_prior,
                                            sausage_generation=sausage_generation, vesc_mock=vesc_mock,
                                            error_mag=error_mag, error_type=error_type, error_range=error_range,
                                            limited_prior=limited_prior, kpriormin_input=kpriormin_input,
                                            kpriormax_input=kpriormax_input)

                corner.hist2d(vesc, k, bins=20, ax=ax, labels=[r'$v_{\rm{esc}}$', r'$k$'], plot_datapoints=False,
                              color='red', alpha=0.1, levels=[0.68], plot_density=True)  # 0.68,

                if include_aic:
                    aic_b = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot,
                                              k_mock=k_mock, k_sausage=k_sausage, frac_sausage=frac_sausage,
                                              fs_sausage_min=fs_sausage_min, fs_sausage_max=fs_sausage_max,
                                              return_n_data=False, inverse_vesc_prior=inverse_vesc_prior,
                                              sausage_generation=sausage_generation, vesc_mock=vesc_mock,
                                              error_mag=error_mag, error_type=error_type, error_range=error_range,
                                              return_bic=return_bic, return_aic=return_aic, limited_prior=limited_prior,
                                              kpriormin_input=kpriormin_input, kpriormax_input=kpriormax_input)

                    delta_bic_list[s] = aic_b - aic_a

        else:
            sausage = True
            two_vesc = False
            inverse_vesc_prior = True

            vesc, k = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                   two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                   return_ndata=False, error_type=error_type, return_vesc_k=True, mean_values=False,
                                   inverse_vesc_prior=inverse_vesc_prior, accreted=accreted, cutoff=cutoff)

            corner.hist2d(vesc, k, bins=20, ax=ax, labels=[r'$v_{\rm{esc}}$', r'$k$'], plot_datapoints=False,
                          color='red', alpha=0.1, levels=[0.68], plot_density=True)

            if include_aic:
                aic_b = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                     two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                     return_ndata=False, error_type=error_type, return_vesc_k=False, mean_values=False,
                                     inverse_vesc_prior=inverse_vesc_prior, return_bic=return_bic,
                                     return_aic=return_aic, accreted=accreted, cutoff=cutoff)

                delta_bic_list[s] = aic_b - aic_a

        if mock:
            ax.axvline(x=vesc_mock, linestyle='--', color='gray')

            vesc_array = np.arange(v_min_plot, v_max_plot, 10)
            k_truth = np.array([k_sausage for _ in range(len(vesc_array))])

            ax.plot(vesc_array, k_truth, linestyle='--', color='gray')

        if s == 2:
            # Getting the Labels correct
            x_labels = [1000, 1050]
            y_labels = [1.0, 1.0]

            ax.plot(x_labels, y_labels, color='blue', label='Single Function')

            if sausage_generation:
                ax.plot(x_labels, y_labels, color='red', label='Two Functions')

            # ax.legend(bbox_to_anchor=(0.95, 0.65), frameon = True, fontsize = 12)
            ax.legend(loc='center right', frameon=True, fontsize=12)

        ax.text(0.05, 0.95, r'$N=$' + str(ndata) + '\n' + r'$v_{\rm{min}} = $' + str(int(vmin)) + ' km/s', ha='left',
                va='top', transform=ax.transAxes, fontsize=14)
        ax.set_xlabel(r'$v_{\rm{esc}}$ [km/s]', fontsize=12)
        ax.set_ylabel(r'$k$', fontsize=12)
        ax.set_xlim([v_min_plot, v_max_plot])
        ax.set_ylim([k_min_plot, k_max_plot])

    if include_aic:
        s += 1
        ax = fig.add_subplot(2, 3, s + 1)

        ax.minorticks_on()
        ax.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
        ax.tick_params('both', length=4, width=1, which='minor', direction='in', labelsize=fontsize)

        ax.plot(vmin_list, np.zeros_like(vmin_list), color='gray', linestyle='--')

        ax.plot(vmin_list, delta_bic_list, color='darkgreen')
        ax.set_ylim([-60, 10])
        ax.set_xlim([vmin_list[0], vmin_list[-1]])
        ax.set_xlabel(r'$v_{\rm{min}}$ [km/s]', fontsize=12)
        ax.set_ylabel(r'$\Delta$ AIC', fontsize=12)

    # # Getting the Labels correct
    # x_labels = [1000,1050]
    # y_labels = [1.0, 1.0]

    # ax.plot(x_labels, y_labels, color = 'blue', label = 'Single Function')

    # if sausage_generation:
    #     ax.plot(x_labels, y_labels, color = 'red', label = 'Two Functions')

    # if include_aic:
    #     ax.legend(bbox_to_anchor=(0.95, 0.65), frameon = False, fontsize = 12)
    # else:
    #     ax.legend(bbox_to_anchor=(2.0, 0.9), frameon = False, fontsize = 12)

    plt.tight_layout()

    if error_type == 'percent':
        error_text = r'Err$=$' + str(100 * error_mag) + r'$\%$'
    elif error_type == 'absolute':
        error_text = r'Err$=$' + str(error_mag) + ' km/s'
    elif error_type == 'no_errors':
        error_text = 'No Errors'

    if mock:
        title_text = r'\textbf{Simulations}'
        filename += '_mock_'
    else:
        if vphicut:
            title_text = r'\textbf{Gaia Retrograde Data}'
            filename += '_retro_data_'
        elif accreted:
            title_text = r'\textbf{Accreted Stars}' + r'$S>$' + str(cutoff)
            filename += '_accreted_' + str(int(100 * cutoff))
        else:
            title_text = r'\textbf{Gaia Data}'
            filename += '_data_'

    extratext = title_text + '\n' + error_text
    if mock:
        if include_aic:
            extratext += '\n' + r'$v_{\rm{esc}}$=' + str(
                int(vesc_mock)) + ' km/s' + ', ' + r'$k_S = %.1f$' % k_sausage + '\n' + r'$k=%.1f$' % k_mock + ', ' + r'$f=%.1f$' % frac_sausage

        else:
            extratext += '\n' + r'$v_{\rm{esc}}$ = ' + str(
                int(vesc_mock)) + ' km/s' + '\n' + r'$k_S = %.1f$' % k_sausage + '\n' + r'$k=%.1f$' % k_mock + '\n' + r'$f=%.1f$' % frac_sausage

    if include_aic:
        ax.text(308, -55, extratext, fontsize=12)
    else:
        ax.text(v_max_plot + 84, 0.1, extratext, fontsize=14)

    full_filename = plots_dir + 'vesc_k_different_methods' + filename + '_err_' + str(error_mag) + error_type + '.pdf'

    plt.savefig(full_filename)
    plt.close()


def make_violin_plot(vesc_list, edges, hist_single, hist_double, factor=1e-2):
    """
    Draws the violin plot of the single and double analysis of vesc
    :param: vesc_list: array of the escape velocities
    :param: edges: array of the edges of the historam to plot against
    :param: hist_single: histogram of the single analysis of vesc
    :param: hist_double: histogram of the double analysis of vesc
    """

    fig, ax = plt.subplots()
    colors = ['red', 'blue', 'green', 'cyan', 'pink']

    for i in range(len(vesc_list)):
        v = vesc_list[i]

        ax.fill_betweenx(edges, v - hist_single[i] * factor, v * np.ones(len(edges)), color=colors[i], alpha=0.8)
        ax.fill_betweenx(edges, v + hist_double[i] * factor, v * np.ones(len(edges)), color=colors[i], alpha=0.4)


def output_error_text(error_mag, error_type):
    """
    Writes out the error standard text for plot legends
    """
    if error_type == 'percent':
        error_text = r'Err$=$' + str(100 * error_mag) + r'$\%$'
    elif error_type == 'absolute':
        error_text = r'Err$=$' + str(error_mag) + ' km/s'
    elif error_type == 'no_errors':
        error_text = 'No Errors'

    return error_text


def make_resulting_vesc_violin_plot(
        vmin_list=[300, 325, 350, 375, 400],
        make_plot=False,
        k_mock=3.5,
        k_sausage=1.0,
        frac_sausage=0.6,
        fs_sausage_min=0.0,
        fs_sausage_max=1.0,
        error_range=0,
        sausage_generation=1,
        vesc_mock=500,
        error_mag=0.1,
        error_type='percent',
        mock=1,
        rgcmin=7.0,
        rgcmax=9.0,
        zmin=0.0,
        zmax=15.0,
        vphicut=1,
        limited_prior=0,
        v_min_plot=400,
        v_max_plot=700,
        k_min_plot=0,
        k_max_plot=9,
        accreted=0,
        cutoff=0.95,
        include_aic=0,
        include_bic=0,
        kpriormin_input=0.0,
        kpriormax_input=15.0,
        f_outlier=0.01,
        filename='',
        include_k_plot=False,
        k_bin=0.04,
        chains_dir='../chains/',
        plots_dir='../plots/',
        substructure=0,
        sub_mean=370,
        sub_dispersion=20,
        frac_substructure=0.2,
        three_functions=0,
        plot_factor=0.8,
        plot_extra_function=0,
        fire=0,
        simulation='m12i',
        lsr=0,
        dr2=1,
        edr3=0,
        plot_truth=0):
    """
    Returns a figure with 5 panels, that summarizes the results with a single and multiple function fit.
    :param: vmin_list: list of vmins to go through
    :param: make_plot: boolean, if true, saves the figure plot as well. That might not really work because of multiple figures open at the same time... #TODO Rethink
    :param: k_mock: the true mock value of k
    :param: k_sausage: the true mock value of k sausage
    :param: frac_sausage: the true sausage fraction
    :param: fs_sausage_min: minimum fraction prior
    :param: fs_sausage_max: maximum fraction prior
    :param: error_range: if the errors are spread out across a range that's capped, instead of fixed.
    :param: sausage_generation: whether this is a single or 2 functions generated
    :param: vesc_mock: true escape velocity
    :param: error_mag: magnitude of the errors
    :param: error_type: type of errors
    :param: mock: boolean, if true it runs the mocks, if false, it runs the data
    :param: rgcmin = 7, minimum distance cut for Gaia runs
    :param: rgcmax = 9, maximum distance cut for Gaia runs
    :param: zmin = 0, minimum vertical cut for Gaia runs
    :param: zmax = 15, maximum vertical cut for Gaia runs
    :param: vphicut = 1, retrograde cut
    :param: two_vesc = 0, making 2 vesc for the fit. I don't really use this
    :param: limited_prior = 0, boolean, if true, limits the priors on k.
    :param: v_min_plot = 400, vmin of the plots
    :param: v_max_plot = 700, vmax for the plots
    :param: k_min_plot = 0, k min for the plots
    :param: k_max_plot = 9, k max for the plots
    :param: accreted = 0, boolean, if true, pulls up the accreted dataset
    :param: cutoff = 0.95, float, accreted score cut
    :param: include_aic: boolean, if true, includes the aic plot as the last panel
    :param: kmin: float, minimum k for the prior on k
    :param: kmax: float, maximum k for the prior on k
    :param: f_outlier: float, fraction of the outlier, to make sure it's reading the right file
    :param: filename: string, in case i want to add another thing to the file name
    :param: include_k_plot: Boolean, if true, the plot will have two panels, the v and the k.
    :param: k_bin: double, size of the binning in k
    :param: chains_dir: string, for the location of the chains
    :param: plots_dir: string, for the location of the plots
    :param: substructure, boolean, if true, it includes another gaussian in the generation for an extra structure.
    :param: sub_mean: double for the mean of the new Gaussian
    :param: sub_dispersion: double for the dispersion of the new Gaussian
    :param: frac_substructure: double for the fraction of the new Gaussian
    :param: three_functions: boolean, if true compares the two function with the three function fits.
    :param: plot_factor: double, how much to plot the functions in as a fraction of the y axis of the plot
    :param: plot_extra_function: if true, plots the 3 function fit at vmin = 300 km/s.
    :param: fire: Boolean, if true this a fire run, not Gaia
    :param: simulation: string, should be 'm12i' or 'm12f'
    :param: lsr: int, 0,1,2 location of the Sun in the simulation
    :param: dr2: Gaia DR2
    :param: edr3: Gaia eDR3
    :param: plot_truth: boolean, if true, it would plot the band of true values
    """

    fontsize = 14

    mpl.rcParams.update({'font.size': fontsize, 'font.family': 'serif'})

    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['patch.linewidth'] = 1
    # mpl.rcParams['figure.figsize'] = 5, 3

    if mock:
        filename += '_inv_k_mock' + str(k_mock) + '_frac_' + str(frac_sausage)
    if error_range:
        filename += 'err_range'

    if limited_prior:
        filename += '_k_prior_' + str(kpriormin_input) + '_' + str(kpriormax_input)

    if include_aic:
        return_bic = False
        return_aic = True
        delta_bic_list = np.zeros_like(vmin_list)

    if (include_aic or include_bic or include_k_plot):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(2, 1, 1)

    else:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)

    ax.minorticks_on()
    ax.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
    ax.tick_params('both', length=4, width=1, which='minor', direction='in', labelsize=fontsize)

    dataset = {}
    dataset_k = {}
    dataset_k3 = {}
    ndatasets = np.zeros(len(vmin_list))

    for s, vmin in enumerate(vmin_list):

        print("vmin", vmin)
        sausage = False
        two_vesc = False
        inverse_vesc_prior = True
        if mock:
            ndata = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot, k_mock=k_mock,
                                      k_sausage=k_sausage, frac_sausage=frac_sausage, fs_sausage_min=fs_sausage_min,
                                      fs_sausage_max=fs_sausage_max, return_n_data=True,
                                      inverse_vesc_prior=inverse_vesc_prior, sausage_generation=sausage_generation,
                                      vesc_mock=vesc_mock, error_mag=error_mag, error_type=error_type,
                                      error_range=error_range, limited_prior=limited_prior,
                                      kpriormin_input=kpriormin_input, kpriormax_input=kpriormax_input,
                                      f_outlier=f_outlier, substructure=substructure, sub_mean=sub_mean,
                                      sub_dispersion=sub_dispersion, frac_substructure=frac_substructure)

            ndatasets[s] = ndata
            # try:
            vesc, k = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot, k_mock=k_mock,
                                        k_sausage=k_sausage, frac_sausage=frac_sausage, fs_sausage_min=fs_sausage_min,
                                        fs_sausage_max=fs_sausage_max, mean_values=False, return_vesc_k=True,
                                        inverse_vesc_prior=inverse_vesc_prior, sausage_generation=sausage_generation,
                                        vesc_mock=vesc_mock, error_mag=error_mag, error_type=error_type,
                                        error_range=error_range, limited_prior=limited_prior,
                                        kpriormin_input=kpriormin_input, kpriormax_input=kpriormax_input,
                                        f_outlier=f_outlier, substructure=substructure, sub_mean=sub_mean,
                                        sub_dispersion=sub_dispersion, frac_substructure=frac_substructure)

            vesc_single = vesc
            k_single = k

            if include_aic:
                aic_a = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot, k_mock=k_mock,
                                          k_sausage=k_sausage, frac_sausage=frac_sausage, fs_sausage_min=fs_sausage_min,
                                          fs_sausage_max=fs_sausage_max, return_n_data=False,
                                          inverse_vesc_prior=inverse_vesc_prior, sausage_generation=sausage_generation,
                                          vesc_mock=vesc_mock, error_mag=error_mag, error_type=error_type,
                                          error_range=error_range, return_bic=return_bic, return_aic=return_aic,
                                          limited_prior=limited_prior, kpriormin_input=kpriormin_input,
                                          kpriormax_input=kpriormax_input, f_outlier=f_outlier,
                                          substructure=substructure, sub_mean=sub_mean, sub_dispersion=sub_dispersion,
                                          frac_substructure=frac_substructure)

        else:
            ndata = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                 two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot, return_ndata=True,
                                 inverse_vesc_prior=inverse_vesc_prior, error_type=error_type, accreted=accreted,
                                 cutoff=cutoff, chains_dir=chains_dir, fire=fire, simulation=simulation, lsr=lsr,
                                 plots_dir=plots_dir, edr3=edr3)

            ndatasets[s] = ndata
            # try:
            vesc, k = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                   two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                   return_ndata=False, inverse_vesc_prior=inverse_vesc_prior, error_type=error_type,
                                   return_vesc_k=True, mean_values=False, accreted=accreted, cutoff=cutoff,
                                   chains_dir=chains_dir, fire=fire, simulation=simulation, lsr=lsr,
                                   plots_dir=plots_dir,
                                   edr3=edr3)

            vesc_single = vesc
            k_single = k

            if include_aic:
                aic_a = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                     two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                     return_ndata=False, error_type=error_type, return_vesc_k=False, mean_values=False,
                                     inverse_vesc_prior=inverse_vesc_prior, return_bic=return_bic,
                                     return_aic=return_aic, accreted=accreted, cutoff=cutoff, chains_dir=chains_dir,
                                     fire=fire, lsr=lsr, simulation=simulation, plots_dir=plots_dir, edr3=edr3)

        if mock:
            if sausage_generation:
                sausage = True
                two_vesc = False
                inverse_vesc_prior = True

                vesc, k = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot,
                                            k_mock=k_mock, k_sausage=k_sausage, frac_sausage=frac_sausage,
                                            fs_sausage_min=fs_sausage_min, fs_sausage_max=fs_sausage_max,
                                            mean_values=False, return_vesc_k=True,
                                            inverse_vesc_prior=inverse_vesc_prior,
                                            sausage_generation=sausage_generation, vesc_mock=vesc_mock,
                                            error_mag=error_mag, error_type=error_type, error_range=error_range,
                                            limited_prior=limited_prior, kpriormin_input=kpriormin_input,
                                            kpriormax_input=kpriormax_input, substructure=substructure,
                                            sub_mean=sub_mean, sub_dispersion=sub_dispersion,
                                            frac_substructure=frac_substructure)

                vesc_double = vesc

                k, kS = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot, k_mock=k_mock,
                                          k_sausage=k_sausage, frac_sausage=frac_sausage, fs_sausage_min=fs_sausage_min,
                                          fs_sausage_max=fs_sausage_max, mean_values=False, return_vesc_k=False,
                                          return_k=True, inverse_vesc_prior=inverse_vesc_prior,
                                          sausage_generation=sausage_generation, vesc_mock=vesc_mock,
                                          error_mag=error_mag, error_type=error_type, error_range=error_range,
                                          limited_prior=limited_prior, kpriormin_input=kpriormin_input,
                                          kpriormax_input=kpriormax_input, substructure=substructure, sub_mean=sub_mean,
                                          sub_dispersion=sub_dispersion, frac_substructure=frac_substructure)

                k_double_one = k
                k_double_S = kS

                if include_aic:
                    aic_b = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot,
                                              k_mock=k_mock, k_sausage=k_sausage, frac_sausage=frac_sausage,
                                              fs_sausage_min=fs_sausage_min, fs_sausage_max=fs_sausage_max,
                                              return_n_data=False, inverse_vesc_prior=inverse_vesc_prior,
                                              sausage_generation=sausage_generation, vesc_mock=vesc_mock,
                                              error_mag=error_mag, error_type=error_type, error_range=error_range,
                                              return_bic=return_bic, return_aic=return_aic, limited_prior=limited_prior,
                                              kpriormin_input=kpriormin_input, kpriormax_input=kpriormax_input,
                                              substructure=substructure, sub_mean=sub_mean,
                                              sub_dispersion=sub_dispersion, frac_substructure=frac_substructure)

                    delta_bic_list[s] = aic_b - aic_a

        else:
            sausage = True
            two_vesc = False
            inverse_vesc_prior = True

            vesc, k = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                   two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                   return_ndata=False, error_type=error_type, return_vesc_k=True, mean_values=False,
                                   inverse_vesc_prior=inverse_vesc_prior, accreted=accreted, cutoff=cutoff,
                                   chains_dir=chains_dir, fire=fire, lsr=lsr, simulation=simulation,
                                   plots_dir=plots_dir,
                                   edr3=edr3)

            vesc_double = vesc

            k, kS = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                 two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                 return_ndata=False, error_type=error_type, return_vesc_k=False, return_k=True,
                                 mean_values=False, inverse_vesc_prior=inverse_vesc_prior, accreted=accreted,
                                 cutoff=cutoff, chains_dir=chains_dir, fire=fire, lsr=lsr, simulation=simulation,
                                 plots_dir=plots_dir, edr3=edr3)

            k_double_one = k
            k_double_S = kS

            if three_functions:
                vesc, k = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                       two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                       return_ndata=False, error_type=error_type, return_vesc_k=True, mean_values=False,
                                       inverse_vesc_prior=inverse_vesc_prior, accreted=accreted, cutoff=cutoff,
                                       chains_dir=chains_dir, three_functions=three_functions, fire=fire, lsr=lsr,
                                       plots_dir=plots_dir, edr3=edr3)

                vesc_three = vesc

                k, kS, k_3 = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                          two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                          return_ndata=False, inverse_vesc_prior=inverse_vesc_prior,
                                          error_type=error_type, return_vesc_k=False, return_k=True, mean_values=False,
                                          accreted=accreted, cutoff=cutoff, chains_dir=chains_dir,
                                          three_functions=three_functions, fire=fire, lsr=lsr, simulation=simulation,
                                          plots_dir=plots_dir, edr3=edr3)

                k_three_one = k
                k_three_two = kS
                k_three_three = k_3

            if include_aic:
                aic_b = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                     two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                     return_ndata=False, error_type=error_type, return_vesc_k=False, mean_values=False,
                                     inverse_vesc_prior=inverse_vesc_prior, return_bic=return_bic,
                                     return_aic=return_aic, accreted=accreted, cutoff=cutoff, chains_dir=chains_dir,
                                     fire=fire, lsr=lsr, simulation=simulation, plots_dir=plots_dir, edr3=edr3)

                delta_bic_list[s] = aic_b - aic_a

        if three_functions:
            dataset[str(vmin)] = [vesc_single, vesc_double, vesc_three]
            dataset_k3[str(vmin)] = [k_three_one, k_three_two, k_three_three]
        else:
            dataset[str(vmin)] = [vesc_single, vesc_double]

        dataset_k[str(vmin)] = [k_single, k_double_one, k_double_S]

        if mock:
            vstep = 10

            vmin_array = np.arange(vmin_list[0] - 25, vmin_list[-1] + 55, vstep)
            vesc_array = np.array([vesc_mock for _ in range(len(vmin_array))])

            ax.plot(vmin_array, vesc_array, linestyle='--', color='gray')

        if s == 2:
            # Getting the Labels correct
            x_labels = [1000, 1050]
            y_labels = [1000, 1050]

            if three_functions:
                ax.plot(x_labels, y_labels, color='darkviolet', label='Three Functions')
            else:
                ax.plot(x_labels, y_labels, color='blue', label='Single Function')

            if sausage_generation:
                ax.plot(x_labels, y_labels, color='red', label='Two Functions')

            if plot_extra_function:
                ax.plot(x_labels, y_labels, color='green', label='Three Functions')
            # ax.legend(bbox_to_anchor=(0.95, 0.65), frameon = True, fontsize = 12)

            if substructure:
                ax.legend(loc='upper left', ncol=1, frameon=False, fontsize=13, handlelength=1)
            else:
                ax.legend(loc='upper left', ncol=2, frameon=False, fontsize=13, handlelength=1)

        ax.text(vmin, v_min_plot * 1.01, r'$N=$' + str(int(ndatasets[s])), ha='left', va='bottom', fontsize=12,
                rotation='horizontal', color='darkgreen')

    max_value = 0
    binning = np.arange(v_min_plot, v_max_plot * plot_factor, 3)

    for vmin in vmin_list:

        if three_functions:
            [vesc1, vesc2, vesc3] = dataset[str(vmin)]

            hist_single, edges = np.histogram(vesc1, bins=binning, density=True)
            hist_double, edges = np.histogram(vesc2, bins=binning, density=True)
            hist_three, edges = np.histogram(vesc3, bins=binning, density=True)

            real_bins = (edges[:-1] + edges[1:]) / 2

            # Here I'm making a choice of comparing vesc2 to vesc3
            current_max_value = np.max([np.max(hist_three), np.max(hist_double)])

            if current_max_value > max_value:
                max_value = current_max_value

        else:
            [vesc1, vesc2] = dataset[str(vmin)]

            hist_single, edges = np.histogram(vesc1, bins=binning, density=True)
            hist_double, edges = np.histogram(vesc2, bins=binning, density=True)

            real_bins = (edges[:-1] + edges[1:]) / 2

            current_max_value = np.max([np.max(hist_single), np.max(hist_double)])

            if current_max_value > max_value:
                max_value = current_max_value

    factor = ((vmin_list[1] - vmin_list[0]) / 2.) / max_value

    for vmin in vmin_list:
        if three_functions:
            [vesc1, vesc2, vesc3] = dataset[str(vmin)]

            hist_double, edges = np.histogram(vesc2, bins=binning, density=True)
            hist_three, edges = np.histogram(vesc3, bins=binning, density=True)

            real_bins = (edges[:-1] + edges[1:]) / 2

            ax.fill_betweenx(real_bins, vmin - hist_three * factor, vmin * np.ones(len(real_bins)), color='darkviolet',
                             alpha=0.4)
            ax.fill_betweenx(real_bins, vmin + hist_double * factor, vmin * np.ones(len(real_bins)), color='red',
                             alpha=0.8)

        else:
            [vesc1, vesc2] = dataset[str(vmin)]

            hist_single, edges = np.histogram(vesc1, bins=binning, density=True)
            hist_double, edges = np.histogram(vesc2, bins=binning, density=True)

            real_bins = (edges[:-1] + edges[1:]) / 2

            ax.fill_betweenx(real_bins, vmin - hist_single * factor, vmin * np.ones(len(real_bins)), color='blue',
                             alpha=0.4)
            ax.fill_betweenx(real_bins, vmin + hist_double * factor, vmin * np.ones(len(real_bins)), color='red',
                             alpha=0.8)

    # ax = sns.violinplot(x=r'$v_{\rm{min}}$ [km/s]', y=r'$v_{\rm{esc}}$ [km/s]', hue="smoker", data=dataset, palette="muted", split=True)

    if plot_extra_function:
        # Adding the three function fit at vmin = 300 km/s

        vmin = 300
        sausage_3 = True
        three_functions_3 = True

        vesc, k = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage_3, vphicut=vphicut,
                               two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot, return_ndata=False,
                               error_type=error_type, return_vesc_k=True, mean_values=False,
                               inverse_vesc_prior=inverse_vesc_prior, accreted=accreted, cutoff=cutoff,
                               chains_dir=chains_dir, three_functions=three_functions_3, fire=fire, lsr=lsr,
                               plots_dir=plots_dir, edr3=edr3)

        vesc_three = vesc

        hist_three, edges = np.histogram(vesc_three, bins=binning, density=True)

        real_bins = (edges[:-1] + edges[1:]) / 2
        ax.fill_betweenx(real_bins, vmin - hist_three * factor, vmin * np.ones(len(real_bins)), color='green',
                         alpha=0.4, linestyle='--', linewidth=1)

    ax.set_ylabel(r'$v_{\rm{esc}}$ [km/s]', fontsize=14)

    ax.set_xticks(vmin_list)  # Set label locations.

    if not (include_k_plot):
        ax.set_xlabel(r'$v_{\rm{min}}$ [km/s]', fontsize=14)
    spacing = vmin_list[1] - vmin_list[0]

    ax.set_xlim([vmin_list[0] - spacing / 2, vmin_list[-1] + spacing / 2])
    ax.set_ylim([v_min_plot, v_max_plot])

    if include_k_plot:

        max_value = 0
        binning = np.arange(k_min_plot, k_max_plot, k_bin)

        ax2 = fig.add_subplot(212, sharex=ax)

        ax2.minorticks_on()
        ax2.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
        ax2.tick_params('both', length=4, width=1, which='minor', direction='in', labelsize=fontsize)

        for vmin in vmin_list:

            if three_functions:
                [k_three_one, k_three_two, k_three_three] = dataset_k3[str(vmin)]

                hist_three_one, edges = np.histogram(k_three_one, bins=binning, density=True)
                hist_three_two, edges = np.histogram(k_three_two, bins=binning, density=True)
                hist_three_three, edges = np.histogram(k_three_three, bins=binning, density=True)

                real_bins = (edges[:-1] + edges[1:]) / 2

                current_max_value = np.max([np.max(hist_three_one), np.max(hist_three_two), np.max(hist_three_three)])

                if current_max_value > max_value:
                    max_value = current_max_value

            else:
                [k_single, k_double_one, k_double_S] = dataset_k[str(vmin)]

                hist_single, edges = np.histogram(k_single, bins=binning, density=True)
                hist_double_one, edges = np.histogram(k_double_one, bins=binning, density=True)
                hist_double_S, edges = np.histogram(k_double_S, bins=binning, density=True)

                real_bins = (edges[:-1] + edges[1:]) / 2

                current_max_value = np.max([np.max(hist_single), np.max(hist_double_one), np.max(hist_double_S)])

                if current_max_value > max_value:
                    max_value = current_max_value

        factor = ((vmin_list[1] - vmin_list[0]) / 2.) / max_value

        for vmin in vmin_list:
            if three_functions:
                [k_three_one, k_three_two, k_three_three] = dataset_k3[str(vmin)]

                hist_three_one, edges = np.histogram(k_three_one, bins=binning, density=True)
                hist_three_two, edges = np.histogram(k_three_two, bins=binning, density=True)
                hist_three_three, edges = np.histogram(k_three_three, bins=binning, density=True)

                real_bins = (edges[:-1] + edges[1:]) / 2

                ax2.fill_betweenx(real_bins, vmin - hist_three_one * factor, vmin * np.ones(len(real_bins)),
                                  color='lime', alpha=0.4)

                ax2.fill_betweenx(real_bins, vmin - hist_three_two * factor, vmin * np.ones(len(real_bins)),
                                  color='green', alpha=0.4)

                ax2.fill_betweenx(real_bins, vmin - hist_three_three * factor, vmin * np.ones(len(real_bins)),
                                  color='turquoise', alpha=0.4)

                [k_single, k_double_one, k_double_S] = dataset_k[str(vmin)]

                hist_double_one, edges = np.histogram(k_double_one, bins=binning, density=True)
                hist_double_S, edges = np.histogram(k_double_S, bins=binning, density=True)

                real_bins = (edges[:-1] + edges[1:]) / 2

                ax2.fill_betweenx(real_bins, vmin + hist_double_one * factor, vmin * np.ones(len(real_bins)),
                                  color='crimson', alpha=0.4)

                ax2.fill_betweenx(real_bins, vmin + hist_double_S * factor, vmin * np.ones(len(real_bins)),
                                  color='lightsalmon', alpha=0.4)

            else:
                [k_single, k_double_one, k_double_S] = dataset_k[str(vmin)]

                hist_single, edges = np.histogram(k_single, bins=binning, density=True)
                hist_double_one, edges = np.histogram(k_double_one, bins=binning, density=True)
                hist_double_S, edges = np.histogram(k_double_S, bins=binning, density=True)

                real_bins = (edges[:-1] + edges[1:]) / 2

                ax2.fill_betweenx(real_bins, vmin - hist_single * factor, vmin * np.ones(len(real_bins)), color='blue',
                                  alpha=0.4)

                ax2.fill_betweenx(real_bins, vmin + hist_double_one * factor, vmin * np.ones(len(real_bins)),
                                  color='crimson', alpha=0.4)

                ax2.fill_betweenx(real_bins, vmin + hist_double_S * factor, vmin * np.ones(len(real_bins)),
                                  color='lightsalmon', alpha=0.4)

        if mock:
            vstep = 10

            vmin_array = np.arange(vmin_list[0] - 25, vmin_list[-1] + 55, vstep)

            k_array = np.array([k_mock for _ in range(len(vmin_array))])

            ax2.plot(vmin_array, k_array, linestyle='--', color='gray')

            k_array = np.array([k_sausage for _ in range(len(vmin_array))])

            ax2.plot(vmin_array, k_array, linestyle='--', color='gray')

        ax2.set_ylabel(r'$k$', fontsize=14)
        ax2.set_xlabel(r'$v_{\rm{min}}$ [km/s]', fontsize=14)
        spacing = vmin_list[1] - vmin_list[0]

        ax2.set_xticks(vmin_list)  # Set label locations.

        ax2.set_xlim([vmin_list[0] - spacing / 2, vmin_list[-1] + spacing / 2])
        ax2.set_ylim([k_min_plot, k_max_plot])

        filename += '_k_plot'
        if three_functions:
            filename += '_3func'

    if include_aic:
        ax2 = fig.add_subplot(2, 1, 2)

        ax2.minorticks_on()
        ax2.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
        ax2.tick_params('both', length=4, width=1, which='minor', direction='in', labelsize=fontsize)

        ax2.plot(vmin_list, np.zeros_like(vmin_list), color='gray', linestyle='--')

        ax2.plot(vmin_list, delta_bic_list, color='darkviolet')
        ax2.set_ylim([-60, 10])
        ax2.set_xlim([vmin_list[0], vmin_list[-1]])
        ax2.set_xlabel(r'$v_{\rm{min}}$ [km/s]', fontsize=12)
        ax2.set_ylabel(r'$\Delta$ AIC', fontsize=12)

    if fire:
        if plot_truth:
            data_dir = '../data/simulations/' + simulation + '/'
            [vesc_min, vesc_max] = np.load(data_dir + 'lsr_' + str(lsr) + '_vesc_based_potential.npy')
            ax.fill_between(vmin_list, vesc_min * np.ones_like(vmin_list), vesc_max * np.ones_like(vmin_list),
                            color='gray', alpha=0.2, zorder=0)

    plt.tight_layout()

    if error_type == 'percent':
        error_text = r'Err$=$' + str(100 * error_mag) + r'$\%$'
    elif error_type == 'absolute':
        error_text = r'Err$=$' + str(error_mag) + ' km/s'
    elif (error_type == 'no_errors') or (error_type == 'no_error'):
        error_text = 'No Errors'

    if mock:
        title_text = r'\textbf{Simulations}'
        filename += '_mock_'
    else:
        if dr2:
            title_text = r'\textbf{Gaia DR2}'
        elif edr3:
            title_text = r'\textbf{Gaia eDR3}'
            filename += 'eder3'
        elif fire:
            title_text = r'\textbf{FIRE, ' + simulation + ', LSR ' + str(lsr) + '}'
            filename += '_lsr_' + str(lsr)

        if vphicut:
            title_text += r'\textbf{ -Retrograde Data}'
            filename += '_retro_data_'
        elif accreted:
            title_text += r'\textbf{ -Accreted Stars}' + r', $S>$' + str(cutoff)
            filename += '_accreted_' + str(int(100 * cutoff))
        # else:
        #     title_text = r'\textbf{Gaia Data}'
        #     filename += '_data_'

    extratext = title_text + '\n' + error_text
    if mock:
        if include_aic:
            extratext += '\n' + r'$v_{\rm{esc}}$=' + str(
                int(vesc_mock)) + ' km/s' + ', ' + r'$k_S = %.1f$' % k_sausage + '\n' + r'$k=%.1f$' % k_mock + ', ' + r'$g_S=%.1f$' % frac_sausage

        else:
            extratext += ', ' + r'$v_{\rm{esc}}$ = ' + str(
                int(vesc_mock)) + ' km/s' + '\n' + r'$k_S = %.1f$' % k_sausage + ', ' + r'$k=%.1f$' % k_mock + ', ' + r'$g_S=%.1f$' % frac_sausage

    if include_aic:
        ax2.text(308, -55, extratext, fontsize=12)
    else:
        ax.text(0.95, 0.95, extratext, fontsize=14, verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes)
        if substructure:
            sub_text = 'Injection, ' + r'$\mu=%2d$ km/s,' % sub_mean + '\n' + r'$\sigma=%2d$ km/s,' % sub_dispersion + r'$f_i=%.2f$' % frac_substructure
            ax.text(vmin_list[1], v_max_plot * 0.93, sub_text, fontsize=14, verticalalignment='top')

    sim_text = ''
    if fire:
        sim_text = '_sim_' + simulation

    full_filename = plots_dir + 'vesc_violin_different_methods' + filename + sim_text + '_err_' + str(
        error_mag) + error_type + '.pdf'

    plt.savefig(full_filename)
    plt.close()


def make_resulting_k_violin_plot(
        vmin_list=[300, 325, 350, 375, 400],
        make_plot=False,
        k_mock=3.5,
        k_sausage=1.0,
        frac_sausage=0.6,
        fs_sausage_min=0.0,
        fs_sausage_max=1.0,
        error_range=0,
        sausage_generation=1,
        vesc_mock=500,
        error_mag=0.1,
        error_type='percent',
        mock=1,
        rgcmin=7.0,
        rgcmax=9.0,
        zmin=0.0,
        zmax=15.0,
        vphicut=1,
        limited_prior=0,
        v_min_plot=400,
        v_max_plot=700,
        k_min_plot=0,
        k_max_plot=9,
        accreted=0,
        cutoff=0.95,
        include_aic=0,
        include_bic=0,
        kpriormin_input=0.0,
        kpriormax_input=15.0,
        f_outlier=0.01,
        filename=''):
    """
    Returns a figure with 5 panels, that summarizes the results with a single and multiple function fit, but in this case with the posteriors of k.
    :param: vmin_list: list of vmins to go through
    :param: make_plot: boolean, if true, saves the figure plot as well. That might not really work because of multiple figures open at the same time... #TODO Rethink
    :param: k_mock: the true mock value of k
    :param: k_sausage: the true mock value of k sausage
    :param: frac_sausage: the true sausage fraction
    :param: fs_sausage_min: minimum fraction prior
    :param: fs_sausage_max: maximum fraction prior
    :param: error_range: if the errors are spread out across a range that's capped, instead of fixed.
    :param: sausage_generation: whether this is a single or 2 functions generated
    :param: vesc_mock: true escape velocity
    :param: error_mag: magnitude of the errors
    :param: error_type: type of errors
    :param: mock: boolean, if true it runs the mocks, if false, it runs the data
    :param: rgcmin = 7, minimum distance cut for Gaia runs
    :param: rgcmax = 9, maximum distance cut for Gaia runs
    :param: zmin = 0, minimum vertical cut for Gaia runs
    :param: zmax = 15, maximum vertical cut for Gaia runs
    :param: vphicut = 1, retrograde cut
    :param: two_vesc = 0, making 2 vesc for the fit. I don't really use this
    :param: limited_prior = 0, boolean, if true, limits the priors on k.
    :param: v_min_plot = 400, vmin of the plots
    :param: v_max_plot = 700, vmax for the plots
    :param: k_min_plot = 0, k min for the plots
    :param: k_max_plot = 9, k max for the plots
    :param: accreted = 0, boolean, if true, pulls up the accreted dataset
    :param: cutoff = 0.95, float, accreted score cut
    :param: include_aic: boolean, if true, includes the aic plot as the last panel
    :param: kmin: float, minimum k for the prior on k
    :param: kmax: float, maximum k for the prior on k
    :param: f_outlier: float, fraction of the outlier, to make sure it's reading the right file
    """

    jobs_dir = '../jobs/'
    chains_dir = '../chains/'
    data_dir = '../data/'
    plots_dir = '../plots/'

    import seaborn as sns

    fontsize = 14

    mpl.rcParams.update({'font.size': fontsize, 'font.family': 'serif'})

    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['patch.linewidth'] = 1
    # mpl.rcParams['figure.figsize'] = 5, 3

    if mock:
        filename = '_inv_k_mock' + str(k_mock) + '_frac_' + str(frac_sausage)
    if error_range:
        filename += 'err_range'

    if limited_prior:
        filename += '_k_prior_' + str(kpriormin_input) + '_' + str(kpriormax_input)

    if include_aic:
        return_bic = False
        return_aic = True
        delta_bic_list = np.zeros_like(vmin_list)

    if (include_aic or include_bic):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(2, 1, 1)

    else:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)

    ax.minorticks_on()
    ax.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
    ax.tick_params('both', length=4, width=1, which='minor', direction='in', labelsize=fontsize)

    dataset = {}
    ndatasets = np.zeros(len(vmin_list))

    for s, vmin in enumerate(vmin_list):

        print("vmin", vmin)
        sausage = False
        two_vesc = False
        inverse_vesc_prior = True
        if mock:
            ndata = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot, k_mock=k_mock,
                                      k_sausage=k_sausage, frac_sausage=frac_sausage, fs_sausage_min=fs_sausage_min,
                                      fs_sausage_max=fs_sausage_max, return_n_data=True,
                                      inverse_vesc_prior=inverse_vesc_prior, sausage_generation=sausage_generation,
                                      vesc_mock=vesc_mock, error_mag=error_mag, error_type=error_type,
                                      error_range=error_range, limited_prior=limited_prior,
                                      kpriormin_input=kpriormin_input, kpriormax_input=kpriormax_input,
                                      f_outlier=f_outlier)

            ndatasets[s] = ndata

            k = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot, k_mock=k_mock,
                                  k_sausage=k_sausage, frac_sausage=frac_sausage, fs_sausage_min=fs_sausage_min,
                                  fs_sausage_max=fs_sausage_max, mean_values=False, return_vesc_k=False, return_k=True,
                                  inverse_vesc_prior=inverse_vesc_prior, sausage_generation=sausage_generation,
                                  vesc_mock=vesc_mock, error_mag=error_mag, error_type=error_type,
                                  error_range=error_range, limited_prior=limited_prior, kpriormin_input=kpriormin_input,
                                  kpriormax_input=kpriormax_input, f_outlier=f_outlier)

            k_single = k

            if include_aic:
                aic_a = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot, k_mock=k_mock,
                                          k_sausage=k_sausage, frac_sausage=frac_sausage, fs_sausage_min=fs_sausage_min,
                                          fs_sausage_max=fs_sausage_max, return_n_data=False,
                                          inverse_vesc_prior=inverse_vesc_prior, sausage_generation=sausage_generation,
                                          vesc_mock=vesc_mock, error_mag=error_mag, error_type=error_type,
                                          error_range=error_range, return_bic=return_bic, return_aic=return_aic,
                                          limited_prior=limited_prior, kpriormin_input=kpriormin_input,
                                          kpriormax_input=kpriormax_input, f_outlier=f_outlier)

        else:
            ndata = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                 two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot, return_ndata=True,
                                 inverse_vesc_prior=inverse_vesc_prior, error_type=error_type, accreted=accreted,
                                 cutoff=cutoff)

            ndatasets[s] = ndata
            # try:
            vesc, k = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                   two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                   return_ndata=False, inverse_vesc_prior=inverse_vesc_prior, error_type=error_type,
                                   return_vesc_k=True, mean_values=False, accreted=accreted, cutoff=cutoff)

            k_single = k

            if include_aic:
                aic_a = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                     two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                     return_ndata=False, error_type=error_type, return_vesc_k=False, mean_values=False,
                                     inverse_vesc_prior=inverse_vesc_prior, return_bic=return_bic,
                                     return_aic=return_aic, accreted=accreted, cutoff=cutoff)

        if mock:
            if sausage_generation:
                sausage = True
                two_vesc = False
                inverse_vesc_prior = True

                k, kS = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot, k_mock=k_mock,
                                          k_sausage=k_sausage, frac_sausage=frac_sausage, fs_sausage_min=fs_sausage_min,
                                          fs_sausage_max=fs_sausage_max, mean_values=False, return_vesc_k=False,
                                          return_k=True, inverse_vesc_prior=inverse_vesc_prior,
                                          sausage_generation=sausage_generation, vesc_mock=vesc_mock,
                                          error_mag=error_mag, error_type=error_type, error_range=error_range,
                                          limited_prior=limited_prior, kpriormin_input=kpriormin_input,
                                          kpriormax_input=kpriormax_input)

                k_double_one = k
                k_double_S = kS

                if include_aic:
                    aic_b = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot,
                                              k_mock=k_mock, k_sausage=k_sausage, frac_sausage=frac_sausage,
                                              fs_sausage_min=fs_sausage_min, fs_sausage_max=fs_sausage_max,
                                              return_n_data=False, inverse_vesc_prior=inverse_vesc_prior,
                                              sausage_generation=sausage_generation, vesc_mock=vesc_mock,
                                              error_mag=error_mag, error_type=error_type, error_range=error_range,
                                              return_bic=return_bic, return_aic=return_aic, limited_prior=limited_prior,
                                              kpriormin_input=kpriormin_input, kpriormax_input=kpriormax_input)

                    delta_bic_list[s] = aic_b - aic_a

        else:
            sausage = True
            two_vesc = False
            inverse_vesc_prior = True

            k, kS = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                 two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                 return_ndata=False, error_type=error_type, return_vesc_k=False, return_k=True,
                                 mean_values=False, inverse_vesc_prior=inverse_vesc_prior, accreted=accreted,
                                 cutoff=cutoff)

            k_double_one = k
            k_double_S = kS

            if include_aic:
                aic_b = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                     two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                     return_ndata=False, error_type=error_type, return_vesc_k=False, mean_values=False,
                                     inverse_vesc_prior=inverse_vesc_prior, return_bic=return_bic,
                                     return_aic=return_aic, accreted=accreted, cutoff=cutoff)

                delta_bic_list[s] = aic_b - aic_a

        dataset[str(vmin)] = [k_single, k_double_one, k_double_S]

        if mock:
            vstep = 10

            vmin_array = np.arange(vmin_list[0] - 25, vmin_list[-1] + 55, vstep)

            k_array = np.array([k_mock for _ in range(len(vmin_array))])

            ax.plot(vmin_array, k_array, linestyle='--', color='gray')

            k_array = np.array([k_sausage for _ in range(len(vmin_array))])

            ax.plot(vmin_array, k_array, linestyle='--', color='gray')

        if s == 2:
            # Getting the Labels correct
            x_labels = [1000, 1050]
            y_labels = [1000, 1050]

            ax.plot(x_labels, y_labels, color='blue', label='Single Function')

            if sausage_generation:
                ax.plot(x_labels, y_labels, color='red', label='Two Functions')

            # ax.legend(bbox_to_anchor=(0.95, 0.65), frameon = True, fontsize = 12)
            ax.legend(loc='upper left', frameon=False, fontsize=14)

        ax.text(vmin, v_min_plot * 1.01, r'$N=$' + str(int(ndatasets[s])), ha='left', va='bottom', fontsize=12,
                rotation='horizontal')

    max_value = 0
    binning = np.arange(0.1, 15, 0.04)

    for vmin in vmin_list:

        [k_single, k_double_one, k_double_S] = dataset[str(vmin)]

        hist_single, edges = np.histogram(k_single, bins=binning, density=True)
        hist_double_one, edges = np.histogram(k_double_one, bins=binning, density=True)
        hist_double_S, edges = np.histogram(k_double_S, bins=binning, density=True)

        real_bins = (edges[:-1] + edges[1:]) / 2

        current_max_value = np.max([np.max(hist_single), np.max(hist_double_one), np.max(hist_double_S)])

        if current_max_value > max_value:
            max_value = current_max_value

    factor = ((vmin_list[1] - vmin_list[0]) / 2.) / max_value

    for vmin in vmin_list:
        [k_single, k_double_one, k_double_S] = dataset[str(vmin)]

        hist_single, edges = np.histogram(k_single, bins=binning, density=True)
        hist_double_one, edges = np.histogram(k_double_one, bins=binning, density=True)
        hist_double_S, edges = np.histogram(k_double_S, bins=binning, density=True)

        real_bins = (edges[:-1] + edges[1:]) / 2

        ax.fill_betweenx(real_bins, vmin - hist_single * factor, vmin * np.ones(len(real_bins)), color='blue',
                         alpha=0.4)

        ax.fill_betweenx(real_bins, vmin + hist_double_one * factor, vmin * np.ones(len(real_bins)), color='darkred',
                         alpha=0.4)

        ax.fill_betweenx(real_bins, vmin + hist_double_S * factor, vmin * np.ones(len(real_bins)), color='lightcoral',
                         alpha=0.8)

    # ax = sns.violinplot(x=r'$v_{\rm{min}}$ [km/s]', y=r'$v_{\rm{esc}}$ [km/s]', hue="smoker", data=dataset, palette="muted", split=True)

    ax.set_ylabel(r'$k$', fontsize=14)
    ax.set_xlabel(r'$v_{\rm{min}}$ [km/s]', fontsize=14)
    spacing = vmin_list[1] - vmin_list[0]

    ax.set_xlim([vmin_list[0] - spacing / 2, vmin_list[-1] + spacing / 2])
    ax.set_ylim([v_min_plot, v_max_plot])

    if include_aic:
        ax2 = fig.add_subplot(2, 1, 2)

        ax2.minorticks_on()
        ax2.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
        ax2.tick_params('both', length=4, width=1, which='minor', direction='in', labelsize=fontsize)

        ax2.plot(vmin_list, np.zeros_like(vmin_list), color='gray', linestyle='--')

        ax2.plot(vmin_list, delta_bic_list, color='darkgreen')
        ax2.set_ylim([-60, 10])
        ax2.set_xlim([vmin_list[0], vmin_list[-1]])
        ax2.set_xlabel(r'$v_{\rm{min}}$ [km/s]', fontsize=12)
        ax2.set_ylabel(r'$\Delta$ AIC', fontsize=12)

    # # Getting the Labels correct
    # x_labels = [1000,1050]
    # y_labels = [1.0, 1.0]

    # ax.plot(x_labels, y_labels, color = 'blue', label = 'Single Function')

    # if sausage_generation:
    #     ax.plot(x_labels, y_labels, color = 'red', label = 'Two Functions')

    # if include_aic:
    #     ax.legend(bbox_to_anchor=(0.95, 0.65), frameon = False, fontsize = 12)
    # else:
    #     ax.legend(bbox_to_anchor=(2.0, 0.9), frameon = False, fontsize = 12)

    plt.tight_layout()

    if error_type == 'percent':
        error_text = r'Err$=$' + str(100 * error_mag) + r'$\%$'
    elif error_type == 'absolute':
        error_text = r'Err$=$' + str(error_mag) + ' km/s'
    elif error_type == 'no_errors':
        error_text = 'No Errors'

    if mock:
        title_text = r'\textbf{Simulations}'
        filename += '_mock_'
    else:
        if vphicut:
            title_text = r'\textbf{Gaia Retrograde Data}'
            filename += '_retro_data_'
        elif accreted:
            title_text = r'\textbf{Accreted Stars}' + r'$S>$' + str(cutoff)
            filename += '_accreted_' + str(int(100 * cutoff))
        else:
            title_text = r'\textbf{Gaia Data}'
            filename += '_data_'

    extratext = title_text + '\n' + error_text
    if mock:
        if include_aic:
            extratext += '\n' + r'$v_{\rm{esc}}$=' + str(
                int(vesc_mock)) + ' km/s' + ', ' + r'$k_S = %.1f$' % k_sausage + '\n' + r'$k=%.1f$' % k_mock + ', ' + r'$f=%.1f$' % frac_sausage

        else:
            extratext += '\n' + r'$v_{\rm{esc}}$ = ' + str(
                int(vesc_mock)) + ' km/s' + '\n' + r'$k_S = %.1f$' % k_sausage + ', ' + r'$k=%.1f$' % k_mock + ', ' + r'$f=%.1f$' % frac_sausage

    if include_aic:
        ax2.text(308, -55, extratext, fontsize=12)
    else:
        ax.text(vmin_list[-1] - 1.1 * spacing, v_max_plot * 0.98, extratext, fontsize=14, verticalalignment='top')

    full_filename = plots_dir + 'k_violin_different_methods' + filename + '_err_' + str(error_mag) + error_type + '.pdf'

    plt.savefig(full_filename)
    plt.close()


def make_resulting_bic_plot_separate_functions(
        vmin_list=[300, 325, 350, 375, 400],
        make_plot=False,
        k_mock=3.5,
        k_sausage=1.0,
        frac_sausage=0.6,
        fs_sausage_min=0.0,
        fs_sausage_max=1.0,
        error_range=0,
        sausage_generation=1,
        vesc_mock=500,
        error_mag=0.1,
        error_type='percent',
        aic=0,
        mock=1,
        rgcmin=7.0,
        rgcmax=9.0,
        zmin=0.0,
        zmax=15.0,
        vphicut=1,
        limited_prior=0,
        accreted=0,
        cutoff=0.95,
        functions_to_plot=[1],
        fire=0,
        lsr=0,
        chains_dir='../chains/',
        plots_dir='../plots/',
        dr2=1,
        edr3=0):
    """
    Returns a figure with 5 panels, that summarizes the results with a single and multiple function fit.
    :param: vmin_list: list of vmins to go through
    :param: make_plot: boolean, if true, saves the figure plot as well. That might not really work because of multiple figures open at the same time... #TODO Rethink
    :param: k_mock: the true mock value of k
    :param: k_sausage: the true mock value of k sausage
    :param: frac_sausage: the true sausage fraction
    :param: fs_sausage_min: minimum fraction prior
    :param: fs_sausage_max: maximum fractrion prior
    :param: error_range: if the errors are spread out across a range that's capped, instead of fixed.
    :param: sausage_generation: whether this is a single or 2 functions generated
    :param: vesc_mock: true escape velocity
    :param: error_mag: magnitude of the errors
    :param: error_type: type of errors
    :param: aic: boolean, if true, it returns the AIC not BIC
    :param: mock: boolean, if true it runs the mocks, if false, it runs the data
    :param: rgcmin = 7, minimum distance cut for Gaia runs
    :param: rgcmax = 9, maximum distance cut for Gaia runs
    :param: zmin = 0, minimum vertical cut for Gaia runs
    :param: zmax = 15, maximum vertical cut for Gaia runs
    :param: vphicut = 1, retrograde cut
    :param: two_vesc = 0, making 2 vesc for the fit. I don't really use this
    :param: limited_prior = 0, boolean, if true, limits the priors on k.
    :param: accreted = 0, boolean, if true, pulls up the accreted dataset
    :param: cutoff = 0.95, float, accreted score cut
    :param: functions_to_plot, a list of the functions to plot, from 1, 2, or 3 function fit.
    :param: fire: Boolean, if true this a fire run, not Gaia
    :param: lsr: int, 0,1,2 location of the Sun in the simulation
    :param: chains_dir = '../chains/' directory of chains
    :param: plots_dir = '../plots/' directory of plots
    :param: dr2 = 1, boolean, Gaia dr2
    :param: edr3 = 0, boolean, Gaia edr3
    """

    fontsize = 14

    mpl.rcParams.update({'font.size': fontsize, 'font.family': 'serif'})

    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['patch.linewidth'] = 1

    fig = plt.figure(figsize=(5, 4))

    filename = ''
    if mock:
        filename = '_inv_k_mock' + str(k_mock) + '_frac_' + str(frac_sausage)
    if error_range:
        filename += 'err_range'
    if accreted:
        filename += '_accreted_' + str(int(100 * cutoff))

    v_min_plot = 400
    v_max_plot = 600
    k_min_plot = 0
    k_max_plot = 3

    ax = fig.add_subplot(1, 1, 1)

    ax.minorticks_on()
    ax.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
    ax.tick_params('both', length=4, width=1, which='minor', direction='in', labelsize=fontsize)

    delta_bic_list = np.zeros_like(vmin_list)

    return_bic = 1
    return_aic = 0

    if aic:
        return_bic = 0
        return_aic = 1

    if 1 in functions_to_plot:
        bic_1 = np.zeros(len(vmin_list))
    if 2 in functions_to_plot:
        bic_2 = np.zeros(len(vmin_list))
    if 3 in functions_to_plot:
        bic_3 = np.zeros(len(vmin_list))

    for s, vmin in enumerate(vmin_list):

        print("vmin", vmin)

        two_vesc = False
        inverse_vesc_prior = True
        for f in functions_to_plot:
            print("f is", f)
            if f == 1:
                sausage = False
                three_functions = False
            elif f == 2:
                sausage = True
                three_functions = False
            elif f == 3:
                sausage = True
                three_functions = True

            # try:
            if mock:
                bic_a = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot,
                                          k_mock=k_mock, k_sausage=k_sausage, frac_sausage=frac_sausage,
                                          fs_sausage_min=fs_sausage_min, fs_sausage_max=fs_sausage_max,
                                          return_n_data=False, inverse_vesc_prior=inverse_vesc_prior,
                                          sausage_generation=sausage_generation, vesc_mock=vesc_mock,
                                          error_mag=error_mag, error_type=error_type, error_range=error_range,
                                          return_bic=return_bic, return_aic=return_aic, limited_prior=limited_prior)
            else:
                bic_a = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                     two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                     return_ndata=False, error_type=error_type, return_vesc_k=False,
                                     mean_values=False, inverse_vesc_prior=inverse_vesc_prior,
                                     return_bic=return_bic, return_aic=return_aic, accreted=accreted, cutoff=cutoff,
                                     three_functions=three_functions, chains_dir=chains_dir, plots_dir=plots_dir,
                                     fire=fire, lsr=lsr, edr3=edr3)
            # except:
            #     bic_a = 0
            #     print("This file does not exist", "f", f, "vmin", vmin)
            if f == 1:
                bic_1[s] = bic_a
            elif f == 2:
                bic_2[s] = bic_a
            elif f == 3:
                bic_3[s] = bic_a

    zero_array_y = np.zeros_like(vmin_list)

    ax.plot(vmin_list, zero_array_y, linestyle='--', color='gray')

    if (1 and 2) in functions_to_plot:
        ax.plot(vmin_list, bic_2 - bic_1, color='red', label=r'AIC$_2$ - AIC$_1$')
    if (2 and 3) in functions_to_plot:
        ax.plot(vmin_list, bic_3 - bic_2, color='darkgreen', label=r'AIC$_3$ - AIC$_2$')
    if (1 and 3) in functions_to_plot:
        ax.plot(vmin_list, bic_3 - bic_1, color='blue', label=r'AIC$_3$ - AIC$_1$')

    ax.set_xlabel(r'$v_{\rm{min}}$ [km/s]', fontsize=12)
    if aic:
        ax.set_ylabel(r'AIC', fontsize=12)
    else:
        ax.set_ylabel(r'BIC', fontsize=12)
    ax.set_xlim([vmin_list[0], vmin_list[-1]])

    ymin = -20
    ymax = 5
    ax.set_ylim([ymin, ymax])

    plt.tight_layout()

    if error_type == 'percent':
        error_text = r'Err$=$' + str(100 * error_mag) + r'$\%$'
    elif error_type == 'absolute':
        error_text = r'Err$=$' + str(error_mag) + ' km/s'
    elif error_type == 'no_errors' or error_type == 'no_error':
        error_text = 'No Errors'

    if mock:
        title_text = r'\textbf{Simulations}'
        filename += '_mock_'
    else:
        if dr2:
            title_text = r'\textbf{Gaia DR2}'
        elif edr3:
            title_text = r'\textbf{Gaia eDR3}'
        elif fire:
            title_text = r'\textbf{FIRE, LSR }' + str(lsr)
            filename += '_lsr_' + str(lsr)

        if vphicut:
            title_text += '\n' + r'\textbf{Retrograde Data}'
            filename += '_retro_data_'
        elif accreted:
            title_text += r'\textbf{, Accreted Stars}' + r', $S>$' + str(cutoff)
            filename += '_accreted_' + str(int(100 * cutoff))

    extratext = title_text + '\n' + error_text
    if mock:
        extratext += '\n' + r'$v_{\rm{esc}}$ = ' + str(
            int(vesc_mock)) + ' km/s' + '\n' + r'$k_S = %.1f$' % k_sausage + '\n' + r'$k=%.1f$' % k_mock + '\n' + r'$f=%.1f$' % frac_sausage

    ax.text(0.95, 0.3, extratext, fontsize=14, ha='right', va='bottom', transform=ax.transAxes)
    plt.legend(fontsize=12, loc='lower right', frameon=False)
    if aic:
        plt.savefig(plots_dir + 'aic_different_methods' + filename + '_err_' + str(error_mag) + error_type + '.pdf')
    else:
        plt.savefig(plots_dir + 'bic_different_methods' + filename + '_err_' + str(error_mag) + error_type + '.pdf')
    plt.close()


def make_resulting_bic_plot(
        vmin_list=[300, 325, 350, 375, 400],
        make_plot=False,
        k_mock=3.5,
        k_sausage=1.0,
        frac_sausage=0.6,
        fs_sausage_min=0.0,
        fs_sausage_max=1.0,
        error_range=0,
        sausage_generation=1,
        vesc_mock=500,
        error_mag=0.1,
        error_type='percent',
        aic=0,
        mock=1,
        rgcmin=7.0,
        rgcmax=9.0,
        zmin=0.0,
        zmax=15.0,
        vphicut=1,
        limited_prior=0,
        accreted=0,
        cutoff=0.95,
        no_plot=0,
        fire=0,
        lsr=0,
        chains_dir='../chains/',
        plots_dir='../plots/',
        data_dir='../data/',
        dr2=1,
        simulation='m12i',
        edr3=0):
    """
    Returns a figure that summarizes the results of the AIC with a single and multiple function fit.
    :param: vmin_list: list of vmins to go through
    :param: make_plot: boolean, if true, saves the data plot as well. That might not really work because of multiple figures open at the same time... #TODO Rethink
    :param: k_mock: the true mock value of k
    :param: k_sausage: the true mock value of k sausage
    :param: frac_sausage: the true sausage fraction
    :param: fs_sausage_min: minimum fraction prior
    :param: fs_sausage_max: maximum fractrion prior
    :param: error_range: if the errors are spread out across a range that's capped, instead of fixed.
    :param: sausage_generation: whether this is a single or 2 functions generated
    :param: vesc_mock: true escape velocity
    :param: error_mag: magnitude of the errors
    :param: error_type: type of errors
    :param: aic: boolean, if true, it returns the AIC not BIC
    :param: mock: boolean, if true it runs the mocks, if false, it runs the data
    :param: rgcmin = 7, minimum distance cut for Gaia runs
    :param: rgcmax = 9, maximum distance cut for Gaia runs
    :param: zmin = 0, minimum vertical cut for Gaia runs
    :param: zmax = 15, maximum vertical cut for Gaia runs
    :param: vphicut = 1, retrograde cut
    :param: two_vesc = 0, making 2 vesc for the fit. I don't really use this
    :param: limited_prior = 0, boolean, if true, limits the priors on k.
    :param: accreted = 0, boolean, if true, pulls up the accreted dataset
    :param: cutoff = 0.95, float, accreted score cut
    :param: no_plot = 0, boolean, if true, does not make the plot, just returns the values of Delta AIC so they can be
    combined in other plots
    :param: fire = 0, boolean, if true analyzes the FIRE simulation
    :param: lsr = 0, int for the location of the sun in FIRE
    :param: chains_dir = '../chains/' location of the chains
    :param: plots_dir = '../plots/' location of the plots
    :param: data_dir = '../data/' location of the data
    :param: dr2=1, Gaia DR2
    :param: edr3=0, Gaia edr3
    :param: simulation='m12i' or 'm12f'
    """

    filename = ''
    if mock:
        filename = '_inv_k_mock' + str(k_mock) + '_frac_' + str(frac_sausage)
    else:
        if vphicut:
            filename += '_retro_data_'
        if accreted:
            filename += '_accreted_' + str(int(100 * cutoff))

    if error_range:
        filename += 'err_range'

    if mock:
        title_text = r'\textbf{Simulations}'
    else:
        if dr2:
            title_text = r'\textbf{Gaia DR2}'
        elif edr3:
            title_text = r'\textbf{Gaia eDR3}'
        elif fire:
            title_text = r'\textbf{FIRE, LSR }' + str(lsr)
            filename += '_lsr_' + str(lsr)

    delta_bic_list = np.zeros_like(vmin_list)

    return_bic = 1
    return_aic = 0

    if aic:
        return_bic = 0
        return_aic = 1

    if no_plot:
        if os.path.isfile(data_dir + 'aic_list_' + filename + '.npy'):
            vmin_list, aic_list = np.load(data_dir + 'aic_list_' + filename + '.npy')
            return vmin_list, aic_list

    for s, vmin in enumerate(vmin_list):

        print("vmin", vmin)
        sausage = False
        two_vesc = False
        inverse_vesc_prior = True
        if mock:
            bic_a = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot, k_mock=k_mock,
                                      k_sausage=k_sausage, frac_sausage=frac_sausage, fs_sausage_min=fs_sausage_min,
                                      fs_sausage_max=fs_sausage_max, return_n_data=False,
                                      inverse_vesc_prior=inverse_vesc_prior, sausage_generation=sausage_generation,
                                      vesc_mock=vesc_mock, error_mag=error_mag, error_type=error_type,
                                      error_range=error_range, return_bic=return_bic, return_aic=return_aic)
        else:
            bic_a = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                 two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                 return_ndata=False, error_type=error_type, return_vesc_k=False, mean_values=False,
                                 inverse_vesc_prior=inverse_vesc_prior, return_bic=return_bic, return_aic=return_aic,
                                 accreted=accreted, cutoff=cutoff, fire=fire, lsr=lsr, chains_dir=chains_dir,
                                 simulation=simulation)

        sausage = True
        two_vesc = False
        inverse_vesc_prior = True

        if mock:
            bic_b = extract_vesc_mock(vmin, sausage=sausage, two_vesc=two_vesc, make_plot=make_plot, k_mock=k_mock,
                                      k_sausage=k_sausage, frac_sausage=frac_sausage, fs_sausage_min=fs_sausage_min,
                                      fs_sausage_max=fs_sausage_max, mean_values=False, return_vesc_k=False,
                                      inverse_vesc_prior=inverse_vesc_prior, sausage_generation=sausage_generation,
                                      vesc_mock=vesc_mock, error_mag=error_mag, error_type=error_type,
                                      error_range=error_range, return_bic=return_bic, return_aic=return_aic)
        else:
            bic_b = extract_vesc(rgcmin, rgcmax, zmin, zmax, error_mag, vmin, sausage=sausage, vphicut=vphicut,
                                 two_vesc=two_vesc, limited_prior=limited_prior, make_plot=make_plot,
                                 return_ndata=False, error_type=error_type, return_vesc_k=False, mean_values=False,
                                 inverse_vesc_prior=inverse_vesc_prior, return_bic=return_bic, return_aic=return_aic,
                                 accreted=accreted, cutoff=cutoff, fire=fire, lsr=lsr, chains_dir=chains_dir,
                                 simulation=simulation)

        delta_bic_list[s] = bic_b - bic_a

    if no_plot:
        np.save(data_dir + 'aic_list_' + filename + '.npy', [vmin_list, delta_bic_list])

        return vmin_list, delta_bic_list

    v_min_plot = 400
    v_max_plot = 600
    k_min_plot = 0
    k_max_plot = 3

    fontsize = 14

    mpl.rcParams.update({'font.size': fontsize, 'font.family': 'serif'})

    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['patch.linewidth'] = 1

    fig = plt.figure(figsize=(5, 4))

    ax = fig.add_subplot(1, 1, 1)

    ax.minorticks_on()
    ax.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
    ax.tick_params('both', length=4, width=1, which='minor', direction='in', labelsize=fontsize)

    ax.axvline(x=vesc_mock, linestyle='--', color='gray')

    zero_array_y = np.zeros_like(vmin_list)

    ax.plot(vmin_list, zero_array_y, linestyle='--', color='gray')
    ax.plot(vmin_list, delta_bic_list, color='r')

    ax.set_xlabel(r'$v_{\rm{min}}$ [km/s]', fontsize=12)
    if aic:
        ax.set_ylabel(r'$\Delta$ AIC', fontsize=12)
    else:
        ax.set_ylabel(r'$\Delta$ BIC', fontsize=12)
    ax.set_xlim([vmin_list[0], vmin_list[-1]])
    # ax.set_ylim([k_min_plot,k_max_plot])

    plt.tight_layout()

    if error_type == 'percent':
        error_text = r'Err$=$' + str(100 * error_mag) + r'$\%$'
    elif error_type == 'absolute':
        error_text = r'Err$=$' + str(error_mag) + ' km/s'
    elif error_type == 'no_errors' or error_type == 'no_error':
        error_text = 'No Errors'

    extratext = title_text + '\n' + error_text
    if mock:
        extratext += '\n' + r'$v_{\rm{esc}}$ = ' + str(
            int(vesc_mock)) + ' km/s' + '\n' + r'$k_S = %.1f$' % k_sausage + '\n' + r'$k=%.1f$' % k_mock + '\n' + r'$f=%.1f$' % frac_sausage

    ax.text(0.4, 0.1, extratext, fontsize=14, ha='left', va='bottom', transform=ax.transAxes)

    if aic:
        plt.savefig(
            plots_dir + 'delta_aic_different_methods' + filename + '_err_' + str(error_mag) + error_type + '.pdf')
    else:
        plt.savefig(
            plots_dir + 'delta_bic_different_methods' + filename + '_err_' + str(error_mag) + error_type + '.pdf')
    plt.close()


def true_fraction(
        vmin_list=[300, 325, 350, 375, 400],
        make_plot=False,
        k_mock=3.5,
        k_sausage=1.0,
        frac_sausage=0.6,
        f_out_true=0.01,
        sigma_mock=1000.0,
        fs_sausage_min=0.0,
        fs_sausage_max=1.0,
        error_range=0,
        sausage_generation=1,
        vesc_mock=500,
        error_mag=0.1,
        error_type='percent',
        f_three=0.0):
    """
    Returns the true fraction of the distribution as a function fo the vmin_list given the slopes,
    the total fraction above vmin = 300km/s, and the escape velocity I guess
    :param: vmin_list: list of vmins to go through
    :param: make_plot: boolean, if true, saves the figure plot as well.
    :param: k_mock: the true mock value of k
    :param: k_sausage: the true mock value of k sausage
    :param: frac_sausage: the true sausage fraction
    :param: f_out_true: float, fraction of the outliers
    :param: fs_sausage_min: minimum fraction prior
    :param: error_range: if the errors are spread out across a range that's capped, instead of fixed.
    :param: sausage_generation: whether this is a single or 2 functions generated
    :param: vesc_mock: true escape velocity
    :param: error_mag: magnitude of the errors
    :param: error_type: type of errors
    :param: f_three: boolean, fraction of the third function, which would be the disk, so that needs to be taken out.
    """

    jobs_dir = '../jobs/'
    chains_dir = '../chains/'
    data_dir = '../data/'
    plots_dir = '../plots/'

    vmin_step = 5
    vmax = 1000
    speed_array = np.arange(vmin_list[0], vmax, vmin_step)

    if error_type == 'percent':
        speed_error_array = error_mag * speed_array
    elif error_type == 'absolute':
        speed_error_array = error_mag
    elif error_type == 'no_errors':
        speed_error_array = np.zeros_like(speed_array)

    assert (sausage_generation), "You cannot get a fraction distribution without generating the sausage component"

    if error_type == 'no_errors':
        data_true = (1 - f_out_true) * (1 - frac_sausage - f_three) * np.exp(np.array(
            functions_MCMC_cy.function_no_err_vectorized(vesc=vesc_mock, k=k_mock, v=speed_array,
                                                         sigma=speed_error_array, vmin=vmin_list[0], relative_error=0,
                                                         kin_energy=0)))

        sausage_true = (1 - f_out_true) * (frac_sausage) * np.exp(np.array(
            functions_MCMC_cy.function_no_err_vectorized(vesc=vesc_mock, k=k_sausage, v=speed_array,
                                                         sigma=speed_error_array, vmin=vmin_list[0], relative_error=0,
                                                         kin_energy=0)))
    else:
        data_true = (1 - f_out_true) * (1 - frac_sausage - f_three) * np.exp(np.array(
            functions_MCMC_cy.function_vectorized(vesc=vesc_mock, k=k_mock, v=speed_array, sigma=speed_error_array,
                                                  vmin=vmin_list[0], relative_error=0, kin_energy=0)))

        sausage_true = (1 - f_out_true) * (frac_sausage) * np.exp(np.array(
            functions_MCMC_cy.function_vectorized(vesc=vesc_mock, k=k_sausage, v=speed_array, sigma=speed_error_array,
                                                  vmin=vmin_list[0], relative_error=0, kin_energy=0)))

    outlier_true = f_out_true * np.exp(np.array(
        functions_MCMC_cy.outliers_normalized_vectorized(sigma_mock, speed_array, speed_error_array, vmin_list[0])))

    if make_plot:
        fontsize = 14

        mpl.rcParams.update({'font.size': fontsize, 'font.family': 'serif'})

        mpl.rcParams['axes.linewidth'] = 1
        mpl.rcParams['lines.linewidth'] = 2
        mpl.rcParams['patch.linewidth'] = 1

        fig = plt.figure(figsize=(5, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        ax2.minorticks_on()
        ax2.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
        ax2.tick_params('both', length=4, width=1, which='minor', direction='in', labelsize=fontsize)

        ax1.minorticks_on()
        ax1.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
        ax1.tick_params('both', length=4, width=1, which='minor', direction='in', labelsize=fontsize)

        ax1.plot(speed_array, data_true, ls='--', lw=2, color='red', label='Halo')
        ax1.plot(speed_array, outlier_true, ls='--', lw=2, color='cyan', label='Outliers')
        ax1.plot(speed_array, sausage_true, ls='--', lw=2, color='darkgreen', label='Sausage')

        ax1.plot(speed_array, data_true + outlier_true + sausage_true, ls='--', lw=2, color='blue', label='Total')

        ax2.plot(speed_array, sausage_true / (data_true + sausage_true), color='purple')

        ax2.plot(speed_array, np.ones_like(speed_array), zorder=2, color='k', lw=1.2)
        ax2.set_xlabel(r'$|\vec{v}|$' + ' [km/s]', fontsize=12)
        ax2.set_ylabel(r'$g_S$ [True]', fontsize=12)
        ax2.set_ylim([0, 1])
        ax2.grid(which='both', axis='y')

        ymin = 1e-6
        ymax = 1e-1
        ax1.set_yscale('log')
        ax1.set_xlim([vmin_list[0], vmin_list[-1]])
        ax1.set_ylim([ymin, ymax])
        ax1.set_ylabel(r'$g(|\vec{v}|)$' + ' [km/s]' + r'$^{-1}$', fontsize=14)
        ax1.legend(fontsize=14, frameon=False)

        plt.tight_layout()
        # plt.title(r'\textbf{Gaia DR2 Fit}', fontsize = 16)
        plt.savefig(plots_dir + 'fraction_plot_k_' + str(k_mock) + '_ks_' + str(k_sausage) + '_fs_' + str(
            frac_sausage) + '_err_' + str(error_mag) + error_type + '.pdf', bbox_inches='tight')
        plt.close()

    import scipy

    sausage_integral = np.array(
        [scipy.integrate.simps(sausage_true[s:], speed_array[s:]) for s in range(len(speed_array))])
    all_data_integral = np.array(
        [scipy.integrate.simps(data_true[s:] + sausage_true[s:], speed_array[s:]) for s in range(len(speed_array))])

    fraction_list = sausage_integral / all_data_integral

    return speed_array, fraction_list


def fraction_true_found(
        vmin_list=[300, 325, 350, 375, 400],
        make_plot=False,
        k_mock=3.5,
        k_sausage=1.0,
        frac_sausage=0.6,
        f_out_true=0.01,
        sigma_mock=1000.0,
        fs_sausage_min=0.0,
        fs_sausage_max=1.0,
        error_range=0,
        sausage_generation=1,
        vesc_mock=500,
        error_mag=0.1,
        error_type='percent',
        violinplot=0):
    """
    Returns a plot where the true fraction (for that specific vmin) compares to the one found
    :param: vmin_list: list of vmins to go through
    :param: make_plot: boolean, if true, saves the figure plot as well.
    :param: k_mock: the true mock value of k
    :param: k_sausage: the true mock value of k sausage
    :param: frac_sausage: the true sausage fraction
    :param: f_out_true: float, fraction of the outliers
    :param: fs_sausage_min: minimum fraction prior
    :param: fs_sausage_max: maximum fraction prior
    :param: error_range: if the errors are spread out across a range that's capped, instead of fixed.
    :param: sausage_generation: whether this is a single or 2 functions generated
    :param: vesc_mock: true escape velocity
    :param: error_mag: magnitude of the errors
    :param: error_type: type of errors
    :param: violinplot: boolean, if true, it plots the whole thing as a violin plot instead of error plot
    """

    speed_array, fraction_list = true_fraction(vmin_list=vmin_list, make_plot=make_plot, k_mock=k_mock,
                                               k_sausage=k_sausage, frac_sausage=frac_sausage,
                                               fs_sausage_min=fs_sausage_min, fs_sausage_max=fs_sausage_max,
                                               error_range=error_range, sausage_generation=sausage_generation,
                                               vesc_mock=vesc_mock, error_mag=error_mag, error_type=error_type)

    from scipy import interpolate

    fraction_true_interpolated = interpolate.interp1d(speed_array, fraction_list)

    fraction_true = [fraction_true_interpolated(vmin) for vmin in vmin_list]

    fontsize = 16

    jobs_dir = '../jobs/'
    chains_dir = '../chains/'
    data_dir = '../data/'
    plots_dir = '../plots/'

    mpl.rcParams.update({'font.size': fontsize, 'font.family': 'serif'})

    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['patch.linewidth'] = 1

    if violinplot:

        fig = plt.figure(figsize=(5, 6))

        ax1 = fig.add_subplot(1, 1, 1)

        ax1.minorticks_on()
        ax1.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
        ax1.tick_params('x', length=4, width=1, which='minor', direction='in', labelsize=fontsize)

        ax1.yaxis.set_tick_params(which='minor', left=False, right=False)

        cmap = plt.cm.get_cmap('YlOrRd')
        color_indices = np.linspace(0, 1, len(vmin_list) + 1)
        color_list = cmap(color_indices)

        for v, vmin in enumerate(vmin_list):
            kS, fS_found = extract_vesc_mock(vmin, sausage=1, two_vesc=0, inverse_vesc_prior=1, limited_prior=0,
                                             kpriormin=0, kpriormax=15, make_plot=0, return_k=0, return_k_fs=1,
                                             k_mock=k_mock, k_sausage=k_sausage, frac_sausage=frac_sausage,
                                             fs_sausage_min=fs_sausage_min, fs_sausage_max=fs_sausage_max,
                                             kpriormin_input=0, kpriormax_input=15, return_vesc_k=0, mean_values=0,
                                             return_n_data=0, return_speed=0, kin_energy=0, sausage_generation=1,
                                             vesc_mock=vesc_mock, error_mag=error_mag, error_type=error_type,
                                             error_range=error_range, return_bic=0, return_aic=0)

            vstep = 0.02
            binning = np.arange(0, 1 + vstep, vstep)

            fS_histogram, edges = np.histogram(fS_found, bins=binning, density=True)

            rescaling_factor = np.max(fS_histogram)

            bins_to_plot = (edges[:-1] + edges[1:]) / 2.

            ax1.fill_between(bins_to_plot, v, (fS_histogram / rescaling_factor) + v, color=color_list[v + 1], alpha=0.4)

            ax1.axvline(fraction_true[v], ymin=v / (len(vmin_list)), ymax=(v + 1) / (len(vmin_list)), linestyle='--',
                        linewidth=2, color=color_list[v + 1])

        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, len(vmin_list)])
        ax1.set_xlabel(r'$g_S$ Found', fontsize=14)
        ax1.set_ylabel(r'$v_{\rm{min}}$ [km/s]', fontsize=14)
        ax1.set_yticks(ticks=np.arange(len(vmin_list)))
        ax1.set_yticklabels(labels=vmin_list)

        plt.tight_layout()
        plt.savefig(
            plots_dir + 'fraction_comparison_plot_violin_k_' + str(k_mock) + '_ks_' + str(k_sausage) + '_fs_' + str(
                frac_sausage) + '_err_' + str(error_mag) + error_type + 'err_range' + str(error_range) + '.pdf',
            bbox_inches='tight')
        plt.close()


    else:
        fS_found = np.zeros((len(vmin_list), 3))
        for v, vmin in enumerate(vmin_list):
            kS, fS_found[v] = extract_vesc_mock(vmin, sausage=1, two_vesc=0, inverse_vesc_prior=1, limited_prior=0,
                                                kpriormin=0, kpriormax=15, make_plot=0, return_k=0, return_k_fs=1,
                                                k_mock=k_mock, k_sausage=k_sausage, frac_sausage=frac_sausage,
                                                fs_sausage_min=fs_sausage_min, fs_sausage_max=fs_sausage_max,
                                                kpriormin_input=0, kpriormax_input=15, return_vesc_k=0, mean_values=1,
                                                return_n_data=0, return_speed=0, kin_energy=0, sausage_generation=1,
                                                vesc_mock=vesc_mock, error_mag=error_mag, error_type=error_type,
                                                error_range=error_range, return_bic=0, return_aic=0)

        fig = plt.figure(figsize=(5, 4))

        ax1 = fig.add_subplot(1, 1, 1)

        ax1.minorticks_on()
        ax1.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
        ax1.tick_params('both', length=4, width=1, which='minor', direction='in', labelsize=fontsize)

        diagonal = np.arange(0, 1.1, 0.1)

        ax1.plot(diagonal, diagonal, ls='--', color='gray')

        fS_mean, fS_up, fS_down = fS_found.T
        print("fraction_true", fraction_true)
        print("fS mean", fS_mean)
        print("yerr", np.array([fS_down, fS_up]))

        cmap = plt.cm.get_cmap('YlOrRd')
        color_indices = np.linspace(0, 1, len(vmin_list) + 1)
        color_list = cmap(color_indices)

        for s in range(len(vmin_list)):
            ax1.errorbar(fraction_true[s], fS_mean[s], yerr=np.array([[fS_down[s]], [fS_up[s]]]), ls='none', capsize=3,
                         marker='o', color=color_list[s + 1])

        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel(r'$g_S$ True', fontsize=14)
        ax1.set_ylabel(r'$g_S$ Found', fontsize=14)

        plt.tight_layout()
        # plt.title(r'\textbf{Gaia DR2 Fit}', fontsize = 16)
        plt.savefig(plots_dir + 'fraction_comparison_plot_k_' + str(k_mock) + '_ks_' + str(k_sausage) + '_fs_' + str(
            frac_sausage) + '_err_' + str(error_mag) + error_type + 'err_range' + str(error_range) + '.pdf',
                    bbox_inches='tight')
        plt.close()
