#[Run Options]
run = 1
save_progress = 1
multiprocessing = 1
overwrite = 0
generate_error = 1       # For simulation runs

#[Plotting]
#plot = 1
test_plotting = 1
n_steps_to_plot = 50
levels = [0.68,0.95]

#[MCMC Parameters]
nwalkers = 50          
nsamples_burnin = 400   # discarded samples per walker
nsamples = 1000         # post-burnin samples per walker
vesc_guess = 500
k_guess = 2.2
frac_guess = -11.5   # = np.log(1e-5)
sigma_guess = 6.7    # = np.log(800)
k_sausage_guess = 0.7
frac_sausage_guess = 0.8
k_three_guess = 1.5
frac_three_guess = 0.1

#[Cuts]
rgcmin = 7.0
rgcmax = 9.0
z_min = 0.0
z_max = 15.0
verrcut = 0.05 # 5% error
error_type = "percent"  # Types of errors: 'percent', 'no_errors', 'absolute'
vmin = 1
vphicut = 0
cutoff = 0.75
n_test = 2000           # Set to -1 for "include all stars after the other cuts"
random_samples = 0  # Set to 1 to get n_test random stars, when 0 use the 
                        # default ordering of pre_processing_FIRE.py, which is by distance 
                        # to axis between galactic center and lsr position
high_z = 0          # Instead of taking the n_test closest stars to the axis, take the n_test furthest
                        # pointless if used with random_samples = 1
#preprocessing_cut_type = '_cylinder_cut'   # from pre_processing_FIRE.py, needs underscore in front
preprocessing_cut_type = ""  # the default heliocentric cut

#[Priors]
vescprior = 1
sigmapriormin = 6
kpriormin = 0.1
kpriormax = 20
fs_sausage_min = 0.0
fs_sausage_max = 1.0
inverse_vesc_prior = 0
relative_error = 0     # What is this?
limited_priors = 0

#[Dataset]
sixdpropermotion = 0
propermotion = 0
dr2 = 0
ananke = 0
mock = 0
edr3 = 0
fire = 1
lsr = 0
simulation="m12f"

#[Analysis]
sausage = 0
three_functions = 1
two_vesc = 0
accreted = 0    # Where should this go??
outlier = 1
kin_energy = 0

#[Mocks] # Options when building or running MCMC on mocks
vmin_mock = 350  # most data is generated above this

error_range = 0

vesc_mock = 500
sigma_mock = 1000
k_mock = 3.5
frac_sausage_mock = 0.6
frac_outlier = 0.01
k_sausage_mock = 1.0

substructure = 0
sub_mean = 370  # 500
sub_dispersion = 20  # 50
frac_substructure = 0.2  # 0.01