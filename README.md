# 6.338project
Final Project for MIT course 6.338 (18.337).

# Practical Bayesian Sampling in Python and Julia

In this report, Markov chain Monte Carlo (MCMC) algorithms are implemented in both Python and
Julia and benchmarked in both serial and parallel via a parameter fitting problem. Both the user
experience of implementing and using these algorithms and the raw computational performance
are considered in order to form a recommendation to young scientists interested in MCMC, in
particular those with existing experience in Python. The conclusion is that although prototyping
and development are faster and more transparent in python, the vast performance increases obtained
in Julia even before optimization make it the better choice, in particular for large datasets or high-
dimensional parameter spaces. It is found that MCMC algorithms run in Julia on a single core of a
consumer laptop outperform almost identical implementations (and the popular package emcee) in
python run on hundreds of cores on a supercomputing cluster.

# List of files
### Base directory
 - final_project_Cian_Roche.pdf : A PDF summarizing the results and outlining the methodology and test problem. A good place to start.
 - language_comparison.ipynb : Primary comparison of the performance of both algorithms in both languages (Python notebook)

### Python results
Can be found in the ``Python" folder.

- Affine_python.ipynb : A Python notebook to run the affine-invariant MCMC algorithm on a test problem
- MH_python.ipynb : A Python notebook to run the Metropolis-Hastings MCMC algorithm on a test problem
- affine_python_as_script.py : same as Affine_python.ipynb but a script for submission to computing clusters
- run_MCMC.batch : slurm submission script used with MIT engaging cluster
- cores_comparison.ipynb : Plotting notebook for the parallel performance of affine_python_as_script.py

### Julia results
Can be found in the ``Julia" folder.

- AffineInvariantMCMC.jl : A collection of functions implementing the affine invariant MCMC algoruithm (adapted from https://git.physics.byu.edu/Modeling/MCJulia.jl/-/tree/master/)
- Affine_julia.ipynb : A Julia notebook to run the affine-invariant MCMC algorithm on a test problem in serial
- Affine_julia_distributed.ipynb : A Julia notebook to run the affine-invariant MCMC algorithm on a test problem on many distributed processes
- Affine_julia_threaded.ipynb : A Julia notebook to run the affine-invariant MCMC algorithm on a test problem on a Julia kernel with access to more than 1 thread
- MH_python.ipynb : A Julia notebook to run the Metropolis-Hastings MCMC algorithm on a test problem
- threads_comparison.ipynb : Plotting notebook (in Python) for the multithreaded performance of Affine_julia_threaded.ipynb


# Principal Results

### Python vs Julia, performance scaling with dataset size
![Python vs Julia, performance scaling with dataset size](https://github.com/CianMRoche/6.338project/blob/e095e10a25e3ddb89b333f3c62a957977b2f46e5/plots/scaling.png "comparison")

Above is the performance of Metropolis-Hastings (MH) and affine-invariant (AI) MCMC imple-
mentations in both Python and Julia, and their scaling with size of dataset used in fitting.

### Python parallel performance scaling
![Python parallel performance scaling](https://github.com/CianMRoche/6.338project/blob/e095e10a25e3ddb89b333f3c62a957977b2f46e5/plots/scaling_cores.png "parallel python")

Scaling of affine-invariant MCMC
implementation in Python with the num-
ber of CPU cores used to run the algo-
rithm. Tests performed with 1000 data
points, 2,000 iterations per walker and vari-
able number of walkers.

### Julia multithreaded performance scaling
![Julia multithreaded performance scaling](https://github.com/CianMRoche/6.338project/blob/e095e10a25e3ddb89b333f3c62a957977b2f46e5/plots/scaling_threads.png "Julia multithreading")

Julia: Performance scaling
of affine-invariant MCMC implementation
with number of threads available to Julia,
shown as iterations per second relative to
the serial implementation. Tests performed
with 5 walkers.
