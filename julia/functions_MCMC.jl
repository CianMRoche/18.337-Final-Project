using SpecialFunctions
#using NumericalIntegration
using QuadGK

# ------------------------- Useful functions --------------------------
function constant_factor_helper_function(x::Float64,sigma::Float64, vesc::Float64, k::Float64, vmin::Float64)
    return (vesc - x)^k * (erf( (x - vmin)/(sqrt(2.0) * sigma) ) + 1.0 )
end


function constant_factor(sigma::Float64, vesc::Float64, k::Float64, vmin::Float64 = 300.0)
    """
    Constant that multiplies the whole function. Needs to be properly normalized to 
        1 in the observed data region [vmin, infinity]  or just large upper vmax
    :theta: values to fit for, [vesc, k] # added for code consistency
    :sigma: dispersion for that particular value of the error
    :vmin: fixed for now
    :relative_error: boolean, whether the errors are relative
    """

    #n_samples = 1001
    min_integral = 0.0
    
    #x_integral = Vector(LinRange(min_integral, vesc , n_samples))
    #y_integral = (abs.(vesc .- x_integral)).^k .* (erf.( (x_integral .- vmin)./(sqrt(2.0) * sigma) ) .+ 1.0 )  
    #integral = 0.5 * integrate(x_integral, y_integral, SimpsonEven()) #if using, ditch the 0.5 in return statement and [1]

    integral = quadgk(x -> constant_factor_helper_function(x,sigma, vesc, k, vmin), min_integral, vesc)

    return - log(0.5*integral[1])
end


function fitting_function_helper_function(x::Float64, v::Float64, vesc::Float64, k::Float64, sigma::Float64)
    return exp(- (v - x)^2 / (2.0*sigma*sigma)) * (vesc - x)^k / (sqrt(2.0*pi) * sigma)
end


function fitting_function(v::Float64,
                        vesc::Float64, k::Float64,
                        sigma::Float64, vmin::Float64 = 300.0)
    """
    Evaluating the function with the requirement that v>vesc -> f = 0
    :v: velocity norm for OBSERVED velocity
    :theta: values to fit for, [vesc, k, norm] # added for code consistency
    :sigma: dispersion for that particular value of the error
    :vmin: minimum velocity for data region
    :relative_error: boolean, whether the errors are relative
    :absolute_error_cut: boolean, if true, then the cut that turns the error integral small is based on a cut on absolute errors
    """

    logC = constant_factor(sigma, vesc, k, vmin)

    #n_samples = 1001 #257
    min_integral = 0.0

    if v < vmin # or (v - 5*sigma > vesc): #-- if v is more than 5 sigma away from vesc, 
        return -Inf
    end

    #= x_integral = Vector(LinRange(min_integral, vesc , n_samples))

    inside_exp = - (v .- x_integral).^2 ./ (2.0*sigma*sigma)
    exp_term = exp.(inside_exp)

    y_integral = exp_term .* (abs.(vesc  .- x_integral)).^k ./ (sqrt(2.0*pi) * sigma)   

    integral = integrate(x_integral, y_integral, SimpsonEven()) =#

    integral = quadgk(x -> fitting_function_helper_function(x, v, vesc, k, sigma), min_integral, vesc)

    return logC + log(integral[1]) 
end


function outliers(v,a,sigma)
    """
    Evaluating the outlier model where it is a Gaussian with a certain dispersion. 
    TODO: think about including the errors on the measurements here
    :v: velocity norm
    :a: normalization
    :sigma: dispersion of the outlier model. We will vary it later  
    """
    return  -(v.*v) ./(2.0.*sigma.*sigma) .- a 
end


# ------------------------ MCMC functions --------------------------
function lnlike(theta::Vector{Float64}, 
    v::Vector{Float64}, verr::Vector{Float64},
    vmin::Float64 = 300.0, outlier::Bool = true, 
    inverse_vesc_prior::Bool = false, sausage::Bool = false)
    """
    Log likelihood of the function evaluated, later we will add outlier model
    :theta: values to fit for, [vesc, k, norm] # added for code consistency
    :v: velocity norm [list]
    :verr: dispersion for that particular value of the error [list]
    :vmin: fixed to 300 km/s for now, but can be updated
    :outlier: boolean to check for the outlier model
    :inverse_vesc_prior: boolean, if true, vesc has a 1/ve prior instead of flat
    :relative_error: boolean, adding a relative error 
    :kin_energy: boolean, if true, the fit is to the square function
    :sausage: boolean, if true, it will fit two components to the regular function
    :two_vesc: boolean, if true, we have two separate v_escapes.
    :three_functions: boolean, if true, we have 3 fitting functions
    """
    if sausage
        vesc, k, frac, log_sigma, k_sausage, frac_sausage = theta

        if inverse_vesc_prior  
            one_over_vesc, k, frac, log_sigma, k_sausage, frac_sausage = theta   
            #this is now using the prior of 0611761, that's with the +vmin, ignoring that for now
            vesc = 1.0/one_over_vesc #+vmin
        else
            vesc, k, frac, log_sigma, k_sausage, frac_sausage = theta
        end

        sigma = sqrt.(verr.^2 .+ exp(log_sigma)^2) # THIS MIGHT BE A BAD PORT FROM PYTHON
        #println("Sigma (lnlike):",sigma)
        a = sqrt(pi/2.0) .* sigma .* erfc.(vmin./ (sqrt(2.0).*sigma))

        
        model =  log(1.0 - frac_sausage) .+  [fitting_function(v_element, vesc, k, verr_element, vmin) for (v_element,verr_element) in zip(v,verr)]
        sausage_likelihood =  log(frac_sausage) .+  [fitting_function(v_element, vesc, k_sausage, verr_element, vmin) for (v_element,verr_element) in zip(v,verr)] 
        
        bound_likelihood =  log(1.0 - exp(frac)) .+ log.( exp.(model) .+ exp.(sausage_likelihood)) 

        outlier_likelihood = frac .+ outliers(v, log.(a), sigma)

        return sum( log.(exp.(bound_likelihood) .+ exp.(outlier_likelihood) ) )

    else
        if inverse_vesc_prior     
            one_over_vesc, k, frac, log_sigma = theta
            #this is now using the prior of 0611761, that's with the +vmin, ignoring that for now
            vesc = 1.0/one_over_vesc #+vmin
        else
            vesc, k, frac, log_sigma = theta # sigma is actually in log now
            #println(vesc, k, frac, log_sigma)
        end
        
        sigma = sqrt.(verr.^2 .+ exp(log_sigma)^2) # THIS MIGHT BE A BAD PORT FROM PYTHON
        a = sqrt(pi/2.0) .* sigma .* erfc.(vmin./ (sqrt(2.0).*sigma))

        model = log(1.0 - exp(frac)) .+ [fitting_function(v_element, vesc, k, verr_element, vmin) for (v_element,verr_element) in zip(v,verr)] 
        outlier_likelihood = frac .+ outliers(v, log.(a), sigma)
        
        return sum( log.(exp.(model) .+ exp.(outlier_likelihood) ) )
        
    end
end

function lnprior(theta::Vector{Float64}, 
        vmin::Float64 = 300.0, outlier::Bool = true, 
        kpriormin::Float64 = 1.0, kpriormax::Float64 = 10.0, 
        sigmapriormin::Float64 = log(600.0), 
        inverse_vesc_prior::Bool = false, 
        sausage::Bool = false, frac_sausage_min::Float64 = 0.0, frac_sausage_max::Float64 = 1.0)
    """
    Adding the priors of the different values
    :theta: [vesc, k] to be fit for
    :vmin: minimum integral
    :outlier: boolean for the outlier
    """ 

    correction_vmin = 1.001


    if sausage
        vesc, k, frac, sigma_out, k_sausage, frac_sausage = theta

        if inverse_vesc_prior
            vesc = 1.0/vesc #+vmin
        end

        if (vmin*correction_vmin < vesc < 1000 
            && kpriormin < k < kpriormax 
            && log(1e-6) < frac < log(1.0) 
            && sigmapriormin < sigma_out < log(3000.0) 
            && kpriormin < k_sausage < k 
            && frac_sausage_min < frac_sausage < frac_sausage_max)
            return 0.0
        end

        return -Inf

    else
        if inverse_vesc_prior
            one_over_vesc, k, frac, sigma_out = theta
            vesc = 1.0/one_over_vesc #+vmin
        else
            vesc, k, frac, sigma_out = theta
        end

        if (vmin*correction_vmin < vesc < 1000.0 
            && kpriormin < k < kpriormax 
            && log(1e-6) < frac < log(1.0) 
            && sigmapriormin < sigma_out < log(3000.0) )
            return 0.0
        end

        return -Inf
    end
end

function lnprob(theta::Vector{Float64}, 
                v::Vector{Float64}, verr::Vector{Float64}, 
                vmin::Float64 = 300.0, outlier::Bool = true, 
                kpriormin::Float64 = 1.0, kpriormax::Float64 = 10.0, 
                sigmapriormin::Float64 = log(600.0), 
                inverse_vesc_prior::Bool = false, sausage::Bool = false, 
                frac_sausage_min::Float64 = 0.0, frac_sausage_max::Float64 = 1.0)
    """
    Combining the probablity with the log likelihood
    :v: velocity of stars
    :verr: errors on the stars 
    :vmin: fixed to 300 km/s for now, but can be updated
    :outlier: boolean for the outlier
    :kin_energy: boolean for the kinetic energy fit
    :sausage: boolean for the sausage
    """
    lp = lnprior(theta, vmin, outlier, kpriormin, kpriormax, sigmapriormin, 
    inverse_vesc_prior, sausage, frac_sausage_min,  frac_sausage_max)

    if isinf(lp)
        return -Inf
    end

    lk = lnlike(theta, v, verr, vmin, outlier, inverse_vesc_prior, sausage)

    return lp + lk 
end
