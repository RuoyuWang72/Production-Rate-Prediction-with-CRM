pwd()
cd("./Production-Rate-Prediction-with-CRM")
include("../../src/utils.jl")

using Pkg; Pkg.activate(".")
using Revise
using Distributions, Plots, IJulia, DifferentialEquations, Lux, Optimization, MLUtils, Random, Zygote, NNlib, ForwardDiff, ReverseDiff
using SciMLSensitivity
using CSV, DataFrames
using BenchmarkTools, LinearAlgebra
using LaTeXStrings
using OptimizationPolyalgorithms, OptimizationOptimisers, OptimizationOptimJL, OptimizationFlux, OptimizationNLopt
using ComponentArrays

gr()

# read the data
df_bhp = DataFrame(CSV.File("./data/CMG/prod_bhp.csv"))
df_prod = DataFrame(CSV.File("./data/CMG/prod_w_bhp.csv"))
df_inj = DataFrame(CSV.File("./data/CMG/inj.csv"))

# create vectors for injection and production rates and BHP
inj1 = df_inj[:, 2]
inj2 = df_inj[:, 3]
bhp1 = df_bhp[:, 2]
bhp2 = df_bhp[:, 3]
bhp3 = df_bhp[:, 4]
bhp4 = df_bhp[:, 5]
prod1 = df_prod[:, 2]
prod2 = df_prod[:, 3]
prod3 = df_prod[:, 4]
prod4 = df_prod[:, 5]

# create timesteps
timesteps = df_bhp[!, 1]

# make known injection rates and BHP functions of time
w1(t) = inj1[searchsortednearest(timesteps, t)]
w2(t) = inj2[searchsortednearest(timesteps, t)]
Pwf1(t) = bhp1[searchsortednearest(timesteps, t)]
Pwf2(t) = bhp2[searchsortednearest(timesteps, t)]
Pwf3(t) = bhp3[searchsortednearest(timesteps, t)]
Pwf4(t) = bhp4[searchsortednearest(timesteps, t)]
Pwfs = [Pwf1, Pwf2, Pwf3, Pwf4]
bhps = [bhp1, bhp2, bhp3, bhp4]
prods = [prod1, prod2, prod3, prod4]

# production rates on day 1
u0s = [prod1[1], prod2[1], prod3[1], prod4[1]]
tspan = (minimum(timesteps), maximum(timesteps))

# try producer 1
i = 1

# treating them as constants
function crmp_constant!(du, u, p, t)
    CtVp, J, F1, F2 = p
    du[1] = (w1(t)*F1 + w2(t)*F2 - CtVp*Pwfs[i](t) .- u[1]) ./ (CtVp / J)
end

function obj(p, x)
    prob_ = ODEProblem(crmp_constant!, x, (minimum(timesteps), maximum(timesteps)), p)
    sol_ = solve(prob_, saveat=timesteps)
    return sum(abs2, Array(sol_) .- prods[i])
end

# define the constraint
cons(res, p, x) = (res .= [p[3]+p[4]])

# initial guess
initial = [6.0, 3.0, 0.1, 0.2]

# define lower and upper bounds
lower = [0.0, 0.0, 0.0, 0.0]
upper = [Inf, Inf, 1.0, 1.0]
    
optprob = OptimizationFunction(obj, Optimization.AutoForwardDiff(), cons=cons)
prob = OptimizationProblem(optprob, initial, [u0s[i]], lb=lower, ub=upper, lcons=[0.0], ucons=[2.0])
opt_param = solve(prob, Optim.IPNewton(), iterations=10000, outer_iterations=10000)

sol_prob = ODEProblem(crmp_constant!, [u0s[i]], tspan, opt_param, saveat=timesteps)
sol_opt = solve(sol_prob)

# initialize the parameters
rng = Random.default_rng()
Random.seed!(rng, 20)
m_J = create_nn(tanh, 1, 10, 32, 1)
ps_J, st_J = Lux.setup(rng, m_J)
ps_CtVp = opt_param[1] * opt_param[2]
ps_gain1 = opt_param[3]
ps_gain2 = opt_param[4]
ps_J = ComponentArray(ps_J)
p = ComponentArray{eltype(ps_J)}()
p = ComponentArray(p; ps_CtVp, ps_gain1, ps_gain2, ps_J)


Pwf_val(t) = Pwfs[i](t)
Pwf_prime(t) = compute_derivative(timesteps, bhps[i])[searchsortednearest(timesteps, t)]

# define the dynamic CRM
function model!(du, u, p, t)
    CtVp, gain1, gain2, J_val, J_prime = p
    w_sum = gain1 * w1(t) + gain2 * w2(t)
    du[1] = (w_sum * J_val^2 - u[1] * J_val^2 + u[1] * J_prime * CtVp + J_val * Pwf_val(t) * J_prime * CtVp - J_prime * Pwf_val(t) * J_val * CtVp - J_val * Pwf_prime(t) * J_val * CtVp) / (J_val * CtVp)
end

# parameterize the dynamic CRM
function parameterized_model!(du, u, p, t)
    J_val = ReverseDiff.value(m_J([t], p.ps_J, st_J)[1][1])
    J_func(t) = m_J([t], p.ps_J, st_J)[1][1]
    arr = J_func.(x_train)
    J_prime_vec = ReverseDiff.value(compute_derivative(x_train, arr))
    J_prime = ReverseDiff.value(J_prime_vec[searchsortednearest(x_train, t)])
    p_ = [p.ps_CtVp, p.ps_gain1, p.ps_gain2, J_val, J_prime]
    model!(du, u, p_, t)
end

# train test split
(x_train, y_train), (x_val, y_val) = splitobs((timesteps, prods[i]); at=0.8, shuffle=false)
(x_val, y_val), (x_test, y_test) = splitobs((x_val, y_val); at=0.5, shuffle=false)

# define the ODE problem
prob_nn = ODEProblem(parameterized_model!, [u0s[i]], (minimum(x_train), maximum(x_train)), p, saveat=x_train)

# solve for ODE
predict_p(θ) = solve(prob_nn, Tsit5(), p=θ, abstol=1e-6, reltol=1e-6, sensealg=InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)), verbose=false, saveat=x_train)

# define the loss function
function loss_p(θ)
    pred = Array(predict_p(θ))
    if length(pred) == length(y_train)
        return sum(abs2, y_train .- pred), pred            
    else
        return Inf, y_train
    end
end


loss_p(p)

# optimization
optfun = OptimizationFunction((x, p) -> loss_p(x), Optimization.AutoZygote())
optprob = OptimizationProblem(optfun, p)
res1_uode = Optimization.solve(optprob, ADAM(0.01), maxiters=500)
optprob = OptimizationProblem(optfun, res1_uode.u)
res2_uode = Optimization.solve(optprob, ADAM(0.01), maxiters=500)
optprob = OptimizationProblem(optfun, res2_uode.u)
res3_uode = Optimization.solve(optprob, ADAM(0.001), maxiters=500)

# solve for the predicted production
prob_opt = ODEProblem(parameterized_model!, [u0s[i]], (minimum(timesteps), maximum(timesteps)), p, saveat=timesteps)
predict_p_opt(θ) = solve(prob_opt, Tsit5(), p=θ, abstol=1e-6, reltol=1e-6, sensealg=InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)), verbose=false, saveat=timesteps)

