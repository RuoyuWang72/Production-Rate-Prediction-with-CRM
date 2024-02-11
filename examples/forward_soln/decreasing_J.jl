pwd()
include("../../src/utils.jl")

# packages used
using Pkg; Pkg.acativate(".")
using Revise
using Distributions, Plots, IJulia, DifferentialEquations, Lux, Optimization, MLUtils, Random, Zygote, NNlib, ForwardDiff
using SciMLSensitivity
using CSV, DataFrames
using BenchmarkTools, LinearAlgebra
using LaTeXStrings
using OptimizationPolyalgorithms, OptimizationOptimisers, OptimizationOptimJL, OptimizationFlux, OptimizationNLopt
using ComponentArrays
using Statistics

# backend for plotting
gr()

# generate timesteps
timesteps = collect(1:1:2001)

# generate synthetic injection rates from two injectors
w1(t) = (500 >= t) ? 7000 .+ t : (800 >= t) ? 4500 .+ t : (1400 <=t<= 2000) ? 6000 .- t : 6500
w2(t) = (240 >= t) ? 4000 .+ t : (700 >= t) ? 8500 : (700 <= t <=1300) ? 3200 : 6000

# generate synthetic declining productivity indices for the four producers
J1(t) = 30 .- t .^ 0.4
J2(t) = 20 .- t .^ 0.35
J3(t) = 40 .- 0.02 .* t
J4(t) = 50 .- 0.025 .* t
Js = [J1, J2, J3, J4]

# generate synthetic bottom-hole pressure for the four producers
Pwf1(t) = 60 .* sin.(0.3e-2 .* t) .+ 100
Pwf2(t) = 55 .* sin.(0.5e-2 .* t) .+ 130
Pwf3(t) = 80 .* cos.(0.7e-2 .* t) .+ 170
Pwf4(t) = 70 .* cos.(0.8e-2 .* t) .+ 125
Pwfs = [Pwf1, Pwf2, Pwf3, Pwf4]

# synthetic gains for forward solution computation
gains = [[0.30, 0.10], [0.15, 0.2], [0.4, 0.4], [0.15, 0.3]]

# synthetic production on day 1 for forward solution computation
u0s = [2000, 2400, 2500, 3000]

# synthetic pore compressibilities 
CtVps = [2000, 1500, 1700, 2100]

# timespan
tspan = (minimum(timesteps), maximum(timesteps))

# the Capacitance Resistance Model in differential form
function model!(dq, q, p, t)
    CtVp, gain = p
    
    # compute the weighted sum of w(t) functions
    w_sum = gain[1] * w1(t) + gain[2] * w2(t)
    
    # compute dq/dt using the derived expression
    dq[1] = (w_sum * J(t)^2 - q[1] * J(t)^2 + q[1] * J'(t) * CtVp + J(t) * Pwf(t) * J'(t) * CtVp - J'(t) * Pwf(t) * J(t) * CtVp - J(t) * Pwf'(t) * J(t) * CtVp) / (J(t) * CtVp)
end

# compute for the forward solution
forward_solns = Dict()
J(t) = J1(t)
Pwf(t) = Pwf1(t)
for i in 1:4
    J(t) = Js[i](t)
    Pwf(t) = Pwfs[i](t)
    prob = ODEProblem(model!, [u0s[i]], tspan, [CtVps[i], gains[i]], saveat=timesteps)
    sol = solve(prob)
    forward_solns[i] = sol
end

# first, we treat all parameters as constants to establish a baseline model
stat_solns = Dict()
stat_param = Dict()

# the Capacitance Resistance Model in differential form with constant pore compressibilities, productivity indices and gains
function crmp_constant!(du, u, p, t)
    CtVp, J, F1, F2 = p
    du[1] = (w1(t)*F1 + w2(t)*F2 - CtVp*Pwfs[i](t) .- u[1]) ./ (CtVp / J)
end

# set the constraint that the gains sum up to less than or equal to 1
cons(res, p, x) = (res .= [p[3]+p[4]])

# initial guess
initial = [6.0, 3.0, 0.1, 0.2]

# define lower and upper bounds for the four parameters
lower = [0.0, 0.0, 0.0, 0.0]
upper = [Inf, Inf, 1.0, 1.0]

# optimize for constant parameters
i = 1
for i in 1:4
    prod = vec(Array(forward_solns[i]))
    (x_train, y_train), (x_val, y_val) = splitobs((timesteps, prod); at=0.7, shuffle=false)
    (x_val, y_val), (x_test, y_test) = splitobs((x_val, y_val); at=0.5, shuffle=false)
    # define the objective function
    function obj(p, x)
        prob_ = ODEProblem(crmp_constant!, x, (minimum(x_train), maximum(x_train)), p)
        sol_ = solve(prob_, saveat=timesteps)
        return sum(abs2, Array(sol_) .- prod)
    end
    optprob = OptimizationFunction(obj, Optimization.AutoForwardDiff(), cons=cons)
    prob = OptimizationProblem(optprob, initial, [u0s[i]], lb=lower, ub=upper, lcons=[0.0], ucons=[2.0])
    opt_param = solve(prob, Optim.IPNewton(), iterations=10000, outer_iterations=10000)
    sol_prob = ODEProblem(crmp_constant!, [u0s[i]], tspan, opt_param, saveat=timesteps)
    sol_opt = solve(sol_prob)
    stat_solns[i] = sol_opt
    stat_param[i] = opt_param
end

# initialize a neural network for J
m_J = create_nn(tanh, 1, 10, 32, 10, 1)

# create componenet array
rng = Random.default_rng()
ps_J, st_J = Lux.setup(rng, m_J)
ps_CtVp = stat_param[1]
ps_gain1 = stat_param[3]
ps_gain2 = stat_param[4]
ps_J = ComponentArray(ps_J)
p = ComponentArray{eltype(ps_J)}()
p = ComponentArray(p; ps_CtVp, ps_gain1, ps_gain2, ps_J)

# define dynamic CRMP
function crmp_dynamic!(du, u, p, t)
    CtVp, gain1, gain2, J, J_prime = p
    w_sum = gain1 * w1(t) + gain2 * w2(t)
    du[1] = (w_sum * J^2 - u[1] * J^2 + u[1] * J_prime * CtVp
    + J * Pwf(t) * J_prime * CtVp - J_prime * Pwf(t) * J * CtVp - J * Pwf'(t) * J * CtVp) / (J * CtVp)
end

# parameterize the dynamic CRMP
function parameterized_model!(du, u, p, t)
    J_func(t) = m_J([t], p.ps_J, st_J)[1][1]
    J = J_func(t)
    J_prime = gradient(t -> J_func(t), t)[1]
    p_ = [p.ps_CtVp, p.ps_gain1, p.ps_gain2, J, J_prime]
    crmp_dynamic!(du, u, p_, t)
end

# Producer 1:
i = 1
prod = vec(Array(forward_solns[i]))
(x_train, y_train), (x_val, y_val) = splitobs((timesteps, prod); at=0.7, shuffle=false)
(x_val, y_val), (x_test, y_test) = splitobs((x_val, y_val); at=0.5, shuffle=false)

prob_nn = ODEProblem(parameterized_model!, u0s[i], (minimum(x_train), maximum(x_train)), p, saveat=1)
sol_nn = solve(prob_nn)
predict_p(θ) = solve(prob_nn, p=θ, abstol=1e-6, reltol=1e-6, sensealg=InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)), verbose=false)

# define loss function 
function loss_p(θ)
    pred = Array(predict_p(θ))
    if length(pred) == length(y_train)
        return sum(abs2, y_train .- pred), pred            
    else
        return Inf, y_train
    end
end

# use ForwardDiff for the optimization function
optfun = OptimizationFunction((x, p) -> loss_p(x), Optimization.AutoZygote())

# create the optimization problem
optprob = OptimizationProblem(optfun, p)

# solve the optimization problem
res1_uode = Optimization.solve(optprob, ADAM(0.01), maxiters=500) # callback = callback

