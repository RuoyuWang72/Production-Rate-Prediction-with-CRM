pwd()
cd("github-projects/Production-Rate-Prediction-with-CRM")
include("../../src/utils.jl")

using Pkg; Pkg.activate(".")
using Revise
using Distributions, Plots, IJulia, DifferentialEquations, Lux, Optimization, MLUtils, Random, Zygote, NNlib, ForwardDiff
using SciMLSensitivity
using CSV, DataFrames
using BenchmarkTools, LinearAlgebra
using LaTeXStrings
using OptimizationPolyalgorithms, OptimizationOptimisers, OptimizationOptimJL, OptimizationFlux, OptimizationNLopt
using ComponentArrays

gr()


df = DataFrame(CSV.File("./data/sample_data.csv"))
first(df, 5)

timesteps = df[:, :time]

inj1 = df[:, 2]
inj2 = df[:, 4]

I1(t) = inj1[searchsortednearest(timesteps, t)]
I2(t) = inj2[searchsortednearest(timesteps, t)]

plot(timesteps, I1, grid=false, framestyle=:box, label="Injector 1", title="Injections vs. Time")
plot!(timesteps, I2, xlabel="Time (Days)", ylabel="Injection (bbl)", label="Injector 2")

function crm_bhp1!(du, u, p, t)
    τ, F1, F2 = p
    du[1] = (I1(t)*F1 + I2(t)*F2 .- u[1]) ./ τ
end

prod1 = df[:, :Prod1]
prod2 = df[:, :Prod2]
prod3 = df[:, :Prod3]
prod4 = df[:, :Prod4]

u0s = [prod1[1], prod2[1], prod3[1], prod4[1]]
tspan = (minimum(timesteps), maximum(timesteps))
prods = [prod1, prod2, prod3, prod4]

opt_params = Dict()

Threads.@threads for i in 1:4
    sol = prods[i]
    
    # define loss function for parameter Optimization, here we only use 80% of the data to fit
    (x_train, y_train), (x_test, y_test) = splitobs((timesteps, sol); at=0.8, shuffle=false)
    function obj(p, x)
        prob_ = ODEProblem(crm_bhp1!, x, (minimum(x_train), maximum(x_train)), p)
        sol_ = solve(prob_, Rosenbrock23(autodiff=false), saveat=1)
        return sum(abs2, Array(sol_) .- reshape(y_train, 1, length(y_train)))
    end

    # initial guess
    initial = [6.0, 0.1, 0.2]

    # define lower and upper bounds
    lower = [0.0, 0.0, 0.0]
    upper = [Inf, 1.0, 1.0]

    # define constraint
    constr(res, p, x) = (res .= [p[2]+p[3]])
    optprob = OptimizationFunction(obj, Optimization.AutoForwardDiff(), cons=constr)
    prob = OptimizationProblem(optprob, initial, [u0s[i]], lb=lower, ub=upper, lcons=[0.0], ucons=[2.0])
    opt_param = solve(prob, Optim.IPNewton())
    
    opt_params[i] = opt_param
    # println(obj(initial, [u0s[i]]))
end

fig1 = plot(layout=(2,2), size=(800, 800), xlabel="Time (Days)", ylabel="Total Production (bbl)", title="Optimization Results", linewidth=3, framestyle=:box)
for i in 1:4
    sp = fig1[i]
    sol = prods[i]
    opt_param = opt_params[i]
    opt_sol = solve(ODEProblem(crm_bhp1!, [u0s[i]], tspan, opt_param, saveat=timesteps))
    plot!(sp, timesteps, sol, label="Actual")
    plot!(sp, opt_sol, label="Optimization")
    title!(sp, "Producer $i")
    vline!(sp, [150*0.8], linestyle=:dash, label=false, linecolor=:black)
    annotate!(sp, (800, 1800, text("Train", 10)), (1800, 1800, text("Test", 10)))
end

plot(fig1)