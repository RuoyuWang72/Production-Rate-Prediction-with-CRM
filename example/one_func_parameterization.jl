pwd()
cd("github-projects/Production-Rate-Prediction-with-CRM")
include("../src/utils.jl")

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


timesteps = collect(1:1:2000)

# injections
inj1(t) = (500 >= t) ? 700 .+ t : (800 >= t) ? 450 .+ t : (1400 <=t<= 2000) ? 2400 .- t : 0
inj2(t) = (240 >= t) ? 400 .+ t : (700 >= t) ? 850 : (700 <= t <=1300) ? 1200 : 500

plot(timesteps, inj1, grid=false, framestyle=:box, label="Injector 1", title="Injections vs. Time")
plot!(timesteps, inj2, xlabel="Time (Days)", ylabel="Injection (bbl)", label="Injector 2")

bhp1(t) = 60 .* sin.(1e-2 .* t) .+ 200
bhp2(t) = 55 .* sin.(1.5e-2 .* t) .+ 130
bhp3(t) = 80 .* cos.(1.2e-2 .* t) .+ 170
bhp4(t) = 70 .* cos.(0.8e-2 .* t) .+ 125

plot(timesteps, [bhp1, bhp2, bhp3, bhp4], plot_title="BHP vs. Time", label=["BHP1" "BHP2" "BHP3" "BHP4"], xlabel="Time", ylabel="BHP", grid=false, framestyle=:box)

J1(t) = 20 .- t .^ 0.3
J2(t) = 25 .- t .^ 0.4
J3(t) = 17 .- t .^ 0.35
J4(t) = 30 .- t .^ 0.4

plot(timesteps, [J1, J2, J3, J4], plot_title="Productivity Index vs. Time", label=[L"J_1" L"J_2" L"J_3" L"J_4"], xlabel="Time", ylabel="Productivity Index", grid=false, framestyle=:box)


function crm_bhp1!(du, u, p, t)
    τ, F1, F2 = p
    du[1] = (inj1(t)*F1 + inj2(t)*F2 - τ*J(t)*Lux.gradient(bhp, t)[1] .- u[1]) ./ τ
end

u0s = [1550, 700, 800, 900]
tspan = (minimum(timesteps), maximum(timesteps))
bhps = [bhp1, bhp2, bhp3, bhp4]
τs = [10, 20, 30, 40]
Js = [J1, J2, J3, J4]
F1s = [0.1, 0.2, 0.5, 0.1]
F2s = [0.2, 0.1, 0.6, 0.1]

τ = τs[1]; J(t) = Js[1](t); bhp(t) = bhps[1](t)
solutions = Dict()
p = plot(xlabel="Time (day)", ylabel="Total Production (bbl)", title="CRMP: Forward Solution", label=false, grid=false, framestyle=:box)
for i in 1:4
    bhp(t) = bhps[i](t)
    J(t) = Js[i](t)
    τ, F1, F2, u0 = τs[1], F1s[i], F2s[i], [u0s[i]]
    prob = ODEProblem(crm_bhp1!, u0, tspan, [τ, F1, F2], saveat=timesteps)
    sol = solve(prob)
    # push!(solutions, sol)
    solutions[i] = sol
    plot!(p, sol, label="Producer $i")
end

p

# try optimization by treating parameters as constants
function crm_bhp2!(du, u, p, t)
    τ, J, F1, F2 = p
    du[1] = (inj1(t)*F1 + inj2(t)*F2 - τ*J*Lux.gradient(bhp, t)[1] .- u[1]) ./ τ
end

opt_params = Dict()
Threads.@threads for i in 1:4
    sol = solutions[i]
    bhp(t) = bhps[i](t)
    
    # define loss function for parameter Optimization, here we only use 80% of the data to fit
    (x_train, y_train), (x_test, y_test) = splitobs((timesteps, Array(sol)); at=0.8, shuffle=false)
    function obj(p, x)
        prob_ = ODEProblem(crm_bhp2!, x, (minimum(x_train), maximum(x_train)), p)
        sol_ = solve(prob_, saveat=x_train)
        return sum(abs2, Array(sol_) .- y_train)
    end
    
    # define constraint
    cons(res, p, x) = (res .= [p[3]+p[4]])

    # initial guess
    initial = [6.0, 3.0, 0.1, 0.2]

    # define lower and upper bounds
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [Inf, Inf, 1.0, 1.0]
    
    optprob = OptimizationFunction(obj, Optimization.AutoForwardDiff(), cons=cons)
    prob = OptimizationProblem(optprob, initial, [u0s[i]], lb=lower, ub=upper, lcons=[0.0], ucons=[2.0])
    opt_param = solve(prob, Optim.IPNewton(), iterations=10000, outer_iterations=10000)
    
    opt_params[i] = opt_param
end

fig1 = plot(layout=(2,2), size=(800, 800), xlabel="Time (Days)", ylabel="Total Production (bbl)", title="Optimization Results", linewidth=3, framestyle=:box)
for i in 1:4
    sp = fig1[i]
    sol = solutions[i]
    opt_param = opt_params[i]
    opt_sol = solve(ODEProblem(crm_bhp2!, [u0s[i]], tspan, opt_param, saveat=timesteps))
    plot!(sp, sol, label="Actual")
    plot!(sp, opt_sol, label="Optimization")
    title!(sp, "Producer $i")
    vline!(sp, [1600], linestyle=:dash, label=false, linecolor=:black)
    # annotate!(sp, (800, 1800, text("Train", 10)), (1800, 1800, text("Test", 10)))
end

plot(fig1)

for i in 1:4
    println(
        "Optimized parameters:\t $(round(opt_params[i][3], digits=3)), $(round(opt_params[i][4], digits=1)), $(round(opt_params[i][1], digits=3)), $(round(opt_params[i][2], digits=1))")
    println(
        "True parameters:\t $(F1s[i]), $(F2s[i]), $(τs[i])")
end

optimized_component_arrays1 = Dict()
Threads.@threads for i in 1:4
    m_J = create_nn(leakyrelu, 1, 10, 32, 1)
    rng = Random.default_rng()
    Random.seed!(rng, 1)
    
    ps_J, st_J = Lux.setup(rng, m_J)
    ps_F1 = opt_params[i][3]
    ps_F2 = opt_params[i][4]
    ps_τ = opt_params[i][1]
    ps_J = ComponentArray(ps_J)
    p = ComponentArray{eltype(ps_J)}()
    p = ComponentArray(p;ps_τ)
    p = ComponentArray(p;ps_J)
    p = ComponentArray(p;ps_F1)
    p = ComponentArray(p;ps_F2)

    function crm_bhp3!(du, u, p, t)
        τ, J, F1, F2 = p
        du[1] = (inj1(t)*F1 + inj2(t)*F2 - τ*J*dp(t) .- u[1]) ./ τ
    end
    
    dp(t) = Lux.gradient(bhps[i], t)[1]
    
    function parameterized_crm_bhp!(du, u, p, t)
        p_ = [p.ps_τ, m_J([t], p.ps_J, st_J)[1][1], p.ps_F1, p.ps_F2]
        crm_bhp3!(du, u, p_, t)
    end
    
    (x_train, y_train), (x_val, y_val) = splitobs((timesteps, Array(solutions[i])); at=0.8, shuffle=false)
    
    (x_val, y_val), (x_test, y_test) = splitobs((x_val, y_val); at=0.5, shuffle=false)
    
    prob = ODEProblem(parameterized_crm_bhp!, [u0s[i]], (minimum(x_train), maximum(x_train)), p, saveat=x_train)
    predict_p(θ) = solve(prob, p=θ, abstol=1e-6, reltol=1e-6, sensealg=InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)), verbose=false)
    
    function loss_p(θ)
        pred = Array(predict_p(θ))
        if length(pred) == length(y_train)
            return sum(abs2, y_train .- pred), pred
        else
            return Inf, y_train
        end
    end
    
    loss_p(p)
    # const losses = []
    callback(θ, l, pred) = begin
        push!(losses, l)
        if length(losses) % 50 == 0
            println(losses[end])
        end
        false
    end
    
    optfun = OptimizationFunction((x, p) -> loss_p(x), Optimization.AutoZygote())
    optprob = OptimizationProblem(optfun, p)
    res1_uode = Optimization.solve(optprob, ADAM(0.01), maxiters=500) # callback = callback

    optprob = OptimizationProblem(optfun, res1_uode.u)
    res2_uode = Optimization.solve(optprob, ADAM(0.001), maxiters=500) # callback=callback

    optimized_component_arrays1[i] = res1_uode
end



fig2 = plot(layout=(2,2), size=(800, 800), xlabel="Time (Days)", ylabel="Total Production (bbl)", title="Optimization Results", linewidth=3, framestyle=:box)
for i in 1:4
    sp = fig2[i]
    p = optimized_component_arrays1[i]
    prob = ODEProblem(parameterized_crm_bhp!, [u0s[i]], tspan, p.u, saveat=1)
    opt_sol = solve(prob, abstol=1e-6, reltol=1e-6, sensealg=InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)), verbose=false)
    sol = solutions[i]
    plot!(sp, sol, label="Actual")
    plot!(sp, opt_sol, label="Optimization")
    title!(sp, "Producer $i")
    vline!(sp, [1600], linestyle=:dash, label=false, linecolor=:black)
    annotate!(sp, (800, maximum(sol)*0.8, text("Train", 10)), (1800, maximum(sol)*0.8, text("Test", 10)))
end

plot(fig2)


"""
case 1 and 3 seem better than 2 and 4, why?
"""

# visualize the productivities
fig3 = plot(layout=(2,2), size=(800, 800), xlabel="Time (Days)", ylabel="Productivity Index", title="Optimization Results", linewidth=3, framestyle=:box)
for i in 1:4
    sp = fig3[i]
    optimized_J(t) = m_J([t], optimized_component_arrays1[i].u.ps_J, st_J)[1][1]
    J(t) = Js[i](t)
    plot!(sp, timesteps, J, label="Actual")
    plot!(sp, timesteps, optimized_J, label="Optimization")
    title!(sp, "Producer $i")
    vline!(sp, [1600], linestyle=:dash, label=false, linecolor=:black)
    # annotate!(sp, (800, maximum(sol)*0.8, text("Train", 10)), (1800, maximum(sol)*0.8, text("Test", 10)))
    println("Actual Parameters for Producer $i are: $(F1s[i]), $(F2s[i]), $(τs[i])")
    println("Optimized Parameters for Producer $i are: $(optimized_component_arrays1[i].u.ps_F1), $(optimized_component_arrays1[i].u.ps_F2), $(optimized_component_arrays1[i].u.ps_τ)")
end

plot(fig3)


"""
producer 3 can probably be improved
"""
i = 2
productivity_index(t) = m_J([t], optimized_component_arrays[i].u.ps_J, st_J)[1][1]

function crm_bhp3!(du, u, p, t)
    τ, J, F1, F2 = p
    du[1] = (inj1(t)*F1 + inj2(t)*F2 - τ*J*dp(t) .- u[1]) ./ τ
end

dp(t) = Lux.gradient(bhps[i], t)[1]

function parameterized_crm_bhp!(du, u, p, t)
    p_ = [p.ps_τ, m_J([t], p.ps_J, st_J)[1][1], p.ps_F1, p.ps_F2]
    crm_bhp3!(du, u, p_, t)
end

p = optimized_component_arrays1[i].u

(x_train, y_train), (x_val, y_val) = splitobs((timesteps, Array(solutions[i])); at=0.8, shuffle=false)

(x_val, y_val), (x_test, y_test) = splitobs((x_val, y_val); at=0.5, shuffle=false)

prob = ODEProblem(parameterized_crm_bhp!, [u0s[i]], (minimum(x_train), maximum(x_train)), p, saveat=x_train)
predict_p(θ) = solve(prob, p=θ, abstol=1e-6, reltol=1e-6, sensealg=InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)), verbose=false)

function loss_p(θ)
    pred = Array(predict_p(θ))
    if length(pred) == length(y_train)
        return sum(abs2, y_train .- pred), pred
    else
        return Inf, y_train
    end
end

loss_p(p)
# const losses = []
callback(θ, l, pred) = begin
    push!(losses, l)
    if length(losses) % 50 == 0
        println(losses[end])
    end
    false
end

optfun = OptimizationFunction((x, p) -> loss_p(x), Optimization.AutoZygote())
optprob = OptimizationProblem(optfun, p)
res1_uode = Optimization.solve(optprob, ADAM(0.01), maxiters=500) # callback = callback

optprob = OptimizationProblem(optfun, res1_uode.u)
res2_uode = Optimization.solve(optprob, ADAM(0.001), maxiters=500) # callback=callback



prob = ODEProblem(parameterized_crm_bhp!, [u0s[i]], tspan, res1_uode.u, saveat=1)
opt_sol = solve(prob, abstol=1e-6, reltol=1e-6, sensealg=InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)), verbose=false)
plot(solutions[i], label="Actual")
# plot!(predict_p(p))
plot!(opt_sol, label="Optimization")
title!("Producer $i")
vline!([1600], linestyle=:dash, label=false, linecolor=:black, size=(400, 400), framestyle=:box)
annotate!((800, maximum(solutions[i])*0.8, text("Train", 10)), (1800, maximum(solutions[i])*0.8, text("Test", 10)))


productivity_index(t) = m_J([t], res1_uode.u.ps_J, st_J)[1][1]

plot(timesteps, J1, label="Actual", xlabel="Time (Days)", ylabel="Productivity Index")
plot!(timesteps, productivity_index, label="Optimization")
title!("Producer $i")
vline!([1600], linestyle=:dash, label=false, linecolor=:black, size=(400, 400), framestyle=:box)