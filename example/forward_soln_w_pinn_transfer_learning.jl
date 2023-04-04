versioninfo()
Threads.nthreads()
pwd()
cd("Production-Rate-Prediction-with-CRM")

using Pkg; Pkg.activate(".")
using Revise
using Distributions, Plots, IJulia, DifferentialEquations, Lux, Optimization, MLUtils, Random, Zygote, NNlib, ForwardDiff
using SciMLSensitivity
using CSV, DataFrames
using BenchmarkTools, LinearAlgebra
using LaTeXStrings
using OptimizationPolyalgorithms, OptimizationOptimisers, OptimizationOptimJL, OptimizationFlux

# include("src/functions.jl")
# include("src/reservoir.jl")

gr()


# time
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

plot(timesteps, [J1, J2, J3, J4], plot_title="Productivity Index vs. Time", label=["J1" "J2" "J3" "J4"], xlabel="Time", ylabel="Productivity Index", grid=false, framestyle=:box)

τ1(t) = 0.04 .* t .+ 10
τ2(t) = 0.03 .* t .+ 20
τ3(t) = 0.03 .* t .+ 25
τ4(t) = 0.02 .* t .+ 15

# τ1(t) = 30 .* t^-0.1 .+ 2
# τ2(t) = 30 .* t^-0.3 .+ 5
# τ3(t) = 15 .* t^-0.2 .+ 7
# τ4(t) = 20 .* t^-0.3 .+ 5

plot(timesteps, [τ1, τ2, τ3, τ4], plot_title="Time Constant vs. Time", label=["τ1" "τ2" "τ3" "τ4"], xlabel="Time", ylabel="Time Constant", grid=false, framestyle=:box)

# define the CRM Differential Equation and compute the forward solution
function crm_bhp1!(du, u, p, t)
    F1, F2 = p
    du[1] = (inj1(t)*F1 + inj2(t)*F2 - τ(t)*J(t)*Lux.gradient(bhp, t)[1] .- u[1]) ./ τ(t)
end

u0s = [1550, 700, 800, 900]
tspan = (minimum(timesteps), maximum(timesteps))
bhps = [bhp1, bhp2, bhp3, bhp4]
τs = [τ1, τ2, τ3, τ4]
Js = [J1, J2, J3, J4]
F1s = [0.1, 0.2, 0.5, 0.1]
F2s = [0.2, 0.1, 0.6, 0.1]
τ(t) = τs[1](t); J(t) = Js[1](t); bhp(t) = bhps[1](t)
solutions = []
p = plot(xlabel="Time (day)", ylabel="Total Production (bbl)", title="CRMP: Forward Solution", label=false, grid=false, framestyle=:box)
for i in 1:4
    bhp(t) = bhps[i](t)
    τ(t) = τs[i](t)
    J(t) = Js[i](t)
    F1, F2, u0 = F1s[i], F2s[i], [u0s[i]]
    prob = ODEProblem(crm_bhp1!, u0, tspan, [F1, F2], saveat=timesteps)
    sol = solve(prob)
    push!(solutions, sol)
    # solutions[i] = sol
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

fig = plot(layout=(2,2), size=(800, 800), xlabel="Time (Days)", ylabel="Total Production (bbl)", title="Optimization Results", linewidth=3, framestyle=:box)
for i in 1:4
    sp = fig[i]
    sol = solutions[i]
    opt_param = opt_params[i]
    opt_sol = solve(ODEProblem(crm_bhp2!, [u0s[i]], tspan, opt_param, saveat=timesteps))
    plot!(sp, sol, label="Actual")
    plot!(sp, opt_sol, label="Optimization")
    title!(sp, "Producer $i")
    vline!(sp, [1600], linestyle=:dash, label=false, linecolor=:black)
    # annotate!(sp, (800, 1800, text("Train", 10)), (1800, 1800, text("Test", 10)))
end

plot(fig)
# savefig("./opt_with_known_bhp.png")
for i in 1:4
    println(round(opt_params[i][3], digits=3), ", ", round(opt_params[i][4], digits=1))
    println(F1s[i], ", ", F2s[i])
end

"""
Assume we know the connectivities, we visualize the loss landscape as a function of **time constant** and **productivity index**

"""
τ_mesh = range(0, 100; length=100)
J_mesh = range(0, 100; length=100)
losses = Dict()
# sol_mesh = Dict()
Threads.@threads for i in 1:4
    sol = solutions[i]
    bhp(t) = bhps[i](t)
    # (x_train, y_train), (x_test, y_test) = splitobs((timesteps, Array(sol)); at=0.8, shuffle=false)
    function plot_loss_landscape(τ, J)
        prob_ = ODEProblem(crm_bhp2!, [u0s[i]], tspan, [F1s[i], F2s[i], τ, J])
        sol_ = solve(prob_, saveat=1)
        return sum(abs2, Array(sol_) .- Array(sol))
    end

    L = plot_loss_landscape.(τ_mesh', J_mesh)
    losses[i] = L
    # sol_mesh[i] = S|
end

plotly()
fig2 = plot(layout=(2,2), size=(800, 800), title="Loss Landscape", linewidth=3, framestyle=:box, xlabel="τ", ylabel="J", zlabel="L")

for i in 1:4
    sp = fig2[i]
    L = losses[i]
    wireframe!(sp, τ_mesh, J_mesh, L, colorbar=false, title="Loss Landscape of Producer $i", xlabel="τ", ylabel="J", zlabel="L")
end

display(fig2)
losses[1] == losses[2]
"""
what about at each individual timesteps?

we compute the loss at time t=ti by solving for the ODE solution within the timespan [ti, ti+10^-2],
saveat=10^-4, u0=u_actual(t=ti), compute the average of the solution vector, and compare it against the actual u
"""

# solve for the solution mesh for every case
sol_meshes = Dict()
Threads.@threads for i in 1:4
    sol = solutions[i]
    bhp(t) = bhps[i](t)
    function compute_mesh_grid_ode_sol(τ, J)
        prob_ = ODEProblem(crm_bhp2!, [u0s[i]], tspan, [F1s[i], F2s[i], τ, J])
        sol_ = solve(prob_, saveat=1)
        return sol_
    end

    sol_mesh = compute_mesh_grid_ode_sol.(τ_mesh', J_mesh)
    sol_meshes[i] = sol_mesh
end


# min_losses_position = []
# for ti in 1:2000
#     sol_at_ti = [sol_meshes[1][i,j](ti)[1] for i in 1:100, j in 1:100]
#     sol_actual_at_ti = [solutions[1](ti)[1] for i in 1:100, j in 1:100]
#     loss = sum.(abs2, sol_actual_at_t10 .- sol_at_t10)
#     push!(min_losses_position, argmin(loss))
# end

sol_at_t10 = [sol_meshes[1][i,j](10)[1] for i in 1:100, j in 1:100]
sol_actual_at_t10 = [solutions[1](10)[1] for i in 1:100, j in 1:100]
loss = sum.(abs2, sol_actual_at_t10 .- sol_at_t10)
argmin(loss)


"""
is delta t=1 too big?
"""
i = 1
sol = solutions[i]
bhp(t) = bhps[i](t)
ti = 10
function sensitivity_test(τ, J)
    prob_ = ODEProblem(crm_bhp2!, sol(ti), (ti, ti+1), [F1s[i], F2s[i], τ, J])
    sol_ = solve(prob_)
    return sol_
end

sensitivity_test(1, 1)
sol[10:11]
sensitivity_test(τ1(ti+1), J1(ti+1))



# Threads.@threads for i in 1:4   
for i in 
    sol_at_t10 = [L[i,j](10)[1] for i in 1:5, j in 1:5]
end
# actual solution at t=ti
sol_actual_at_t10 = [sol(10)[1] for i in 1:5, j in 1:5]

# compute the loss grid
loss = sum.(abs2, sol_actual_at_t10 .- sol_at_t10)
plot_loss_landscape(0, 1.25)(10)

sol_at_t10

plotly()
wireframe(τ_mesh, J_mesh, loss, colorbar=false, title="Loss Landscape of Producer $i", xlabel="τ", ylabel="J", zlabel="L")

size(sol_meshes[1], 1)

loss




wireframe(τ_mesh, J_mesh, L, colorbar=false, title="Loss Landscape of Producer $i", xlabel="τ", ylabel="J", zlabel="L")
L[1,1]








t = 10
Threads.@threads for i in 1:4
    sol = solutions[i]
    bhp(t) = bhps[i](t)
    # (x_train, y_train), (x_test, y_test) = splitobs((timesteps, Array(sol)); at=0.8, shuffle=false)
    function plot_loss_landscape(τ, J)
        prob_ = ODEProblem(crm_bhp2!, [u0s[i]], tspan, [F1s[i], F2s[i], τ, J])
        sol_ = solve(prob_, saveat=1)
        return sum(abs2, Array(sol_) .- Array(sol))
    end

    L = plot_loss_landscape.(τ_mesh', J_mesh)
    losses[i] = L
    # sol_mesh[i] = S
end


























# plotly()
fig3 = plot(layout=(2,2), size=(800, 800), title="Loss Landscape", linewidth=3, framestyle=:box, xlabel="τ", ylabel="J", zlabel="L")

for i in 1:4
    sp = fig3[i]
    L = losses2[i]
    surface!(sp, τ_mesh, J_mesh, L, colorbar=false, title="Loss Landscape of Producer $i", xlabel="τ", ylabel="J", zlabel="L")
end

display(fig3)
idx = sortperm(losses2[2][:], rev=false)[1:10] # get the indexes of the 10 smallest numbers
println(idx)


losses2[2]



using Interact
@manipulate for i in 1:1999
    u0 = sol(i)
    sol_true = Array(sol(i:i+1))

    function plot_loss_landscape(τ, J)
        prob_ = ODEProblem(crm_bhp2!, u0, (i, i+1), [F1s[1], F2s[1], τ, J])
        sol_ = solve(prob_, saveat=1)
        return sum(abs2, Array(sol_) .- sol_true)
    end

    L = plot_loss_landscape.(τ_mesh', J_mesh)

    surface(τ_mesh, J_mesh, L, colorbar=false, title="Loss Landscape of Producer $i", xlabel="τ", ylabel="J", zlabel="L")
end

i=1
prob_ = ODEProblem(crm_bhp2!, [1], (i, i+1), [F1s[1], F2s[1], 1,1])
sol_ = solve(prob_, saveat=1)
Array(sol_)
Array(sol(1:2))

sol = solutions[1]

function plot_loss_landscape(start_time, end_time)
    prob_ = ODEProblem(crm_bhp2!, [u0s[1]], (minimum(x_train), maximum(x_train)), [F1s[1], F2s[1], τ, J])
    sol_ = solve(prob_, saveat=x_train)
    return sum(abs2, Array(sol_) .- y_train)
end

function obj(p, x)
    prob_ = ODEProblem(crm_bhp2!, x, (minimum(x_train), maximum(x_train)), p)
    sol_ = solve(prob_, saveat=x_train)
    return sum(abs2, Array(sol_) .- y_train)
end



# inj_norms = batch_norm_sca([inj1, inj2], timesteps, normalization_, 0.8)
# bhp_norms = batch_norm_vec(bhps, timesteps, normalization_, 0.8)
# solutions_norm = []
# for i in 1:4
#     sol = map(x -> x[1], solutions[i].u)
#     sol = normalization_(sol, sol[1:Int(length(sol)*0.8)])
#     push!(solutions_norm, sol)
# end



# function searchsortednearest(a, x)
#    idx = searchsortedfirst(a, x)
#    if (idx==1); return idx; end
#    if (idx>length(a)); return length(a); end
#    if (a[idx]==x); return idx; end
#    if (abs(a[idx]-x) < abs(a[idx-1]-x))
#       return idx
#    else
#       return idx-1
#    end
#  end

# i = 1
# sol = solutions_norm[i]
# # bhp(t) = link_xy(timesteps, bhp_norms[i])(t)
# # inj1(t) = link_xy(timesteps, inj_norms[1])(t)
# # inj2(t) = link_xy(timesteps, inj_norms[2])(t)
# bhp_norm(t) = normalization_(bhp1(t), bhp1(timesteps)[1:1600])
# inj_norms = batch_norm_sca([inj1, inj2], timesteps, normalization_, 0.8)
# inj1_norm(t) = normalization_(inj1(t), inj_norms[1])
# inj2_norm(t) = normalization_(inj2(t), inj_norms[2])

# function crm_bhp3!(du, u, p, t)
#     τ, J, F1, F2 = p
#     du[1] = (inj1_norm(t)*F1 + inj2_norm(t)*F2 - τ*J*Lux.gradient(bhp_norm, t)[1] .- u[1]) ./ τ
# end

# # # define loss function for parameter Optimization, here we only use 80% of the data to fit
# (x_train, y_train), (x_test, y_test) = splitobs((timesteps, Array(sol)); at=0.8, shuffle=false)
# prob_ = ODEProblem(crm_bhp3!, [800], (minimum(x_train), maximum(x_train)), [6.0, 3.0, 0.1, 0.2])
# sol_ = solve(prob_, saveat=1)
# # function obj(p, x)
# #     prob_ = ODEProblem(crm_bhp2!, x, (minimum(x_train), maximum(x_train)), p)
# #     sol_ = solve(prob_, saveat=x_train)
# #     return sum(abs2, Array(sol_) .- y_train)
# # end

# # # define constraint
# # cons(res, p, x) = (res .= [p[3]+p[4]])

# # # initial guess
# # initial = [6.0, 3.0, 0.01, 0.02]

# # # define lower and upper bounds
# # lower = [0.0, 0.0, 0.0, 0.0]
# # upper = [Inf, Inf, 1.0, 1.0]

# # optprob = OptimizationFunction(obj, Optimization.AutoForwardDiff(), cons=cons)
# # prob = OptimizationProblem(optprob, initial, [u0s[i]], lb=lower, ub=upper, lcons=[0.0], ucons=[2.0])
# # opt_param = solve(prob, Optim.IPNewton(), iterations=10000, outer_iterations=10000)

# # opt_params[i] = opt_param

p_list = []
j = 16
k = 32
l = 64
m = 128
n = 64
o = 32
q = 16
Threads.@threads for i in 1:1000
    rng = Random.default_rng()
    Random.seed!(rng, Int(i))

    m_τ = Lux.Chain(
        Lux.Dense(1, j, swish),
        Lux.Dense(j, k, swish),
        Lux.Dense(k, l, swish),
        Lux.Dense(l, m, swish),
        Lux.Dense(m, n, swish),
        Lux.Dense(n, o, swish),
        Lux.Dense(o, q, swish),
        Lux.Dense(q, 1, relu))
    ps_τ, st_τ = Lux.setup(rng, m_τ)
    
    
    m_J = Lux.Chain(
        Lux.Dense(1, j, swish),
        Lux.Dense(j, k, swish),
        Lux.Dense(k, l, swish),
        Lux.Dense(l, m, swish),
        Lux.Dense(m, n, swish),
        Lux.Dense(n, o, swish),
        Lux.Dense(o, q, swish),
        Lux.Dense(q, 1, relu))
    ps_J, st_J = Lux.setup(rng, m_J)
    
    m_p = Lux.Chain(
        Lux.Dense(1, j, swish),
        Lux.Dense(j, k, swish),
        Lux.Dense(k, l, swish),
        Lux.Dense(l, m, swish),
        Lux.Dense(m, n, swish),
        Lux.Dense(n, o, swish),
        Lux.Dense(o, q, swish),
        Lux.Dense(q, 1, swish))
    ps_p, st_p = Lux.setup(rng, m_p)

    ps_F1 = opt_params[1][3]
    ps_F2 = opt_params[1][4]

    ps_τ = Lux.ComponentArray(ps_τ)
    ps_J = Lux.ComponentArray(ps_J)
    ps_p = Lux.ComponentArray(ps_p)
    p = Lux.ComponentArray{eltype(ps_τ)}()
    p = Lux.ComponentArray(p;ps_τ)
    p = Lux.ComponentArray(p;ps_J)
    p = Lux.ComponentArray(p;ps_p)
    p = Lux.ComponentArray(p;ps_F1)
    p = Lux.ComponentArray(p;ps_F2)

    if loss_p(p)[1] != Inf
        push!(p_list, i)
        # println(loss_p(p)[1])
    end

    if length(p_list) >= 100
        break
    end
end

    

final_params_w_lowest_val_loss4 = Dict()
max_val_loss = Inf
for i in p_list
    rng = Random.default_rng()
    Random.seed!(rng, Int(i))
    m_τ = Lux.Chain(
        Lux.Dense(1, j, swish),
        Lux.Dense(j, k, swish),
        Lux.Dense(k, l, swish),
        Lux.Dense(l, m, swish),
        Lux.Dense(m, n, swish),
        Lux.Dense(n, o, swish),
        Lux.Dense(o, q, swish),
        Lux.Dense(q, 1, relu))
    ps_τ, st_τ = Lux.setup(rng, m_τ)


    m_J = Lux.Chain(
        Lux.Dense(1, j, swish),
        Lux.Dense(j, k, swish),
        Lux.Dense(k, l, swish),
        Lux.Dense(l, m, swish),
        Lux.Dense(m, n, swish),
        Lux.Dense(n, o, swish),
        Lux.Dense(o, q, swish),
        Lux.Dense(q, 1, relu))
    ps_J, st_J = Lux.setup(rng, m_J)

    m_p = Lux.Chain(
        Lux.Dense(1, j, swish),
        Lux.Dense(j, k, swish),
        Lux.Dense(k, l, swish),
        Lux.Dense(l, m, swish),
        Lux.Dense(m, n, swish),
        Lux.Dense(n, o, swish),
        Lux.Dense(o, q, swish),
        Lux.Dense(q, 1, swish))
    ps_p, st_p = Lux.setup(rng, m_p)

    ps_F1 = opt_params[1][3]
    ps_F2 = opt_params[1][4]

    ps_τ = Lux.ComponentArray(ps_τ)
    ps_J = Lux.ComponentArray(ps_J)
    ps_p = Lux.ComponentArray(ps_p)
    p = Lux.ComponentArray{eltype(ps_τ)}()
    p = Lux.ComponentArray(p;ps_τ)
    p = Lux.ComponentArray(p;ps_J)
    p = Lux.ComponentArray(p;ps_p)
    p = Lux.ComponentArray(p;ps_F1)
    p = Lux.ComponentArray(p;ps_F2)

    optfun = OptimizationFunction((x, p) -> loss_p(x), Optimization.AutoZygote())
    optprob = OptimizationProblem(optfun, p)

    res1_uode = Optimization.solve(optprob, ADAM(), maxiters=500) # callback=callback

    if loss_p(res1_uode.u)[1] != Inf
        val_prob = ODEProblem(parameterized_crm_bhp!, predict_p(res1_uode.u)[end], (minimum(x_val), maximum(x_val)), res1_uode.u, saveat=x_val)
        pred = Array(solve(val_prob))
        if length(pred) == length(y_val)
            val_loss = sum(abs2, y_val .- pred)
            if val_loss < max_val_loss
                max_val_loss = val_loss
                final_params_w_lowest_val_loss4[i] = (i, res1_uode.u, val_loss)
            end
        else
            val_loss = Inf
        end

    end
    # print(i)
    # print(loss_p(res1_uode.u)[1])
    # print(loss_p(p)[1])
end
    


final_params_w_lowest_val_loss4

final_params_w_lowest_val_loss4

[t[3] for t in values(final_params_w_lowest_val_loss4)]

loss_p(final_params_w_lowest_val_loss4[126][2])[1]

prob = ODEProblem(parameterized_crm_bhp!, [u0s[4]], tspan, final_params_w_lowest_val_loss4[141][2], saveat=timesteps)
soln = solve(prob)
plt4 = plot(grid=false, framestyle=:box, title="Producer 3", ylabel="Production Rates (bbl)")
plot!(plt4, solutions[4], label="Actual")
plot!(plt4, soln, label="Optimization", xlabel="Time (Days)")
vline!(plt4, [1600, 1800], linestyle=:dash, label=false, linecolor=:black)

plt2

# initializing neural nets
i = 126
j = 16
k = 32
l = 64
m = 128
n = 64
o = 32
q = 16

rng = Random.default_rng()
Random.seed!(rng, Int(i))

m_τ = Lux.Chain(
    Lux.Dense(1, j, swish),
    Lux.Dense(j, k, swish),
    Lux.Dense(k, l, swish),
    Lux.Dense(l, m, swish),
    Lux.Dense(m, n, swish),
    Lux.Dense(n, o, swish),
    Lux.Dense(o, q, swish),
    Lux.Dense(q, 1, relu))
ps_τ, st_τ = Lux.setup(rng, m_τ)


m_J = Lux.Chain(
    Lux.Dense(1, j, swish),
    Lux.Dense(j, k, swish),
    Lux.Dense(k, l, swish),
    Lux.Dense(l, m, swish),
    Lux.Dense(m, n, swish),
    Lux.Dense(n, o, swish),
    Lux.Dense(o, q, swish),
    Lux.Dense(q, 1, relu))
ps_J, st_J = Lux.setup(rng, m_J)

m_p = Lux.Chain(
    Lux.Dense(1, j, swish),
    Lux.Dense(j, k, swish),
    Lux.Dense(k, l, swish),
    Lux.Dense(l, m, swish),
    Lux.Dense(m, n, swish),
    Lux.Dense(n, o, swish),
    Lux.Dense(o, q, swish),
    Lux.Dense(q, 1, swish))
ps_p, st_p = Lux.setup(rng, m_p)

ps_F1 = opt_params[1][3]
ps_F2 = opt_params[1][4]

ps_τ = Lux.ComponentArray(ps_τ)
ps_J = Lux.ComponentArray(ps_J)
ps_p = Lux.ComponentArray(ps_p)
p = Lux.ComponentArray{eltype(ps_τ)}()
p = Lux.ComponentArray(p;ps_τ)
p = Lux.ComponentArray(p;ps_J)
p = Lux.ComponentArray(p;ps_p)
p = Lux.ComponentArray(p;ps_F1)
p = Lux.ComponentArray(p;ps_F2)

function crm_bhp4!(du, u, p, t)
    τ, J, dpdt, F1, F2 = p
    du[1] = (inj1(t)*F1 + inj2(t)*F2 - τ*J*dpdt .- u[1]) ./ τ
end

function parameterized_crm_bhp!(du, u, p, t)
    p_ = [m_τ([t], p.ps_τ, st_τ)[1][1], m_J([t], p.ps_J, st_J)[1][1], m_p([t], p.ps_p, st_p)[1][1], p.ps_F1, p.ps_F2]
    crm_bhp4!(du, u, p_, t)
end

(x_train, y_train), (x_val, y_val) = splitobs((timesteps, Array(solutions[4])); at=0.8, shuffle=false)

(x_val, y_val), (x_test, y_test) = splitobs((x_val, y_val); at=0.5, shuffle=false)


prob = ODEProblem(parameterized_crm_bhp!, [u0s[1]], (minimum(x_train), maximum(x_train)), p, saveat=x_train)
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
callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses) % 100 == 0
        println(losses[end])
    end
    false
end

loss_p(p)

optfun = OptimizationFunction((x, p) -> loss_p(x), Optimization.AutoZygote())
optprob = OptimizationProblem(optfun, p)

res1_uode = Optimization.solve(optprob, ADAM(), maxiters=500, callback=callback) # callback=callback

# plt1
plt1 = plot!(plt1, ylims=(0,2000))
plt2 = plot!(plt2, ylims=(0,2000))
plt3 = plot!(plt3, ylims=(0,2000))
plt4 = plot!(plt4, ylims=(0,2000))


plt1

plot(plt1, plt2, plt3, plt4, layout=(2, 2), size=(800, 800))
# savefig("./opt_wo_bhp.png")

plot(solutions[4], label="Actual")
plot!(predict_p(p), label="Initialized Parameters")
plot!(predict_p(res1_uode.u), label="Optimization", xlabel="Time", ylabel="BHP", grid=false, framestyle=:box)

optfun = OptimizationFunction((x, p) -> loss_p(x), Optimization.AutoZygote())
optprob = OptimizationProblem(optfun, res1_uode.u)

res2_uode = Optimization.solve(optprob, ADAM(), maxiters=1000) # callback=callback

plot(solutions[1], label="Actual")
plot!(predict_p(p), label="Initialized Parameters")
plot!(predict_p(res2_uode.u), label="Optimization")

# bhp_norms = batch_norm_vec(bhps, timesteps, normalization, 0.8)

# final_p_list = Dict()
# m1 = Lux.Chain(Lux.Dense(1 => 100, swish), Lux.Dense(100 => 1, swish))

# bhp_norms = batch_norm_vec(bhps, timesteps, normalization, 0.8)
# timesteps_norm = normalization(timesteps, timesteps)

# final_p_list = Dict()
# # model = Lux.Chain(Lux.Dense(1, 16, swish), Lux.Dense(16, 1))
# model = Lux.Chain(Lux.Dense(1 => 100, swish), Lux.Dense(100 => 100, swish), Lux.Dense(100 => 100, swish), Lux.Dense(100 => 100, swish), Lux.Dense(100 => 100, swish), Lux.Dense(100 => 1))
# opt = ADAM(0.01)

# function loss_function(model, ps, st, data)
#     y_pred, st = Lux.apply(model, data[1], ps, st)
#     mse_loss = mean(abs2, y_pred .- data[2])
#     return mse_loss, st, ()
# end

# tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
# vjp_rule = Lux.Training.ZygoteVJP()

# function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data::Tuple,
#     epochs::Int)
#     data = data .|> Lux.gpu
#     for epoch in 1:epochs
#     grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function,
#                                                             data, tstate)
#     @info epoch=epoch loss=loss
#     tstate = Lux.Training.apply_gradients(tstate, grads)
#     end
#     return tstate
# end


# (x_train, y_train), (x_test, y_test) = splitobs((timesteps_norm, bhp_norms[1]); at=0.8, shuffle=false)
# # function generate_data(rng::AbstractRNG)
# #     x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
# #     y = evalpoly.(x, ((0, -2, 1),)) .+ randn(rng, (1, 128)) .* 0.1f0
# #     return (x, y)
# # end
# # rng = MersenneTwister()
# # Random.seed!(rng, 12345)

# # (x_train, y_train) = generate_data(rng)

# tstate = main(tstate, vjp_rule, (vec(x_train)', vec(y_train)'), 100000)
# y_pred = Lux.cpu(Lux.apply(tstate.model, Lux.gpu(vec(x_train)'), tstate.parameters, tstate.states)[1])




# optfun = OptimizationFunction((x, p) -> loss_p(x), Optimization.AutoZygote())
# optprob = OptimizationProblem(optfun, res1_uode.u)

# res2_uode = Optimization.solve(optprob, ADAM(), maxiters=1000) # callback=callback

# plot(solutions[1], label="Actual")
# plot!(predict_p(res2_uode.u), label="Optimization")
# # scatter!(predict_p(res1_uode.u), label="Optimization1")

# optfun = OptimizationFunction((x, p) -> loss_p(x), Optimization.AutoZygote())
# optprob = OptimizationProblem(optfun, res2_uode.u)

# res3_uode = Optimization.solve(optprob, ADAM(0.0001), maxiters=1000) # callback=callback

# plot(solutions[1], label="Actual")
# plot!(predict_p(res2_uode.u), label="Optimization")
# plot!(predict_p(res3_uode.u), label="Optimization1")




final_params_w_lowest_val_loss1_wobhp = Dict()
max_val_loss = Inf
for i in p_list
    rng = Random.default_rng()
    Random.seed!(rng, Int(i))
    m_τ = Lux.Chain(
        Lux.Dense(1, j, swish),
        Lux.Dense(j, k, swish),
        Lux.Dense(k, l, swish),
        Lux.Dense(l, m, swish),
        Lux.Dense(m, n, swish),
        Lux.Dense(n, o, swish),
        Lux.Dense(o, q, swish),
        Lux.Dense(q, 1, relu))
    ps_τ, st_τ = Lux.setup(rng, m_τ)


    m_J = Lux.Chain(
        Lux.Dense(1, j, swish),
        Lux.Dense(j, k, swish),
        Lux.Dense(k, l, swish),
        Lux.Dense(l, m, swish),
        Lux.Dense(m, n, swish),
        Lux.Dense(n, o, swish),
        Lux.Dense(o, q, swish),
        Lux.Dense(q, 1, relu))
    ps_J, st_J = Lux.setup(rng, m_J)


    ps_F1 = opt_params[1][3]
    ps_F2 = opt_params[1][4]

    ps_τ = Lux.ComponentArray(ps_τ)
    ps_J = Lux.ComponentArray(ps_J)
    p = Lux.ComponentArray{eltype(ps_τ)}()
    p = Lux.ComponentArray(p;ps_τ)
    p = Lux.ComponentArray(p;ps_J)
    p = Lux.ComponentArray(p;ps_F1)
    p = Lux.ComponentArray(p;ps_F2)

    optfun = OptimizationFunction((x, p) -> loss_p(x), Optimization.AutoZygote())
    optprob = OptimizationProblem(optfun, p)

    res1_uode = Optimization.solve(optprob, ADAM(), maxiters=1) # callback=callback

    if loss_p(res1_uode.u)[1] != Inf
        val_prob = ODEProblem(parameterized_crm_bhp2!, predict_p(res1_uode.u)[end], (minimum(x_val), maximum(x_val)), res1_uode.u, saveat=x_val)
        pred = Array(solve(val_prob))
        if length(pred) == length(y_val)
            val_loss = sum(abs2, y_val .- pred)
            if val_loss < max_val_loss
                max_val_loss = val_loss
                final_params_w_lowest_val_loss1_wobhp[i] = (i, res1_uode.u, val_loss)
            end
        else
            val_loss = Inf
        end

    end
    # print(i)
    # print(loss_p(res1_uode.u)[1])
    # print(loss_p(p)[1])
end
    


[t[3] for t in values(final_params_w_lowest_val_loss1_wobhp)]


final_params_w_lowest_val_loss1_wobhp


# loss_p(final_params_w_lowest_val_loss1_wobhp[126][2])[1]
prob = ODEProblem(parameterized_crm_bhp2!, [u0s[1]], tspan, final_params_w_lowest_val_loss1_wobhp[11][2], saveat=timesteps)
soln = solve(prob)
plt11 = plot(grid=false, framestyle=:box, title="Producer 3", ylabel="Production Rates (bbl)")
plot!(plt11, solutions[1], label="Actual")
plot!(plt11, soln, label="Optimization", xlabel="Time (Days)")
vline!(plt11, [1600, 1800], linestyle=:dash, label=false, linecolor=:black)

p_list = []
j = 16
k = 32
l = 64
m = 128
n = 64
o = 32
q = 16
Threads.@threads for i in 1:1000
    rng = Random.default_rng()
    Random.seed!(rng, Int(i))

    m_τ = Lux.Chain(
        Lux.Dense(1, j, swish),
        Lux.Dense(j, k, swish),
        Lux.Dense(k, l, swish),
        Lux.Dense(l, m, swish),
        Lux.Dense(m, n, swish),
        Lux.Dense(n, o, swish),
        Lux.Dense(o, q, swish),
        Lux.Dense(q, 1, relu))
    ps_τ, st_τ = Lux.setup(rng, m_τ)
    
    
    m_J = Lux.Chain(
        Lux.Dense(1, j, swish),
        Lux.Dense(j, k, swish),
        Lux.Dense(k, l, swish),
        Lux.Dense(l, m, swish),
        Lux.Dense(m, n, swish),
        Lux.Dense(n, o, swish),
        Lux.Dense(o, q, swish),
        Lux.Dense(q, 1, relu))
    ps_J, st_J = Lux.setup(rng, m_J)
    

    ps_F1 = opt_params[1][3]
    ps_F2 = opt_params[1][4]

    ps_τ = Lux.ComponentArray(ps_τ)
    ps_J = Lux.ComponentArray(ps_J)
    p = Lux.ComponentArray{eltype(ps_τ)}()
    p = Lux.ComponentArray(p;ps_τ)
    p = Lux.ComponentArray(p;ps_J)
    p = Lux.ComponentArray(p;ps_F1)
    p = Lux.ComponentArray(p;ps_F2)

    if loss_p(p)[1] != Inf
        push!(p_list, i)
        # println(loss_p(p)[1])
    end

    if length(p_list) >= 100
        break
    end
end

    

dp1(t) = Lux.gradient(bhp1, t)[1]
dp2(t) = Lux.gradient(bhp2, t)[1]
dp3(t) = Lux.gradient(bhp3, t)[1]
dp4(t) = Lux.gradient(bhp4, t)[1]

function crm_bhp5!(du, u, p, t)
    τ, J, F1, F2 = p
    du[1] = (inj1(t)*F1 + inj2(t)*F2 - τ*J*dp1(t) .- u[1]) ./ τ
end

i = 1
j = 16
k = 32
l = 64
m = 128
n = 64
o = 32
q = 16

rng = Random.default_rng()
Random.seed!(rng, Int(i))

m_τ = Lux.Chain(
    Lux.Dense(1, j, swish),
    Lux.Dense(j, k, swish),
    Lux.Dense(k, l, swish),
    Lux.Dense(l, m, swish),
    Lux.Dense(m, n, swish),
    Lux.Dense(n, o, swish),
    Lux.Dense(o, q, swish),
    Lux.Dense(q, 1, relu))
ps_τ, st_τ = Lux.setup(rng, m_τ)


m_J = Lux.Chain(
    Lux.Dense(1, j, swish),
    Lux.Dense(j, k, swish),
    Lux.Dense(k, l, swish),
    Lux.Dense(l, m, swish),
    Lux.Dense(m, n, swish),
    Lux.Dense(n, o, swish),
    Lux.Dense(o, q, swish),
    Lux.Dense(q, 1, relu))
ps_J, st_J = Lux.setup(rng, m_J)

ps_F1 = opt_params[1][3]
ps_F2 = opt_params[1][4]

ps_τ = Lux.ComponentArray(ps_τ)
ps_J = Lux.ComponentArray(ps_J)
p = Lux.ComponentArray{eltype(ps_τ)}()
p = Lux.ComponentArray(p;ps_τ)
p = Lux.ComponentArray(p;ps_J)
p = Lux.ComponentArray(p;ps_F1)
p = Lux.ComponentArray(p;ps_F2)

function parameterized_crm_bhp2!(du, u, p, t)
    p_ = [m_τ([t], p.ps_τ, st_τ)[1][1], m_J([t], p.ps_J, st_J)[1][1], p.ps_F1, p.ps_F2]
    crm_bhp5!(du, u, p_, t)
end

(x_train, y_train), (x_val, y_val) = splitobs((timesteps, Array(solutions[4])); at=0.8, shuffle=false)

(x_val, y_val), (x_test, y_test) = splitobs((x_val, y_val); at=0.5, shuffle=false)

prob = ODEProblem(parameterized_crm_bhp2!, [u0s[1]], (minimum(x_train), maximum(x_train)), p, saveat=x_train)
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
callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses) % 100 == 0
        println(losses[end])
    end
    false
end

# p_list = []
# Threads.@threads for i in 1:10
#     j = 16
#     k = 32
#     l = 64
#     m = 128
#     n = 64
#     o = 32
#     q = 16

#     rng = Random.default_rng()
#     Random.seed!(rng, Int(i))

#     m_τ = Lux.Chain(
#         Lux.Dense(1, j, swish),
#         Lux.Dense(j, k, swish),
#         Lux.Dense(k, l, swish),
#         Lux.Dense(l, m, swish),
#         Lux.Dense(m, n, swish),
#         Lux.Dense(n, o, swish),
#         Lux.Dense(o, q, swish),
#         Lux.Dense(q, 1, relu))
#     ps_τ, st_τ = Lux.setup(rng, m_τ)


#     m_J = Lux.Chain(
#         Lux.Dense(1, j, swish),
#         Lux.Dense(j, k, swish),
#         Lux.Dense(k, l, swish),
#         Lux.Dense(l, m, swish),
#         Lux.Dense(m, n, swish),
#         Lux.Dense(n, o, swish),
#         Lux.Dense(o, q, swish),
#         Lux.Dense(q, 1, relu))
#     ps_J, st_J = Lux.setup(rng, m_J)

#     ps_F1 = opt_params[1][3]
#     ps_F2 = opt_params[1][4]

#     ps_τ = Lux.ComponentArray(ps_τ)
#     ps_J = Lux.ComponentArray(ps_J)
#     p = Lux.ComponentArray{eltype(ps_τ)}()
#     p = Lux.ComponentArray(p;ps_τ)
#     p = Lux.ComponentArray(p;ps_J)
#     p = Lux.ComponentArray(p;ps_F1)
#     p = Lux.ComponentArray(p;ps_F2)
#     if loss_p(p)[1] != Inf
#         push!(p_list, i)
#         println(loss_p(p)[1])
#     end
# end

plot(solutions[1])
plot!(predict_p(p))

optfun = OptimizationFunction((x, p) -> loss_p(x), Optimization.AutoZygote())
optprob = OptimizationProblem(optfun, p)

res1_uode = Optimization.solve(optprob, ADAM(), maxiters=1000, callback=callback) # callback=callback

plot(solutions[1], label="Actual")
plot!(predict_p(p), label="Initialized Parameters")
plot!(predict_p(res1_uode.u), label="Optimization")

# optfun = OptimizationFunction((x, p) -> loss_p(x), Optimization.AutoZygote())
# optprob = OptimizationProblem(optfun, res2_uode.u)

# res3_uode = Optimization.solve(optprob, BFGS(), maxiters=1000) # callback=callback

plot(solutions[1], label="Actual")
plot!(predict_p(res2_uode.u), label="Optimization")
plot!(predict_p(res3_uode.u), label="Optimization1")

param = final_params_w_lowest_val_loss1_wobhp[11][2]
# final_params_w_lowest_val_loss[35][2]

τ_NN(t) = m_τ([t], param.ps_τ, st_τ)[1][1]
J_NN(t) = m_J([t], final_params_w_lowest_val_loss[35][2].ps_p, st_J)[1][1]

plot(timesteps, τ_NN, label="Time Constant Optimization")
scatter!(timesteps, τ, label="Time Constant Actual")

plot(timesteps, J_NN, label="Productivity Index Optimization")
scatter!(timesteps, J, label="Productivity Index Actual")

BHP(t) = Lux.gradient(bhp1, t)[1]
bhp_nn(t) = m_p([t], param.ps_p, st_p)[1][1]
plot(timesteps, BHP)
plot!(timesteps, bhp_nn)



