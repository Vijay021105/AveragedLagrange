using DifferentialEquations, LinearAlgebra
using NonlinearSolve
using GLMakie, CairoMakie
using BSON: @save, @load

# ---------------------- CONFIGURATION ----------------------
save_figures = false
save_data     = true
analyze       = true
save_figures ? CairoMakie.activate!() : GLMakie.activate!()


# ---------------------- SYSTEM DEFINITIONS ----------------------
function MMS!(du, u, p, t)
    A, B = u
    mu, W, w, a, F = p
    du[1] = -mu * A - (F / (2 * w)) * sin(B)
    du[2] = (3 * a * A^2) / (8 * w) - (W - w) - (F / (2 * w * A)) * cos(B)
end

function EMS!(du, u, p, t)
    A, B = u
    mu, W, w, a, F = p
    du[1] = -mu * A - (F / (2 * W)) * sin(B)
    du[2] = (3 * a * A^2) / (8 * W) + ((w^2 - W^2) / (2 * W)) - (F * cos(B)) / (2 * A * W)
end

# ---------------------- PARAMETERS ----------------------
mu = 0.01
w = 2.0
a = 0.225
F = 0.1

Ws_forward  = collect(1.25:0.025:3.25)
Ws_backward = collect(3.25:-0.025:1.25)

function solve_system(Ws)
    x_mms = Float64[]
    x_ems = Float64[]
    s0 = [0.0, 0.0]
    sol_prev_mms = zeros(2)

    for (i, W) in enumerate(Ws)
        p = [mu, W, w, a, F]
        if i == 1
            linear_response = F / ((w^2 - W^2) + 2 * mu * W * im)
            s0[1] = abs(linear_response)
            s0[2] = angle(linear_response)
        else
            s0 .= sol_prev_mms
        end

        prob_mms = SteadyStateProblem(MMS!, s0, p)
        prob_ems = SteadyStateProblem(EMS!, s0, p)
        sol_mms = solve(prob_mms, SSRootfind())
        sol_ems = solve(prob_ems, SSRootfind())
        
        sol_prev_mms .= sol_mms.u
        push!(x_mms, abs(sol_mms.u[1]))
        push!(x_ems, abs(sol_ems.u[1]))
    end
    return x_mms, x_ems
end

# ---------------------- COMPUTE / LOAD ----------------------
if analyze
    x_mms_forward,  x_ems_forward  = solve_system(Ws_forward)
    x_mms_backward, x_ems_backward = solve_system(Ws_backward)

    if save_data
        @save "data/a1_FR_mmsems.bson" x_mms_forward x_ems_forward x_mms_backward x_ems_backward Ws_forward Ws_backward
    end
else
    @load "data/a1_FR_mmsems.bson" x_mms_forward x_ems_forward x_mms_backward x_ems_backward Ws_forward Ws_backward
end

# ---------------------- PLOTTING ----------------------
fig = Figure(resolution=(1000, 500))
ax = Axis(fig[1, 1],
    xlabel = "Excitation Frequency W (rad/s)",
    ylabel = "Amplitude (m)",
    title  = "MMS vs EMS Forward & Backward Sweep")

# Forward
scatter!(ax, Ws_forward,  x_mms_forward,  color = :blue,  marker = :circle, markersize=6, label = "MMS Forward")
scatter!(ax, Ws_forward,  x_ems_forward,  color = :red,   marker = :circle, markersize=6, label = "EMS Forward")

# Backward
scatter!(ax, Ws_backward, x_mms_backward, color = :blue,  marker = :utriangle, markersize=6, label = "MMS Backward")
scatter!(ax, Ws_backward, x_ems_backward, color = :red,   marker = :utriangle, markersize=6, label = "EMS Backward")

axislegend(ax, position = :rb)
fig[1, 1] = ax

if save_figures
    save("figures/a1_FR_mmsems.pdf", fig)
end

display(fig)
