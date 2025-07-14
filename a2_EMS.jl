using DifferentialEquations, LinearAlgebra, NonlinearSolve
using GLMakie, CairoMakie
using BSON: @save, @load

# ---------------- CONFIG ----------------
save_figures = false
save_data    = true
analyze      = true

datafile = "data/homotopy_solution.bson"
figfile  = "figures/homotopy_solution.pdf"

save_figures ? CairoMakie.activate!() : GLMakie.activate!()

# ---------------- SYSTEM DEFINITIONS ----------------
function homotopy_ode!(du, u, p, t)
    x, dx = u
    mu, W, w, a, F = p
    du[1] = dx
    du[2] = -(2*mu*dx + w^2*x + a*x^3 - F*cos(W*t))
end

function homotopy_slow!(du, u, p, t=nothing)
    A, B = u
    mu, W, w, a, F = p
    du[1] = -mu*A - (F / (2*W)) * sin(B)
    du[2] = (3*a*A^2)/(8*W) + ((w^2 - W^2)/(2*W)) - (F * cos(B)) / (2*A*W)
end

function slowflow!(du, u, p, t=nothing)
    A, B = u
    mu, W, w, a, F = p
    du[1] = -mu*A - (F / (2*w)) * sin(B)
    du[2] = (3*a*A^2)/(8*w) - (W - w) - (F * cos(B)) / (2*w*A)
end

# ---------------- PARAMETERS ----------------
mu, W, w, a, F = 0.01, 1.5, 2.0, 0.01, 0.05
p = [mu, W, w, a, F]
tspan = (0.0, 600.0)
dt = 0.01
tpoints = collect(tspan[1]:dt:tspan[2])
u0 = [0.01, 0.0]

# ---------------- SOLVE or LOAD ----------------
if analyze
    # Time-domain ODE
    prob_ode   = ODEProblem(homotopy_ode!, u0, tspan, p)
    sol_ode    = solve(prob_ode, Tsit5(), saveat=tpoints)

    # Homotopy slow flow
    prob_slow1 = ODEProblem(homotopy_slow!, u0, tspan, p)
    sol_slow1  = solve(prob_slow1, Tsit5(), saveat=tpoints)

    # MMS-style slow flow
    prob_slow2 = ODEProblem(slowflow!, u0, tspan, p)
    sol_slow2  = solve(prob_slow2, Tsit5(), saveat=tpoints)

    # Steady-state EMS and MMS
    sol_ss1 = solve(SteadyStateProblem(homotopy_slow!, [0.01, 1.0], p), SSRootfind())
    sol_ss2 = solve(SteadyStateProblem(slowflow!, sol_ss1.u, p), SSRootfind())

    if save_data
        @save datafile tpoints sol_ode sol_slow1 sol_slow2 sol_ss1 sol_ss2
    end
else
    @load datafile tpoints sol_ode sol_slow1 sol_slow2 sol_ss1 sol_ss2
end

x_ode = [u[1] for u in sol_ode.u]

# ---------------- PLOTTING ----------------
fig = Figure(resolution=(1000, 500))
ax = Axis(fig[1, 1], xlabel="Time (s)", ylabel="x(t)", title="Homotopy ODE vs Slow Flow")

lines!(ax, tpoints, x_ode, label="ODE", linewidth=2, color=:black)

# Steady-state overlays
lines!(ax, [tpoints[1], tpoints[end]], fill(sol_ss1.u[1], 2), label="EMS-SS", color=:red, linestyle=:dash, linewidth=2)
lines!(ax, [tpoints[1], tpoints[end]], fill(sol_ss2.u[1], 2), label="MMS-SS", color=:blue, linestyle=:dot, linewidth=2)

axislegend(ax, position=:rb)

if save_figures
    save(figfile, fig)
end

display(fig)
