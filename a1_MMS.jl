using DifferentialEquations, LinearAlgebra
using GLMakie, CairoMakie
using BSON: @save, @load

# ---------------- CONFIGURATION ----------------
save_data    = true
analyze      = true
save_figures = false

datafile = "data/duffing_slowflow.bson"
figfile  = "figures/duffing_slowflow.pdf"

save_figures ? CairoMakie.activate!() : GLMakie.activate!()

# ---------------- PARAMETERS ----------------
a, b, c = 0.01, 1.0, 0.01
F, w = 0.05, 1.01
p = [a, b, c, F, w]
u0_duffing = [0.01, 0.0]
s0_slowflow = [0.01, 0.0]
tspan = (0.0, 500.0)
dt = 0.01
tpoints = collect(tspan[1]:dt:tspan[2])

# ---------------- SYSTEM DEFINITIONS ----------------
function slowflow!(du, u, p, t)
    A, B = u
    a, b, c, F, w = p
    du[1] = -a*A - (F / (2 * sqrt(b))) * sin(B)
    du[2] = (3c*A^2 / (8 * sqrt(b))) - (w - sqrt(b)) - (F / (2 * sqrt(b) * A)) * cos(B)
end

function duffing!(du, u, p, t)
    x, dx = u
    a, b, c, F, w = p
    du[1] = dx
    du[2] = -(2a*dx + b*x + c*x^3 - F*cos(w*t))
end

# ---------------- SOLVE OR LOAD ----------------
if analyze
    prob_slow = ODEProblem(slowflow!, s0_slowflow, tspan, p)
    sol_slow  = solve(prob_slow, Tsit5(), saveat=tpoints)

    prob_duff = ODEProblem(duffing!, u0_duffing, tspan, p)
    sol_duff  = solve(prob_duff, Tsit5(), saveat=tpoints)

    if save_data
        @save datafile tpoints sol_slow sol_duff
    end
else
    @load datafile tpoints sol_slow sol_duff
end

A_t = [u[1] for u in sol_slow.u]
x_t = [u[1] for u in sol_duff.u]

# ---------------- PLOTTING ----------------
fig = Figure(resolution = (900, 400))
ax = Axis(fig[1, 1], xlabel="Time (s)", ylabel="Amplitude", title="Duffing vs Slow Flow Amplitude Comparison")

lines!(ax, tpoints, A_t, label = "Slow Flow A(t)", linewidth=2, color=:blue)
lines!(ax, tpoints, x_t, label = "Duffing x(t)", linewidth=2, linestyle = :dash, color=:red)

axislegend(ax, position=:rb)

if save_figures
    save(figfile, fig)
end

display(fig)
