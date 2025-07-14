using DifferentialEquations, LinearAlgebra
using GLMakie, CairoMakie
using BSON: @save, @load

# -------------------- CONFIG --------------------
save_data    = true
analyze      = true    # Set to true for re-run
save_figures = false   # Use GLMakie if false, CairoMakie for vector export

datafile = "data/a0_duffing.bson"
figfile  = "figures/a0_duffing.pdf"

save_figures ? CairoMakie.activate!() : GLMakie.activate!()

# -------------------- SYSTEM PARAMETERS --------------------
a, b, c = 0.01, 1.0, 0.01
F, ω    = 0.05, 1.01
p       = [a, b, c, F, ω]

u0      = [0.01, 0.0]
tspan   = (0.0, 500.0)
tpoints = collect(tspan[1]:0.01:tspan[2])

# -------------------- ODE SYSTEM --------------------
function duffing!(du, u, p, t)
    x, dx = u
    a, b, c, F, ω = p
    du[1] = dx
    du[2] = -(2a * dx + b * x + c * x^3 - F * cos(ω * t))
end

# -------------------- SOLVE OR LOAD --------------------
if analyze
    prob = ODEProblem(duffing!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=tpoints)

    if save_data
        @save datafile tpoints sol
    end
else
    @load datafile tpoints sol
end

# -------------------- PLOT --------------------
fig = Figure(resolution = (900, 400))
ax = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Displacement (x)",
          title = "Duffing Oscillator Response")

lines!(ax, sol.t, [u[1] for u in sol.u], color = :navy, linewidth = 2)
axislegend(ax, position = :rb)

if save_figures
    save(figfile, fig)
end

display(fig)
