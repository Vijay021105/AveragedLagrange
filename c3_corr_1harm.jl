using DifferentialEquations
using NonlinearSolve
using LinearAlgebra
using GLMakie, CairoMakie
using BSON: @save, @load

# ---------------- CONFIG ----------------
save_figures = true
save_data = true
analyze = true

save_figures ? CairoMakie.activate!() : GLMakie.activate!()

# ---------------- PARAMETERS ----------------
m = 1.0
c = 0.25e-2
k = 4.0
kt = 5.0
fs = 1.0
Ωrange = 0.25:0.05:5.0
F_values = [0.1, 0.25, 0.5, 1.0]
tspan = (0.0, 100.0)
dt = 0.01
tpoints = tspan[1]:dt:tspan[2]
u0 = [0.01, 0.0, 0.0]
v0 = [0.01, 0.0]
Om = 2.0

# ---------------- MODELS ----------------
function jerkins!(du, u, p, t)
    m, c, k, kt, F, fs, Om = p
    du[1] = u[2]
    du[2] = -(c*u[2] + k*u[1] + kt*(u[1] - u[3]) - F*cos(Om*t)) / m
    du[3] = (kt*(u[1] - u[3]) * sign(u[2])) <= fs ? 0.0 : u[2]
end

function safe_acos(x)
    abs(x) < 1 ? acos(x) : π
end

function avglag!(du, u, p, t)
    m, c, k, kt, F, fs, Om = p
    Uc, Us = u
    τst = safe_acos(1 - (2 * fs / (kt * sqrt(Uc^2 + Us^2 + 1e-12))))
    du[1] = -(((Us*kt*sin(2*τst))/(4*π) - (Uc*kt*cos(2*τst))/(4*π) - (Us*kt*τst)/(2*π) +
              (Us*m*Om^2)/2 + (Uc*c*Om)/2 + (Uc*kt)/(4*π) - (Us*k)/2) / (m*Om))
    du[2] = ((Uc*kt*sin(2*τst))/(4*π) + (Us*kt*cos(2*τst))/(4*π) - (Uc*kt*τst)/(2*π) +
             (Uc*m*Om^2)/2 - (Us*c*Om)/2 - (Us*kt)/(4*π) - (Uc*k)/2 + F/2) / (m*Om)
end

# ---------------- SIMULATION ----------------
correlation_all = Vector{Vector{Float64}}(undef, length(F_values))
amplitude_all = Vector{Vector{Float64}}(undef, length(F_values))
if analyze
for (fi, F) in enumerate(F_values)
    correlation = zeros(length(Ωrange))
    amplitudes = zeros(length(Ωrange))

    for (i, W) in enumerate(Ωrange)
        p = [m, c, k, kt, F, fs, W]

        sol_direct = solve(ODEProblem(jerkins!, u0, tspan, p), Tsit5(), saveat=dt, reltol=1e-8, abstol=1e-12)
        sol_lag = solve(ODEProblem(avglag!, v0, tspan, p), Tsit5(), saveat=dt, reltol=1e-8, abstol=1e-12)
        sol_steady = solve(SteadyStateProblem(avglag!, [0.01, 0.0], p), SSRootfind())

        x_direct = [u[1] for u in sol_direct.u]
        x_lag = [u[1]*cos(Om*t) + u[2]*sin(Om*t) for (u, t) in zip(sol_lag.u, sol_lag.t)]
        x_ss = [sol_steady.u[1]*cos(Om*t) + sol_steady.u[2]*sin(Om*t) for t in sol_lag.t]

        Δd = x_direct[1:length(x_ss)] .- x_ss
        Δl = x_lag .- x_ss

        correlation[i] = abs(dot(Δd, Δl) / (norm(Δd)*norm(Δl)))
        amplitudes[i] = sqrt(sol_steady.u[1]^2 + sol_steady.u[2]^2)
    end

    correlation_all[fi] = correlation
    amplitude_all[fi] = amplitudes

    if save_data
        @save "data/c3_corr_force_$(F).bson" correlation amplitudes
    end
end
else 
    @load "data/c3_corr_force_$(F).bson" correlation amplitudes
end

# ---------------- PLOTTING ----------------
fig = Figure(resolution = (1200, 1000))
for (fi, F) in enumerate(F_values)
    row = div(fi-1, 2)+1
    col = mod(fi-1, 2)+1

    ax1 = Axis(fig[row, col], xlabel="Ω (rad/s)", ylabel="Correlation Coefficient", title="F = $(F)", ylabelcolor=:blue)
    ax2 = Axis(fig[row, col], yaxisposition=:right, ylabel="FRF Amplitude", ylabelcolor=:red, yscale=log10)

    lines!(ax1, Ωrange, correlation_all[fi], color = :blue, label = "Correlation")
    lines!(ax2, Ωrange, amplitude_all[fi], color = :red, linestyle = :dash, label = "FRF")

    ylims!(ax1, -0.1, 1.05)
    axislegend(ax1, position=:rb)
end

save_figures && save("figures/c3_corr_E0.pdf", fig)
display(fig)
