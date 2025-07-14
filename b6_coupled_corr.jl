using DifferentialEquations, LinearAlgebra, CairoMakie, GLMakie, Printf
using BSON: @save, @load

# ---------------------- CONFIGURATION ----------------------
save_figures = true
save_data = true
analyze = true
save_figures ? CairoMakie.activate!() : GLMakie.activate!()

# ---------------------- PARAMETERS ----------------------
m, k = 1.0, 4.0               # Mass, stiffness
F = 0.5                      # Forcing amplitude
c_values = [0.0025, 0.25]    # Light and heavy damping
ω_range = 0.1:0.01:4.0       # Frequency sweep
tspan = (0.0, 100.0)
dt = 0.1
tpoints = tspan[1]:dt:tspan[2]
u0_direct = [0.1, 0.0]
u0_lin = [0.1, 0.0, 0.0, 0.0]

# ---------------------- SYSTEM DEFINITIONS ----------------------
function linearODE!(du, u, p, t)
    x, dx = u
    m, c, k, F, ω = p
    du[1] = dx
    du[2] = -(c*dx + k*x - F*cos(ω*t)) / m
end

function linearized!(du, u, p, t)
    A, B, dA, dB = u
    m, c, k, F, ω = p
    du[1] = dA
    du[2] = dB
    du[3] = A*ω^2 - (B*c*ω)/m - 2*dB*ω - (A*k)/m - (dA*c)/m + F/m
    du[4] = (B*m*ω^2 + (2*dA*m + A*c)*ω - B*k - dB*c)/m
end

function ez_linear!(du, u, p, t)
    A, B = u
    m, c, k, F, ω = p
    du[1] = -((B*ω)/2) + (B*k)/(2*m*ω) - (A*c)/(2*m)
    du[2] = (A*m*ω^2 - B*c*ω - A*k + F)/(2*m*ω)
end

function steady_state(t, p)
    m, c, k, F, ω = p
    sol_steady = solve(SteadyStateProblem(ez_linear!,u0_direct,p),SSRootfind())
    return sol_steady.u[1]*cos(ω * t) + sol_steady.u[2]*sin(ω * t)
end

# ---------------------- SIMULATION + ANALYSIS ----------------------
correlations_all = Matrix{Vector{Float64}}(undef, 2, 2)  # [ci, lin/ez]
amplitudes_all = Vector{Vector{Float64}}(undef, 2)

for (ci, c) in enumerate(c_values)
    correlations_lin = zeros(length(ω_range))
    correlations_ez = zeros(length(ω_range))
    amplitudes = zeros(length(ω_range))

    for (i, ω) in enumerate(ω_range)
        p = (m, c, k, F, ω)

        prob_direct = ODEProblem(linearODE!, u0_direct, tspan, p)
        prob_lin = ODEProblem(linearized!, u0_lin, tspan, p)
        prob_ez = ODEProblem(ez_linear!, u0_lin[1:2], tspan, p)

        sol_direct = solve(prob_direct, Tsit5(), saveat=tpoints)
        sol_lin = solve(prob_lin, Tsit5(), saveat=tpoints)
        sol_ez = solve(prob_ez, Tsit5(), saveat=tpoints)

        x_direct = [u[1] for u in sol_direct.u]
        x_lin = [A*cos(ω*t) + B*sin(ω*t) for (t, (A, B, _, _)) in zip(sol_lin.t, sol_lin.u)]
        x_ez = [A*cos(ω*t) + B*sin(ω*t) for (t, (A, B)) in zip(sol_ez.t, sol_ez.u)]
        x_ss = [steady_state(t, p) for t in tpoints]

        Δ_direct = x_direct .- x_ss
        Δ_lin = x_lin .- x_ss
        Δ_ez = x_ez .- x_ss

        correlations_lin[i] = abs(dot(Δ_direct, Δ_lin) / (norm(Δ_direct)*norm(Δ_lin)))
        correlations_ez[i] = abs(dot(Δ_direct, Δ_ez) / (norm(Δ_direct)*norm(Δ_ez)))
        amplitudes[i] = F / sqrt((k - m*ω^2)^2 + (c*ω)^2)
    end

    correlations_all[ci, 1] = correlations_lin
    correlations_all[ci, 2] = correlations_ez
    amplitudes_all[ci] = amplitudes

    if analyze && save_data
        @save "data/b6_corr_c=$(c).bson" correlations_lin correlations_ez amplitudes
    elseif !analyze
        @load "data/b6_corr_c=$(c).bson" correlations_lin correlations_ez amplitudes
    end
end

# ---------------------- FINAL PLOTTING ----------------------
fig = Figure(resolution = (1100, 500))
for (ci, c) in enumerate(c_values)
    ax1 = Axis(fig[1, ci],
        title = "c = $(c)",
        xlabel = "Excitation Frequency ω (rad/s)",
        ylabel = "Correlation Coefficient",
        ylabelcolor = :blue)

    ax2 = Axis(fig[1, ci],
        yaxisposition = :right,
        ylabel = "FRF Amplitude (m)",
        ylabelcolor = :red,
        yscale = log10)

#    lines!(ax1, ω_range, correlations_all[ci, 1], color = :blue, label = "Coupled Avg Lag")
    lines!(ax1, ω_range, correlations_all[ci, 2], color = :green, linestyle = :dash, label = "Avg Lag without ̈A and ̈B")
    lines!(ax2, ω_range, amplitudes_all[ci], color = :red, linewidth = 2, label = "FRF")

    vlines!(ax1, [sqrt(k/m)], color = :gray, linestyle = :dot)
    text!(ax1, "ωₙ ≈ $(round(sqrt(k/m), digits=2))", position = (sqrt(k/m)+0.1, 1.05))

    ylims!(ax1, -0.1, 1.1)
    ylims!(ax2, (0.01, 150))
    axislegend(ax1, position = :rt)
end
for (ci, c) in enumerate(c_values)
    ax1 = Axis(fig[2, ci],
        title = "c = $(c)",
        xlabel = "Excitation Frequency ω (rad/s)",
        ylabel = "Correlation Coefficient",
        ylabelcolor = :blue)

    ax2 = Axis(fig[2, ci],
        yaxisposition = :right,
        ylabel = "FRF Amplitude (m)",
        ylabelcolor = :red,
        yscale = log10)

    lines!(ax1, ω_range, correlations_all[ci, 1], color = :blue, label = "Coupled Avg Lag")
#    lines!(ax1, ω_range, correlations_all[ci, 2], color = :green, linestyle = :dash, label = "Avg Lag without ̈A and ̈B")
    lines!(ax2, ω_range, amplitudes_all[ci], color = :red, linewidth = 2, label = "FRF")

    vlines!(ax1, [sqrt(k/m)], color = :gray, linestyle = :dot)
    text!(ax1, "ωₙ ≈ $(round(sqrt(k/m), digits=2))", position = (sqrt(k/m)+0.1, 1.05))

    ylims!(ax1, -0.1, 1.1)
    ylims!(ax2, (0.01, 150))
    axislegend(ax1, position = :rt)
end 

save_figures && save("figures/b6_corr_all.pdf", fig)
display(fig)
