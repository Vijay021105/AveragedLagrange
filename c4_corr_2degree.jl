using DifferentialEquations, NonlinearSolve, LinearAlgebra
using GLMakie, CairoMakie
using BSON: @save, @load

# ---------------- CONFIGURATION ----------------
save_figures = false
save_data = true
analyze = false

save_figures ? CairoMakie.activate!() : GLMakie.activate!()

# ---------------- PARAMETERS ----------------
m, c, k = 1.0, 0.0025, 4.0
kt, fs = 5.0, 1.0
Ωrange = 0.25:0.01:5.0
F_values = [0.1, 0.25, 0.5, 1.0]
tspan = (0.0, 50*sqrt(k/m))
u0 = [0.01, 0.0, 0.0]
v00 = [0.01, 0.0, 0.0, 0.0]

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

function avglagcouple!(du, u, p, t)
    m, c, k, kt, F, fs, Om = p
    Uc, Us, dUc, dUs = u
    τst = safe_acos(1 - (2 * fs / (kt * sqrt(Uc^2 + Us^2))))
    du[1] = dUc
    du[2] = dUs
    du[3] = (0.5*Uc*kt*sin(2*τst))/(π*m)+(0.5*Us*kt*cos(2*τst))/(π*m)-(Uc*kt*τst)/(π*m)+Uc*Om^2-(Us*c*Om)/m-2.0*dUs*Om-(0.5*Us*kt)/(π*m)-(Uc*k)/m-(dUc*c)/m+(F)/m
    du[4] = (0.5*Us*kt*sin(2*τst))/(π*m)-(0.5*Uc*kt*cos(2*τst))/(π*m)-(Us*kt*τst)/(π*m)+Us*Om^2+(Uc*c*Om)/m+2.0*dUc*Om+(0.5*Uc*kt)/(π*m)-(Us*k)/m-(dUs*c)/m
end

# ---------------- SIMULATION ----------------
correlation_all = Vector{Vector{Float64}}(undef, length(F_values))
amplitude_all = Vector{Vector{Float64}}(undef, length(F_values))

if analyze
    for (idx, F) in enumerate(F_values)
        correlation = zeros(length(Ωrange))
        amplitude = zeros(length(Ωrange))

        for (i, W) in enumerate(Ωrange)
            p = [m, c, k, kt, F, fs, W]

            sol = solve(ODEProblem(jerkins!, u0, tspan, p), Vern9(), reltol=1e-8, abstol=1e-12, saveat=0.01)
            sol2 = solve(ODEProblem(avglagcouple!, v00, tspan, p), Vern9(), reltol=1e-8, abstol=1e-12, saveat=0.01)
            sol3 = solve(SteadyStateProblem(avglagcouple!, v00, p), SSRootfind())

            t_common = sol2.t
            x_direct = [u[1] for u in sol.u][1:length(t_common)]
            x_lag = [sol2.u[j][1]*cos(W*t_common[j]) + sol2.u[j][2]*sin(W*t_common[j]) for j in eachindex(t_common)]
            x_ss = [sol3.u[1]*cos(W*t) + sol3.u[2]*sin(W*t) for t in t_common]

            Δd = x_direct .- x_ss
            Δl = x_lag .- x_ss
            correlation[i] = abs(dot(Δd, Δl) / (norm(Δd)*norm(Δl)))
            amplitude[i] = sqrt(sol3.u[1]^2 + sol3.u[2]^2)
        end

        correlation_all[idx] = correlation
        amplitude_all[idx] = amplitude

        if save_data
            @save "data/c4_corr_FR_coupled_$(F).bson" correlation amplitude
        end
    end
else
    for (idx, F) in enumerate(F_values)
        @load "data/c4_corr_FR_coupled_$(F).bson" correlation amplitude
        correlation_all[idx] = correlation
        amplitude_all[idx] = amplitude
    end
end

# ---------------- PLOTTING ----------------
fig = Figure(resolution = (1200, 900))
for (idx, F) in enumerate(F_values)
    row = div(idx-1, 2) + 1
    col = mod(idx-1, 2) + 1

    # Create axes
    ax1 = Axis(fig[row, col], xlabel="Ω (rad/s)", ylabel="Correlation", title="F = $(F)", ylabelcolor=:blue)
    ax2 = Axis(fig[row, col], yaxisposition=:right, ylabel="FRF Amplitude", ylabelcolor=:red, yscale=log10)

    # Plot
    lines!(ax1, Ωrange, correlation_all[idx], color=:blue, linewidth=2, label="Correlation")
    lines!(ax2, Ωrange, amplitude_all[idx], color=:red, linestyle=:dash, linewidth=2, label="FRF Amplitude")

    ylims!(ax1, -0.1, 1.05)
end

save_figures && save("figures/c4_corr_fr_couple.pdf", fig)
display(fig)
