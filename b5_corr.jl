using DifferentialEquations, GLMakie, LinearAlgebra, CairoMakie, Printf,NonlinearSolve
using BSON: @save, @load

# ---------------------- CONFIGURATION ----------------------
save_figures = false  # Set to true to export the plot
save_data = true
analyze = true
save_figures ? CairoMakie.activate!() : GLMakie.activate!()


# ---------------------- PARAMETERS ----------------------
m, c, k = 1.0, 0.25, 4.0     # Mass, damping, stiffness
F = 0.5                      # Forcing amplitude
ω_range = 0.1:0.01:4.0      # Excitation frequency range (rad/s)
tspan = (0.0, 100.0)
dt=0.01
tsteps = tspan[1]:dt:tspan[2]
u0_al = [0.01,0.0]
u0_num = [0.01,0.0]

# ---------------------- SYSTEM DEFINITION ----------------------

# Main ODE
function linearODE!(du, u, p, t)
    x, dx = u
    m, c, k, F, ω = p
    du[1] = dx
    du[2] = -(c*dx + k*x - F*cos(ω*t)) / m
end

# Linearized averaged system (frequency domain)
function linearized!(du, u, p, t)
    A, B = u
    m, c, k, F, ω = p
    du[1] = ((B * (k - m*ω^2) + A*c*ω) - F) / (m*ω)
    du[2] = -((A * (k - m*ω^2) - B*c*ω)) / (m*ω)
end

# Steady-state analytical solution
function steady_state(t, p)
    m, c, k, F, ω = p
    denom = sqrt((k - m*ω^2)^2 + (c*ω)^2)
    ϕ = atan(c*ω, k - m*ω^2)
    return (F / denom) * cos(ω * t - ϕ)
end

# ---------------------- SWEEP LOOP ----------------------
correlations = zeros(length(ω_range))
FRF = zeros(length(ω_range))
for (i,ω) in enumerate(ω_range)
    p = (m, c, k, F, ω)

    sol_main = solve(ODEProblem(linearODE!, u0_num, tspan, p), Tsit5(), saveat=0.01)
    sol_lin  = solve(ODEProblem(linearized!, u0_al, tspan, p), Tsit5(), saveat=0.01)
    sol_ss = solve(SteadyStateProblem(linearized!,u0_al,p),SSRootfind())
    A_ss, B_ss = sol_ss.u
    x_main = [u[1] for u in sol_main.u]
    x_ss = [A_ss * cos(ω*t) + B_ss * sin(ω*t) for t in sol_lin.t]
    x_lin = [sol_lin.u[i][1] * cos(ω*sol_lin.t[i]) + sol_lin.u[i][2]*sin(ω*sol_lin.t[i]) for t in eachindex(sol_lin.t)]
    err_main = x_main - x_ss
    err_lin = x_lin - x_ss  # Replace with same → correlation = 1 always for sanity
    correlations[i] = abs(dot(err_lin, err_main) / (norm(err_lin)*norm(err_main)))  

    FRF[i] = sqrt(sol_lin.u[end][1]^2+sol_lin.u[end][2]^2)
end
if analyze
    println(correlations)
    if save_data
        @save "data/a1_corr_linear" correlations FRF
    end
else
    @load "data/a1_corr_linear" correlations FRF
end

# ---------------------- DUAL Y-AXIS PLOT ----------------------
fig = Figure(resolution=(1100, 600))
ax1 = Axis(fig[1, 1], xlabel="Excitation Frequency ω (rad/s)", ylabel="Correlation Coefficient",
    title="Transient Accuracy & FRF vs Frequency")

ax2 = Axis(fig[1, 1], ylabel="FRF Amplitude", yaxisposition=:right, yscale=log10)

# Plot correlation on left y-axis
lines!(ax1, ω_range, correlations, color=:blue, label="Correlation", linewidth=2)

# Plot FRF on right y-axis (log scale for better contrast)
lines!(ax2, ω_range, FRF, color=:orange, label="FRF Amplitude", linewidth=2, linestyle=:dash)
ylims!(ax1,-0.1,1.1)

axislegend(ax1, position=:rt)


# Show or save
if save_figures
    save(fig_path, fig)
end

display(fig)