using DifferentialEquations, GLMakie, LinearAlgebra, CairoMakie, Printf, NonlinearSolve
using BSON: @save, @load

# ---------------------- CONFIGURATION ----------------------
save_figures = true
save_data    = true
analyze      = true  # Set false to load saved data

save_figures ? CairoMakie.activate!() : GLMakie.activate!()

# ---------------------- PARAMETERS ----------------------
m, c, k = 1.0, 0.25, 4.0
F, a = 0.5, 0.0
w0 = sqrt(k/m)

ω_range = 0.25:0.01:4.0
tspan = (0.0, 100.0)
dt = 0.01
tsteps = tspan[1]:dt:tspan[2]

u0_num = [0.01, 0.0]
u0_al  = [0.01, 0.0]

# ---------------------- SYSTEM DEFINITIONS ----------------------

function linearODE!(du, u, p, t)
    x, dx = u
    m, c, k, F, ω = p
    du[1] = dx
    du[2] = -(c*dx + k*x - F*cos(ω*t)) / m
end

function linearized!(du, u, p, t)
    A, B = u
    m, c, k, F, Ω, a, w0 = p
    denom = 4π*m*(Ω^2 - w0^2)

    du[1] = -(
        (2*F*w0*cos(π*Ω/w0)) / denom +
        (2π*A*c*Ω^2 - 2π*A*c*w0^2) / denom -
        (2*F*w0) / denom +
        (B*w0)/2 - (B*k)/(2*m*w0) -
        (3*B^3*a)/(8*m*w0) - (3*A^2*B*a)/(8*m*w0)
    )

    du[2] = (
        (2*F*Ω*sin(π*Ω/w0)) / denom -
        (2π*B*c*Ω^2 - 2π*B*c*w0^2) / denom +
        (A*w0)/2 - (A*k)/(2*m*w0) -
        (3*A*B^2*a)/(8*m*w0) - (3*A^3*a)/(8*m*w0)
    )
end


# ---------------------- SWEEP LOOP ----------------------
correlations = zeros(length(ω_range))
FRF = zeros(length(ω_range))

for (i, ω) in enumerate(ω_range)
    if abs(ω - w0) < 1e-5  # 🛑 Skip resonance point
        println("Skipping ω = $ω ≈ √(k/m) = $w0 to avoid division by zero.")
        correlations[i] = NaN
        FRF[i] = NaN
        continue
    end

    p_ode = (m, c, k, F, ω)
    p_avg = (m, c, k, F, ω, a, w0)

    sol_main = solve(ODEProblem(linearODE!, u0_num, tspan, p_ode), Tsit5(), saveat=dt)
    sol_lin  = solve(ODEProblem(linearized!, u0_al, tspan, p_avg), Tsit5(), saveat=dt)

    A_ss, B_ss = sol_lin[end][1], sol_lin[end][2]
    x_main = [u[1] for u in sol_main.u]
    x_ss   = [A_ss * cos(w0*t) + B_ss * sin(w0*t) for t in sol_lin.t]
    x_lin  = [sol_lin.u[i][1] * cos(w0*t) + sol_lin.u[i][2] * sin(w0*t) for (i, t) in enumerate(sol_lin.t)]

    err_main = x_main .- x_ss
    err_lin  = x_lin  .- x_ss

    correlations[i] = abs(dot(err_lin, err_main) / (norm(err_lin) * norm(err_main)))
    FRF[i] = sqrt(A_ss^2 + B_ss^2)
end

if analyze
    println("Correlation coefficients: ", correlations)
    if save_data
        @save "data/b8_corr_linear.bson" correlations FRF
    end
else
    @load "data/b8_corr_linear.bson" correlations FRF
end

# ---------------------- DUAL Y-AXIS PLOT ----------------------
fig = Figure(resolution=(1100, 600))
ax1 = Axis(fig[1, 1],
    xlabel = "Excitation Frequency ω (rad/s)",
    ylabel = "Correlation Coefficient",
    title  = "Transient Accuracy & FRF vs Frequency"
)
ax2 = Axis(fig[1, 1],
    ylabel = "FRF Amplitude", yaxisposition = :right, yscale = log10)

lines!(ax1, ω_range, correlations, color = :blue, label = "Correlation", linewidth = 2)
lines!(ax2, ω_range, FRF, color = :orange, linestyle = :dash, label = "FRF Amplitude", linewidth = 2)
ylims!(ax1, -0.1, 1.1)
axislegend(ax1, position = :rt)


if save_figures
    save("figures/b8_corr_linear.pdf", fig)
end

display(fig)
