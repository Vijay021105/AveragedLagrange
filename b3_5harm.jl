using DifferentialEquations, GLMakie, CairoMakie
using BSON: @save, @load

# ------------------------- CONFIGURATION -------------------------
save_figures = true
save_data = true
analyze = true
save_figures ? CairoMakie.activate!() : GLMakie.activate!()

# ------------------------- PARAMETERS -------------------------
m, c, k = 1.0, 0.25, 4.0
a, F, ω = 10, 0.5, 2
p = (m, c, k, a, P, ω)
tspan = (0.0, 100.0)
dt = 0.01
tsteps = dt
u0_al = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
u0_num = [0.01, 0.0]

# ------------------------- ODE DEFINITIONS -------------------------
function highavgLag!(du, u, p, t)
    A, B, C, D, E, F = u
    m, c, k, a, P, w = p
    du[1] = -((B*w)/2) + (B*k)/(2m*w) + (3*B*F^2*a)/(4m*w) + (3*D^2*F*a)/(8m*w) - (3*B*D*F*a)/(4m*w) -
            (3*C^2*F*a)/(8m*w) + (3*A*C*F*a)/(4m*w) + (3*B*E^2*a)/(4m*w) + (3*C*D*E*a)/(4m*w) -
            (3*A*D*E*a)/(4m*w) - (3*B*C*E*a)/(4m*w) + (3*B*D^2*a)/(4m*w) - (3*B^2*D*a)/(8m*w) +
            (3*A^2*D*a)/(8m*w) + (3*B*C^2*a)/(4m*w) - (3*A*B*C*a)/(4m*w) + (3*B^3*a)/(8m*w) +
            (3*A^2*B*a)/(8m*w) - (A*c)/(2m)

    du[2] = (A*w)/2 - (A*k)/(2m*w) - (3*A*F^2*a)/(4m*w) - (3*C*D*F*a)/(4m*w) - (3*A*D*F*a)/(4m*w) -
            (3*B*C*F*a)/(4m*w) - (3*A*E^2*a)/(4m*w) + (3*D^2*E*a)/(8m*w) + (3*B*D*E*a)/(4m*w) -
            (3*C^2*E*a)/(8m*w) - (3*A*C*E*a)/(4m*w) - (3*A*D^2*a)/(4m*w) - (3*A*B*D*a)/(4m*w) -
            (3*A*C^2*a)/(4m*w) + (3*B^2*C*a)/(8m*w) - (3*A^2*C*a)/(8m*w) - (3*A*B^2*a)/(8m*w) -
            (3*A^3*a)/(8m*w) + P/(2m*w) - (B*c)/(2m)

    du[3] = -((3*D*w)/2) + (D*k)/(6m*w) + (D*F^2*a)/(4m*w) + (B*D*F*a)/(4m*w) + (A*C*F*a)/(4m*w) -
            (B^2*F*a)/(8m*w) + (A^2*F*a)/(8m*w) + (D*E^2*a)/(4m*w) - (A*D*E*a)/(4m*w) +
            (B*C*E*a)/(4m*w) - (A*B*E*a)/(4m*w) + (D^3*a)/(8m*w) + (C^2*D*a)/(8m*w) +
            (B^2*D*a)/(4m*w) + (A^2*D*a)/(4m*w) - (B^3*a)/(24m*w) + (A^2*B*a)/(8m*w) - (C*c)/(2m)

    du[4] = (3*C*w)/2 - (C*k)/(6m*w) - (C*F^2*a)/(4m*w) - (A*D*F*a)/(4m*w) + (B*C*F*a)/(4m*w) -
            (A*B*F*a)/(4m*w) - (C*E^2*a)/(4m*w) - (B*D*E*a)/(4m*w) - (A*C*E*a)/(4m*w) +
            (B^2*E*a)/(8m*w) - (A^2*E*a)/(8m*w) - (C*D^2*a)/(8m*w) - (C^3*a)/(8m*w) -
            (B^2*C*a)/(4m*w) - (A^2*C*a)/(4m*w) + (A*B^2*a)/(8m*w) - (A^3*a)/(24m*w) - (D*c)/(2m)

    du[5] = -((5*F*w)/2) + (F*k)/(10m*w) + (3*F^3*a)/(40m*w) + (3*E^2*F*a)/(40m*w) +
            (3*D^2*F*a)/(20m*w) + (3*C^2*F*a)/(20m*w) + (3*B^2*F*a)/(20m*w) + (3*A^2*F*a)/(20m*w) +
            (3*B*D^2*a)/(40m*w) + (3*A*C*D*a)/(20m*w) - (3*B^2*D*a)/(40m*w) +
            (3*A^2*D*a)/(40m*w) - (3*B*C^2*a)/(40m*w) + (3*A*B*C*a)/(20m*w) - (E*c)/(2m)

    du[6] = (5*E*w)/2 - (E*k)/(10m*w) - (3*E*F^2*a)/(40m*w) - (3*E^3*a)/(40m*w) -
            (3*D^2*E*a)/(20m*w) - (3*C^2*E*a)/(20m*w) - (3*B^2*E*a)/(20m*w) - (3*A^2*E*a)/(20m*w) +
            (3*A*D^2*a)/(40m*w) - (3*B*C*D*a)/(20m*w) + (3*A*B*D*a)/(20m*w) -
            (3*A*C^2*a)/(40m*w) + (3*B^2*C*a)/(40m*w) - (3*A^2*C*a)/(40m*w) - (F*c)/(2m)
end

function duffing!(du, u, p, t)
    x, dx = u
    m, c, k, a, P, w = p
    du[1] = dx
    du[2] = -(c*dx + k*x + a*x^3 - P*cos(w*t)) / m
end

# ------------------------- SOLVE -------------------------
sol_high = solve(ODEProblem(highavgLag!, u0_al, tspan, p), Tsit5(), saveat=tsteps)
sol_num  = solve(ODEProblem(duffing!,   u0_num, tspan, p), Tsit5(), saveat=tsteps)

# ------------------------- COMPUTE RESPONSE -------------------------
t = sol_high.t
x_exact = getindex.(sol_num.u, 1)
A, B = getindex.(sol_high.u, 1), getindex.(sol_high.u, 2)
C, D = getindex.(sol_high.u, 3), getindex.(sol_high.u, 4)
E, F = getindex.(sol_high.u, 5), getindex.(sol_high.u, 6)

x1 = @. A .* cos.(ω .* t) .+ B .* sin.(ω .* t)
x3 = @. C .* cos.(3ω .* t) .+ D .* sin.(3ω .* t)
x5 = @. E .* cos.(5ω .* t) .+ F .* sin.(5ω .* t)
@views x_high = A .* cos.(ω .* t) .+ B .* sin.(ω .* t) .+ 
                C .* cos.(3ω .* t) .+ D .* sin.(3ω .* t) .+ 
                E .* cos.(5ω .* t) .+ F .* sin.(5ω .* t)

if analyze
    envelope_ab = @. sqrt(A^2 + B^2)
    envelope_cd = @. sqrt(C^2 + D^2)
    envelope_ef = @. sqrt(E^2 + F^2)
    if save_data
        @save "data/b3_5harm.bson" sol_high sol_num envelope_ab envelope_cd envelope_ef x1 x3 x5 x_high x_exact
    end
else
    @load "data/b3_5harm.bson" sol_high sol_num envelope_ab envelope_cd envelope_ef x1 x3 x5 x_high x_exact
end

# ------------------------- PLOT -------------------------
fig = Figure(resolution=(1200, 900))
Label(fig[0, :], "Duffing Oscillator: Harmonic Balance Comparison", fontsize=22)

# Steady-State
ax1 = Axis(fig[1, 1], xlabel="Time (s)", ylabel="Displacement (m)", title="Steady-State Comparison")
lines!(ax1, t, x_exact, label="Numerical", color=:black)
lines!(ax1, t, x_high,  label="Analytical", color=:blue, linestyle=:dash)
xlims!(ax1, 80, 100)
axislegend(ax1, position=:rb)

# Transient
ax2 = Axis(fig[1, 2], xlabel="Time (s)", ylabel="Displacement (m)", title="Transient Comparison")
lines!(ax2, t, x_exact, label="Numerical", color=:black)
lines!(ax2, t, x_high,  label="Analytical", color=:blue, linestyle=:dash)
xlims!(ax2, 0, 30)
axislegend(ax2, position=:rb)

# Harmonic Plots
ax3 = Axis(fig[2, 1], xlabel="Time (s)", ylabel="x₁(t)", title="1st Harmonic Response")
lines!(ax3, t, x1, color=:green)

ax4 = Axis(fig[2, 2], xlabel="Time (s)", ylabel="x₃(t)", title="3rd Harmonic Response")
lines!(ax4, t, x3, color=:orange)

ax5 = Axis(fig[3, 1:2], xlabel="Time (s)", ylabel="x₅(t)", title="5th Harmonic Response")
lines!(ax5, t, x5, color=:purple)

# Parameter Box
param_text = """
Parameters:
m = $m     c = $c
k = $k     a = $a
P = $P     ω = $ω
"""
Label(fig[1:3, 3], param_text, halign=:left, valign=:top, fontsize=16, padding=(10,10,10,10))
Box(fig[1:3, 3], strokecolor=:black, strokewidth=1)

colsize!(fig.layout, 1, Relative(0.42))
colsize!(fig.layout, 2, Relative(0.42))
colsize!(fig.layout, 3, Relative(0.16))

display(fig)

# ------------------------- SAVE -------------------------
if save_figures
    save("figures/b3_5harm.pdf", fig)
end
