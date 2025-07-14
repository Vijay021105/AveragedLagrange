using DifferentialEquations, GLMakie, CairoMakie
using BSON: @save, @load

# ------------------------- CONFIGURATION -------------------------
save_figures = true
save_data = true
analyze = true
save_figures ? CairoMakie.activate!() : GLMakie.activate!()

# ------------------------- PARAMETERS -------------------------
m, c, k = 1.0, 0.25, 4.0
a, F, W = 10, 0.5, 2
p = (m, c, k, a, F, w)
tspan = (0.0, 100.0)
dt = 0.01
tsteps = tspan[1]:dt:tspan[2]
u0_al = [0.01, 0.0, 0.0, 0.0]
u0_num = [0.01, 0.0]

# ------------------------- ODE DEFINITIONS -------------------------
function avgLag!(du, u, p, t)
    A, B, C, D = u
    m, c, k, a, F, w = p
    du[1] = -((16B*m*w^2 - 16B*k + ((-24B*D^2 + (12B^2 - 12A^2)*D - 24B*C^2 + 24A*B*C - 12B^3 - 12A^2*B)*a))/32 +
             (A*c*w - du[2]*c)/2) / (m*w)
    du[2] = ((16A*m*w^2 - 16A*k + ((-24A*D^2 - 24A*B*D - 24A*C^2 + (12B^2 - 12A^2)*C - 12A*B^2 - 12A^3)*a))/32 -
             (B*c*w + du[1]*c - F)/2) / (m*w)
    du[3] = -((144D*m*w^2 - 16D*k + ((-12D^3 + 2*(-6C^2 - 12B^2 - 12A^2)*D + 4B^3 - 12A^2*B)*a))/32 +
             (3C*c*w - du[4]*c)/2) / (3m*w)
    du[4] = ((144C*m*w^2 - 16C*k + ((-12C*D^2 - 12C^3 + 2*(-12B^2 - 12A^2)*C + 12A*B^2 - 4A^3)*a))/32 -
             (3D*c*w + du[3]*c)/2) / (3m*w)
end

function duffing!(du, u, p, t)
    x, dx = u
    m, c, k, a, F, w = p
    du[1] = dx
    du[2] = -(c*dx + k*x + a*x^3 - F*cos(w*t)) / m
end

# ------------------------- SOLVE -------------------------

sol_al = solve(ODEProblem(avgLag!, u0_al, tspan, p), Tsit5(), saveat=tsteps)

sol_num = solve(ODEProblem(duffing!, u0_num, tspan, p), Tsit5(), saveat=tsteps)

# ------------------------- EXTRACT + PROCESS -------------------------
A = getindex.(sol_al.u, 1)
B = getindex.(sol_al.u, 2)
C = getindex.(sol_al.u, 3)
D = getindex.(sol_al.u, 4)
x1 = A .* cos.(w .* sol_al.t) .+ B .* sin.(w .* sol_al.t)
x2 = C .* cos.(3w .* sol_al.t) .+ D .* sin.(3w .* sol_al.t)
@views x_al = A .* cos.(w .* sol_al.t) .+ B .* sin.(w .* sol_al.t) .+
              C .* cos.(3w .* sol_al.t) .+ D .* sin.(3w .* sol_al.t)
if analyze 
    x_num = getindex.(sol_num.u, 1)
    amplitude_ab = @. sqrt(A^2 + B^2)
    amplitude_cd = @. sqrt(C^2 + D^2)

    if save_data
      @save "data/b2_3harm.bson" sol_al sol_num amplitude_ab amplitude_cd x_al x1 x2
    end
else
    @load "data/b2_3harm.bson" sol_al sol_num amplitude_ab amplitude_cd x_al x1 x2
end

# ------------------------- PLOT -------------------------
fig = Figure(resolution=(1100, 850))
Label(fig[0, :], "Duffing Oscillator:3rd Harmonics vs Numerical", fontsize=20)

# Use a 2×3 layout
ax1 = Axis(fig[1, 1], xlabel="Time (s)", ylabel="x(t)", title="Steady-State Time Response")
lines!(ax1, sol_al.t, x_al, label="Avg. Lagrange", color=:blue)
lines!(ax1, sol_num.t, x_num, label="Numerical", linestyle=:dash, color=:red)
xlims!(ax1, 85, 100)
axislegend(ax1, position=:rb)

ax2 = Axis(fig[1, 2], xlabel="Time (s)", ylabel="x(t)", title="Transient Time Response")
lines!(ax2, sol_al.t, x_al, label="Avg. Lagrange", color=:blue)
lines!(ax2, sol_num.t, x_num, label="Numerical", linestyle=:dash, color=:red)
xlims!(ax2, 0, 30)
axislegend(ax2, position=:rb)

ax3 = Axis(fig[2, 1], xlabel="Time (s)", ylabel="Amplitude", title="1st Harmonic Envelope")
lines!(ax3, sol_al.t, x1, label="Acos(wt)+Bsin(wt)", color=:blue)

ax4 = Axis(fig[2, 2], xlabel="Time (s)", ylabel="Amplitude", title="3rd Harmonic Envelope")
lines!(ax4, sol_al.t, x2, label="Ccos(3wt)+Dsin(3wt)", color=:green)

# === PARAMETER BOX ===
param_text = """
Parameters:
m = $(m); c = $(c)
k = $(k); a = $(a)
F = $(F); ω = $(w)
"""
Label(fig[1:2, 3], param_text, halign=:left, valign=:top, fontsize=16, padding=(10,10,10,10))

# Adjust layout
colsize!(fig.layout, 1, Relative(0.43))
colsize!(fig.layout, 2, Relative(0.43))
colsize!(fig.layout, 3, Relative(0.14))

display(fig)

# ------------------------- SAVE -------------------------
if save_figures
    save("figures/b2_3harm_all.pdf", fig)
end
