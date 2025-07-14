using DifferentialEquations
using GLMakie, CairoMakie
using BSON: @save, @load

# ------------------------- CONFIGURATION -------------------------
save_figures = true
save_data = true
analyze = true
save_figures ? CairoMakie.activate!() : GLMakie.activate!()
# ------------------------- PARAMETERS -------------------------

m, c, k = 1.0, 0.25, 4.0
a, F, W = 10, 0.5, 2
p = (m, c, k, a, F, W)
tspan = (0.0, 100.0)
dt = 0.01
tsteps = tspan[1]:dt:tspan[2]
u0_al = [0.01, 0.0, 0.0, 0.0]
u0_num = [0.01, 0.0]

function coupledavglag!(du, u, p, t)
    A,B,dA,dB = u 
    m,c,k,a,P,w = p
    du[1] = dA
    du[2] = dB
    du[3] = ((4*A*m*w^2+(-(8*(dB)*m)-4*B*c)*w-4*A*k-4*(dA)*c+(-(3*A*B^2)-3*A^3)*a+4*P)/(4*m))
    du[4] = (4*B*m*w^2+(8*(dA)*m+4*A*c)*w-4*B*k-4*(dB)*c+(-(3*B^3)-3*A^2*B)*a)/(4*m)
end

function ODE!(du, u, p, t)
    x, dx = u
    m, c, k, a, P, w = p
    du[1] = dx
    du[2] = -(c*dx + k*x + a*x^3 - P*cos(w*t))/m
end
# ------------------------- SOLVE -------------------------

sol_coupled_avg = solve(ODEProblem(coupledavglag!, u0_al, tspan, p), Tsit5(),saveat=tsteps)
sol_num = solve(ODEProblem(ODE!, u0_num, tspan, p), Tsit5(),saveat=tsteps)
# ------------------------- EXTRACT + PROCESS -------------------------

A = getindex.(sol_coupled_avg.u, 1)
B = getindex.(sol_coupled_avg.u, 2)
@views x_al = A .* cos.(W .* sol_coupled_avg.t) .+ B .* sin.(W .* sol_coupled_avg.t)

if analyze
    envelope_al = @. sqrt(A^2 + B^2)
    phase_al = @. atan(B, A)
    if save_data
        @save "data/b4_2degree.bson" sol_coupled_avg sol_num envelope_al phase_al 
    end
else
    @load "data/b4_2degree.bson" sol_coupled_avg sol_num envelope_al phase_al 
end


# ------------------------- PLOT -------------------------
fig = Figure(resolution=(1000, 800))
Label(fig[0, :], "Duffing Oscillator: Average Lagrange vs Numerical", fontsize=20)

# Use a 2×3 layout: 3rd column for parameter display
grid = fig[1:2, 1:2]  # Plot axes go here
params = fig[1:2, 3]  # This is the column for parameter box

# === AXES ===
ax1 = Axis(fig[1, 1], xlabel="Time (s)", ylabel="Displacement (m)", title="Time Domain at steady state")
lines!(ax1, sol_coupled_avg.t, x_al, label="Avg. Lagrange", color=:blue)
lines!(ax1, sol_num.t, x_num, label="Numerical", linestyle=:dash, color=:red)
xlims!(ax1, 80, 100)
axislegend(ax1, position=:rb)

ax2 = Axis(fig[1, 2], xlabel="Time (s)", ylabel="Displacement", title="Time Domain at transient state")
lines!(ax2, sol_coupled_avg.t, x_al, label="Avg. Lagrange", color=:blue)
lines!(ax2, sol_num.t, x_num, label="Numerical", linestyle=:dash, color=:red)
xlims!(ax2, 0, 30)
axislegend(ax2, position=:rb)

ax3 = Axis(fig[2, 1:2], xlabel="Time (s)", ylabel="Envelope (m)", title="Amplitude Envelopes")
lines!(ax3, sol_coupled_avg.t, envelope_al, label="Avg. Lagrange", color=:blue)
lines!(ax3, sol_num.t, x_num, label="Numerical", linestyle=:dash, color=:red)
axislegend(ax3, position=:rb)

# === PARAMETER BOX ===
param_text = """
Parameters:
m = $(m); c = $(c) 
k = $(k); a = $(a)
F = $(F); ω = $(W)
"""
Label(fig[1:2, 3], param_text, halign=:left, valign=:top, fontsize=16, padding=(10,10,10,10))

# Adjust layout
colsize!(fig.layout, 1, Relative(0.45))
colsize!(fig.layout, 2, Relative(0.45))
colsize!(fig.layout, 3, Relative(0.1))

display(fig)


# ------------------------- SAVE -------------------------
if save_figures
    save("figures/b4_1harm_2degree.pdf", fig)

end