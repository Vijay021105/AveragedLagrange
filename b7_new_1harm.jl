using DifferentialEquations, GLMakie, CairoMakie, LinearAlgebra, Printf
using BSON: @save, @load

# ---------------------- CONFIG ----------------------
save_figures = true
save_data = true
analyze = true
save_figures ? CairoMakie.activate!() : GLMakie.activate!()

# ---------------------- PARAMETERS ----------------------
m = 1.0
c = 0.25
k = 4.0
a = 10
F = 0.5
Ω = 1.99
w0 = sqrt(k / m)
tspan = (0.0, 100.0)
dt = 0.01

# ---------------------- SYSTEM EQUATIONS ----------------------
function oscillator!(du, u, p, t)
    A, B = u
    m, c, k, a, F, Ω, w0 = p
    denom = 4π * m * (Ω^2 - w0^2)

    du[1] = -( (2F*w0*cos((2π*Ω)/w0) + 2π*A*c*Ω^2 - 2π*A*c*w0^2 - 2F*w0) / denom +
               B*w0/2 - B*k/(2m*w0) - (3*B^3*a)/(8m*w0) - (3*A^2*B*a)/(8m*w0) )

    du[2] = ( (2F*Ω*sin((2π*Ω)/w0) - 2π*B*c*Ω^2 + 2π*B*c*w0^2) / denom +
              A*w0/2 - A*k/(2m*w0) - (3*A*B^2*a)/(8m*w0) - (3*A^3*a)/(8m*w0) )
end

function duffingODE!(du, u, p, t)
    x, dx = u
    m, c, k, a, F, Ω = p
    du[1] = dx
    du[2] = -(c*dx + k*x + a*x^3 - F*cos(Ω*t)) / m
end

# ---------------------- SOLVE ----------------------
u0_amp = [0.01, 0.0]
u0_num = [0.01, 0.0]
p_amp = [m, c, k, a, F, Ω, w0]
p_num = [m, c, k, a, F, Ω]

prob_amp = ODEProblem(oscillator!, u0_amp, tspan, p_amp)
sol_amp = solve(prob_amp, Tsit5(), saveat=dt)

prob_num = ODEProblem(duffingODE!, u0_num, tspan, p_num)
sol_num = solve(prob_num, Tsit5(), saveat=dt)

# ---------------------- POST-PROCESS ----------------------
t_vals = sol_amp.t
A_vals = [u[1] for u in sol_amp.u]
B_vals = [u[2] for u in sol_amp.u]
x_recon = A_vals .* cos.(w0 .* t_vals) .+ B_vals .* sin.(w0 .* t_vals)
x_num = [u[1] for u in sol_num.u]
amplitude_envelope = sqrt.(A_vals .^ 2 .+ B_vals .^ 2)

if save_data && analyze
    @save "data/b7_new_AL.bson" sol_amp sol_num A_vals B_vals x_recon x_num amplitude_envelope t_vals
elseif !analyze
    @load "data/b7_new_AL.bson" sol_amp sol_num A_vals B_vals x_recon x_num amplitude_envelope t_vals
end

# ---------------------- PLOTTING ----------------------
fig = Figure(resolution=(1200, 600), fontsize=16)

# (1) Steady-state Comparison
ax1 = Axis(fig[1, 1], xlabel="Time (s)", ylabel="Displacement (m)", title="Steady-State Comparison")
lines!(ax1, t_vals, x_num, label="Numerical", color=:blue, linewidth=2)
lines!(ax1, t_vals, x_recon, label="Analytical (A,B)", color=:red, linestyle=:dash, linewidth=2)
xlims!(ax1,85,100)
axislegend(ax1, position=:rb)

# (2) Transient Envelope (A,B) vs Numerical
ax2 = Axis(fig[1, 2], xlabel="Time (s)", ylabel="Displacement (m)", title="Transient A,B Evolution")
lines!(ax2, t_vals, x_num, color=:gray, label="Numerical", linewidth=1.5)
lines!(ax2, t_vals,x_recon, color=:green, linestyle=:dot, label="Analytical")
xlims!(ax2,0,15)
axislegend(ax2, position=:rb)

# (3) Amplitude Envelope
ax3 = Axis(fig[2, 1], xlabel="Time (s)", ylabel="Displacement (m)", title="Envelope over Numerical")
lines!(ax3, t_vals, x_num, label="Numerical", color=:black, linewidth=1.5)
lines!(ax3, t_vals, amplitude_envelope, color=:red, linestyle=:dash, label="Envelope")
axislegend(ax3, position=:rb)

# Save or Display
if save_figures
    save("figures/b7_new_AL.pdf",fig)
end


display(fig)