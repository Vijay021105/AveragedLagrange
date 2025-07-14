using DifferentialEquations
using NonlinearSolve
using LinearAlgebra
using GLMakie

# System dynamics
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
    τst = safe_acos(1 - (2 * fs / (kt * sqrt(Uc^2 + Us^2 ))))
    du[1] = dUc
    du[2] = dUs
    du[3] = (0.5*Uc*kt*sin(2*τst))/(π*m)+(0.5*Us*kt*cos(2*τst))/(π*m)-(1.0*Uc*kt*τst)/(π*m)+Uc*Om^2-(Us*c*Om)/m-2.0*dUs*Om-(0.5*Us*kt)/(π*m)-(Uc*k)/m-(dUc*c)/m+(F)/m
    du[4] = (0.5*Us*kt*sin(2*τst))/(π*m)-(0.5*Uc*kt*cos(2*τst))/(π*m)-(Us*kt*τst)/(π*m)+Us*Om^2+(Uc*c*Om)/m+2.0*dUc*Om+(0.5*Uc*kt)/(π*m)-(Us*k)/m-(dUs*c)/m
end

# Parameters
m = 1.0
c = 0.25
k = 4.0
kt = 5.0
fs = 1.0
Ω = 3.0
F = 1.0
params = [m, c, k, kt, F, fs, Ω]
tspan = (0.0, 50*sqrt(k/m))

# Initial conditions
u0 = [0.01, 0.0, 0.0]
v00 = [0.01, 0.0, 0.0, 0.0]

# Simulate direct model
prob = ODEProblem(jerkins!, u0, tspan, params)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-12, saveat=0.005)

# Simulate second-order averaged lag model
avgcouple = ODEProblem(avglagcouple!, v00, tspan, params)
sol2 = solve(avgcouple, RK4(), reltol=1e-8, abstol=1e-12, saveat=0.005)

# Steady-state from second-order model
steadystate = SteadyStateProblem(avglagcouple!, v00, params)
sol3 = solve(steadystate, SSRootfind())

# Compute signals
t_common = sol2.t
x_direct = [u[1] for u in sol.u][1:length(t_common)]
x_transient = [sol2.u[j][1]*cos(Ω*t_common[j]) + sol2.u[j][2]*sin(Ω*t_common[j]) for j in eachindex(t_common)]
x_steady = [sol3.u[1]*cos(Ω*t) + sol3.u[2]*sin(Ω*t) for t in t_common]

# Errors
x_error_direct = x_direct - x_steady
x_error_lag = x_transient - x_steady

# Correlation
correlation = abs(dot(x_error_direct, x_error_lag) / (norm(x_error_direct) * norm(x_error_lag)))
println("Correlation at F = $F, Ω = $Ω: $correlation")

# Plotting
fig = Figure(resolution=(1000, 400))

ax1 = Axis(fig[1, 1], xlabel="Time (s)", ylabel="Error", title="Error Comparison at F = $F, Ω = $Ω")
lines!(ax1, t_common, x_direct, color=:red, label="Direct ")
lines!(ax1, t_common, x_transient, color=:blue, label="Lag ")
axislegend(ax1)

display(fig)
