using DifferentialEquations
using Plots

function jerkins!(du, u, p, t)
    m, c, k, kt, F, fs,Om = p
    
    du[1] = u[2]  # u[1] is displacement, u[2] is velocity
    du[2] = -(c*u[2] + k*u[1] + kt*(u[1]-u[3]) - F*cos(Om*t))/m 
    du[3] = (kt*(u[1]-u[3]) * sign(u[2])) <= fs ? 0.0 : u[2]  # u[3] is w
end

# Parameters
m = 1.0      # kg
c = 0.1      # kg/s
k = 4.0      # kg/s²
kt = 4.0     # kg/(m² s²)
F = 1.1      # N
fs = 1.0     # N (static friction force)
Om = 2.0    # rad/s (frequency of the external force)
p = [m, c, k, kt, F, fs,Om]

# Initial conditions [displacement, velocity, w]
u0 = [0.01, 0.0, 0.0]
tspan = (0.0, 100.0)

# Solve the ODE
prob = ODEProblem(jerkins!, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-12)

# Extract values for plotting
u = [state[1] for state in sol.u]  # Displacement
force = kt .* (u .- [state[3] for state in sol.u])  # kt*(u-w)

# Create the force vs displacement plot
plot(u, force, 
     label="kt*(u-w)", 
     xlabel="Displacement u (m)", 
     ylabel="Force kt(u-w) (N)",
     title="Force vs Displacement",
     lw=2,
     legend=:bottomright)

