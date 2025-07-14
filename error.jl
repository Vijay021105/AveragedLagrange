using DifferentialEquations
using LinearAlgebra
using Plots
using NonlinearSolve

function MMS!(du, u, p, t)
    A, B = u
    mu, W, w, a, F = p
    du[1] = -mu*A - (F/(2*w))*sin(B)
    du[2] = (3*a*A^2/(8*w)) - (W - w) - (F/(2*w*A))*cos(B)
end

function Lag!(du, u, p, t)
    A, B = u
    m, c, k, W, a = p
    du[1] = ((3*B^3 + 3*A^2*B)*a - 4*B*m*W^2 - 4*A*c*W + 4*B*k)/(8*m*W)
    du[2] = -((3*A^3 + 3*B^2*A)*a - 4*A*m*W^2 + 4*B*c*W + 4*A*k - 4*F)/(8*m*W)
end

function ODE!(du, u, p, t)
    x, dx = u
    m, c, k, a, F, W = p
    du[1] = dx
    du[2] = -(c*dx + k*x + a*x^3 - F*cos(W*t))/m
end

# Parameters
m = 1
c = 0.01
k = 4
w = sqrt(k/m)
mu = c/(2*m)
a = 0.0  # Non-linearity coefficient
F = 0.5

# Frequency range
Ws_forward = 1.25:0.025:3.25

function solve_system(Ws)
    x_mms = Float64[]
    x_lag = Float64[]
    amplitude_ode = Float64[]
    
    s0 = [0.0, 0.0]
    sol_prev_mms = zeros(2)
    
    for i in eachindex(Ws)
        W = Ws[i]
        p = [mu, W, w, a, F]
        q = [m, c, k, W, a, F]
        
        # Initial guess
        if i == 1
            linear_response = F / ((w^2 - W^2) + 2*mu*W*im)
            s0[1] = abs(linear_response)
            s0[2] = angle(linear_response)
        else
            s0 .= sol_prev_mms
        end        
        # Solve MMS
        prob_mms = SteadyStateProblem(MMS!, s0, p)
        sol_mms = solve(prob_mms, SSRootfind())
        
        # Solve Lagrangian method
        prob_lag = SteadyStateProblem(Lag!, s0, q)
        sol_lag = solve(prob_lag, SSRootfind())
        
        # Solve ODE and fit to A*cos(Wt) + B*sin(Wt)
        prob_ode = ODEProblem(ODE!, [0.01, 0.0], [490, 500.0], q)
        sol_ode = solve(prob_ode, Tsit5())
        
        # Extract steady-state portion (last 50 time units)
        steady_start_idx = findfirst(t -> t >= 450.0, sol_ode.t)
        if steady_start_idx === nothing
            steady_start_idx = div(length(sol_ode.t), 2)
        end
        
        t_steady = sol_ode.t[steady_start_idx:end]
        x_steady = [u[1] for u in sol_ode.u[steady_start_idx:end]]
        
        # Fit to A*cos(Wt) + B*sin(Wt) using least squares
        n_points = length(t_steady)
        design_matrix = zeros(n_points, 2)
        
        for j in 1:n_points
            design_matrix[j, 1] = cos(W * t_steady[j])  # cos(Wt) coefficient
            design_matrix[j, 2] = sin(W * t_steady[j])  # sin(Wt) coefficient
        end
        
        # Solve for coefficients [A_fit, B_fit]
        coeffs = design_matrix \ x_steady
        A_fit, B_fit = coeffs
        
        # Calculate amplitude: sqrt(A^2 + B^2)
        amplitude_fit = sqrt(A_fit^2 + B_fit^2)
        
        # Store results
        sol_prev_mms .= sol_mms.u
        push!(x_mms, abs(sol_mms.u[1]))
        push!(x_lag, sqrt(sol_lag.u[1]^2 + sol_lag.u[2]^2))  # sqrt(A^2 + B^2) for Lagrangian
        push!(amplitude_ode, amplitude_fit)
    end
    
    return x_mms, x_lag, amplitude_ode
end

# Solve system
x_mms_forward, x_lag_forward, amplitude_ode_forward = solve_system(Ws_forward)

# Calculate errors
error_mms = -amplitude_ode_forward .+ x_mms_forward
error_lag = -amplitude_ode_forward .+ x_lag_forward

# Create plot showing errors
plot(title="Error Analysis: ODE vs Analytical Methods", 
     xlabel="Frequency W", 
     ylabel="Error (ODE Amplitude - Analytical Amplitude)", 
     legend=:topleft,
     grid=true)

# Plot errors
plot!(Ws_forward, error_mms, 
      label="Error: ODE - MMS", 
      color=:blue, 
      linewidth=2, 
      markershape=:circle, 
      markersize=3)

plot!(Ws_forward, error_lag, 
      label="Error: ODE - Lagrangian", 
      color=:red, 
      linewidth=2, 
      markershape=:square, 
      markersize=3)

# Add horizontal line at zero for reference
hline!([0], color=:black, linestyle=:dash, alpha=0.5, label="Zero Error")

# Display the plot
display(current())