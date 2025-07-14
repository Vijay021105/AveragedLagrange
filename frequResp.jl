using DifferentialEquations
using LinearAlgebra
using GLMakie
using NonlinearSolve

function MMS!(du, u, p, t=nothing)
    A, B = u
    mu, W, w, a, F = p
    du[1] = -mu*A - (F/(2*w))*sin(B)
    du[2] = (3*a*A^2/(8*w)) - (W - w) - (F/(2*w*A))*cos(B)
end
function EMS!(du, u, p, t=nothing)
    A,B=u
    mu,W,w,a,F=p
    du[1] = - mu*A - (F/(2W))*sin(B)
    du[2] =  (3*a*A^2)/(8W) + ((w^2-W^2)/2W) - F*cos(B)/(2*A*W) 
end

mu = 0.01
w = 2.0
a = 0.1
F = 0.1

# Range of W values
Ws = 1.75:0.01:2.75

# Initialize arrays to store solutions
x = Float64[]
y = Float64[]

# Initial guess vector
s0 = zeros(2)

# Iterate over each W value
# for (i, W) in enumerate(Ws)
for i in eachindex(Ws)
W = Ws[i];

    p = [mu, W, w, a, F]
    
    # Initial guess for the first iteration
    if i == 1
        
        linear_response = F / w^2 - W^2 + 2*mu*W*im
        s0[1] = abs(linear_response)
        s0[2] = angle(linear_response)
    else
        # Use previous solution as initial guess
        s0 .= sol_prev
    end
    
    # Set up and solve the steady state problem
    prob = SteadyStateProblem(MMS!, s0, p)
    sol = solve(prob, SSRootfind())
    
    # Check for convergence
    if sol.retcode != ReturnCode.Success
        @warn "Solution did not converge for W = $W"
        # continue
    end
    
    # Store the solution
    push!(x, sol.u[1])
    push!(y, sol.u[2])
    
    # Update previous solution
    sol_prev = sol.u
end

# Plot the results
fig = Figure()
ax = Axis(fig[1, 1], title="Amplitude vs Phase", xlabel="A", ylabel="B")
scatterlines!(ax, Ws, abs.(x), color=:blue, linewidth=2)
display(fig)