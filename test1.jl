using NonlinearSolve
using Plots

function MMS!(du, u, p)
    A, B = u
    mu, W, w, a, F = p
    du[1] = -mu*A - (F/(2*w))*sin(B)
    du[2] = (3*a*A^2/(8*w)) - (W - w) - (F/(2*w*A))*cos(B)
end

function main()
    mu = 0.01
    w = 2.0
    a = 0.01
    F = 0.05
    Ws = 0.1:0.01:4.0  # Full resolution
    
    # Preallocate storage
    A_vals = Vector{Float64}(undef, length(Ws))
    B_vals = Vector{Float64}(undef, length(Ws))
    
    # Initial guess from linear response
    W_first = first(Ws)
    complex_response = F / (w^2 - W_first^2 + 2*mu*W_first*im)
    u0 = [abs(complex_response), angle(complex_response)]
    
    # Configure solver
    solver = NewtonRaphson()
    progress = Progress(length(Ws), 1)  # Progress meter
    
    @time for (i, W) in enumerate(Ws)
        p = (mu, W, w, a, F)
        prob = NonlinearProblem(MMS!, u0, p)
        sol = solve(prob, solver, maxiters=1000)
        
        A_vals[i] = sol[1]
        B_vals[i] = sol[2]
        u0 .= sol.u  # Update initial guess
        
        next!(progress)  # Update progress bar
    end
    
    # Create plots
    p1 = plot(Ws, A_vals, 
         xlabel="Frequency W", ylabel="Amplitude A", 
         title="Frequency Response Curve", legend=false)
    
    p2 = plot(Ws, B_vals, 
         xlabel="Frequency W", ylabel="Phase B", 
         title="Phase Response", legend=false)
    
    plot(p1, p2, layout=(2,1), size=(800,600))
end

main()