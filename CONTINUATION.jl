using NonlinearSolve
using LinearAlgebra
using Printf
using ForwardDiff
using Markdown
using Infiltrator

# * Custom Abstract Types
const nvTypes = Union{Nothing, Vector{Float64}};
const nmTypes = Union{Nothing, Matrix{Float64}};

# * Define a struct to store A Solution Point

"""
   myNLSoln

Struct storing the solution, Jacobian, and (unit) tangent.

# Fields
- `up::nvTypes`: Solution point ([u;p])
- `J::nmTypes`: Jacobian dr/du
- `Jp::nvTypes`: Parameter Jacobian dr/dp
- `dupds::nvTypes`: Tangent vector (has to be vector only)  
"""
struct myNLSoln
    up::nvTypes
    J::nmTypes
    Jp::nvTypes
    dupds::nvTypes  # HAS to be a vector only!
end

"""
   myNLSoln(up::nvTypes=nothing; J::nmTypes=nothing, Jp::nvTypes=nothing)

Constructor for myNLSoln. Can specify one or all the arguments (point, jacobian, paramjac).

# Arguments
- `up::nvTypes`       : (default nothing) Solution point
- `J::nmTypes`        : (default nothing) Jacobian
- `Jp::nvTypes`       : (default nothing) paramjac
- `save_jacs::Bool`   : (default false) Whether or not to store the Jacobians.

If both J and Jp are provided, the unit tangent dupds is computed using `nullspace([J Jp])[:,1]`.
"""
function myNLSoln(up::nvTypes=nothing; J::nmTypes=nothing, Jp::nvTypes=nothing, save_jacs::Bool=false)
    if J === nothing || Jp === nothing
        dupds = nothing;
    else
        # dupds = -J\Jp;  # Naive, requires IFT to hold.
        # dupds = nullspace([J Jp])[:,1]; # General, but have to handle bifurcations better!
        dupds = normalize([-J\Jp;1.0]);
        if !save_jacs
            J = nothing;
            Jp = nothing;
        end
    end
    return myNLSoln(up, J, Jp, dupds);
end

"""
   Base.show(io::IO, p::myNLSoln)

Overloading base show function to display myNLSoln object.
"""
function Base.show(io::IO, p::myNLSoln)
    print(io, " up = $(p.up)\n dupds = $(p.dupds)\n J = $(p.J)\n Jp = $(p.Jp)")
end

"""
   Base.:-(v1::myNLSoln, v2::myNLSoln)

Overloading the subtraction operator to take the difference between two myNLSoln 
"""
function Base.:-(v1::myNLSoln, v2::myNLSoln)
    if v1.up !== nothing && v2.up !== nothing
        dup = v1.up-v2.up;
    else
        dup = nothing;
    end
    if v1.J !== nothing && v2.J !== nothing
        dJ = v1.J-v2.J;
    else
        dJ = nothing;
    end
    if v1.Jp !== nothing && v2.Jp !== nothing
        dJp = v1.Jp-v2.Jp;
    else
        dJp = nothing;
    end
    if v1.dupds !== nothing && v2.dupds !== nothing
        ddupds = v1.dupds-v2.dupds;
    else
        ddupds = nothing;
    end
    return myNLSoln(dup, dJ, dJp, ddupds);
end

# * Continuation Utilities

# ** Extended Residue Function
"""
   EXTRESFUN!(up::nvTypes, fun, sol0::myNLSoln, ds::Float64; parm::Symbol=:riks, Dsc::Vector{Float64}, dup::nvTypes=nothing, Jf::nmTypes=nothing)

Returns the bordered/extended residue function. Appends residue with required arclength constraint.

# Note
The function involves very naive autodiff calls. Does not respect or try to detect jacobian sparsity. For now it is best to provide analytical jacobians for very large problems. 

# Arguments
+ `up::nvTypes`: Solution point for evaluation

+ `fun`: Nonlinear function specified in a fashion that can be called as
  + `fun.f(du,up[1:end-1],up[end])` for the residue.

  + `fun.jac(J,up[1:end-1],up[end])` for the jacobian (uses `ForwardDiff.jacobian!` if `fun.jac` is `nothing`)

  + `fun.paramjac(Jp,up[1:end-1],up[end])` for the parameter jacobian (uses `ForwardDiff.jacobian!` if `fun.paramjac` is `nothing`)

  + Development Done based on `NonlinearFunction` from `NonlinearSolve`

+ `sol0::myNLSoln`: Previous solution point.

+ `ds::Float64`: Step size.

+ `parm::Symbol=:riks`: Arclength parameterization. Defaults to :riks. Possible are:

  + `:riks`: Riks' or normal parameterization.

  + `:arclength`: Arclength parameterization.

+ `Dsc::Vector{Float64}`: Scaling vector. Recommended to be of the order of magnitude of the expected unknowns. Size same as `up`.

+ `dup::nvTypes=nothing`: Residue vector (for iip evaluation)

+ `Jf::nmTypes=nothing`: Jacobian matrix (for iip evaluation)

# Returns
+ `dup::nvTypes=nothing`: Residue vector (for iip evaluation)

+ `Jf::nmTypes=nothing`: Jacobian matrix (for iip evaluation)

"""
function EXTRESFUN!(up, fun, sol0, ds; parm=:arclength, Dsc=ones(eltype(up), length(up)), Ralsc=1.0, dup=nothing, Jf=nothing)

    # Residue Evaluation
    if dup !== nothing
        du = @view dup[1:end-1]

        fun.f(du, up[1:end-1], up[end])
    end
    if Jf !== nothing
        J = @view Jf[1:end-1, 1:end-1];
        Jp = @view Jf[1:end-1, end];

        if fun.jac !== nothing
            fun.jac(J, up[1:end-1], up[end]);
        else
            Jt = @view Jf[1:end-1, :];
            R = similar(up[1:end-1]);
            ForwardDiff.jacobian!(Jt, (du,up) -> fun.f(du,up[1:end-1], up[end]), R, up);
        end
        if fun.paramjac !== nothing
            fun.paramjac(Jp, up[1:end-1], up[end]);
        end
    end

    # Arclength Constraint Evaluation
    if dup !== nothing
        if parm==:riks
            tgt_ = normalize(sol0.dupds./Dsc);
    	    dup[end] = tgt_'*((up-sol0.up)./Dsc)-ds;
        elseif parm==:arclength
            # dup[end] = norm((up-sol0.up)./Dsc)^2-ds^2;
            dup[end] = norm((up-sol0.up)./Dsc)-ds;
        end
        dup[end] *= Ralsc;
    end
    if Jf !== nothing
        if parm==:riks
            tgt_ = normalize(sol0.dupds./Dsc);
    	    Jf[end, :] = (tgt_./Dsc)';
        elseif parm==:arclength
            # Jf[end, :] = 2((up-sol0.up)./Dsc.^2)';
            Jf[end, :] = ((up-sol0.up)./Dsc.^2)'/norm((up-sol0.up)./Dsc);
        end
        Jf[end, :] .*= Ralsc;
    end
    return nothing;
    
end

# *** Scaled EXTRESFUN. Don't Think This is Necessary
function EXTRESFUN_scaled!(up, fun, sol0, ds; parm=:arclength, Dsc, dup=nothing, Jf=nothing)

    # Residue Evaluation
    if dup !== nothing
        du = @view dup[1:end-1]

        fun.f(du, Dsc[1:end-1].*up[1:end-1], Dsc[end]up[end])
    end
    if Jf !== nothing
        J = @view Jf[1:end-1, 1:end-1];
        Jp = @view Jf[1:end-1, end];

        if fun.jac !== nothing
            fun.jac(J, Dsc[1:end-1].*up[1:end-1], Dsc[end]up[end]);
            J = J*diagm(Dsc[1:end-1]);
        else
            Jt = @view Jf[1:end-1, :];
            R = similar(up[1:end-1]);
            ForwardDiff.jacobian!(Jt, (du,up) -> fun.f(du,Dsc[1:end-1].*up[1:end-1], Dsc[end]up[end]), R, up);
        end
        if fun.paramjac !== nothing
            fun.paramjac(Jp, Dsc[1:end-1].*up[1:end-1], Dsc[end]up[end]);
            Jp = Jp*Dsc[end];
        end
    end

    # Arclength Constraint Evaluation
    if dup !== nothing
        if parm==:riks
            tgt_ = normalize(sol0.dupds./Dsc);
    	    dup[end] = tgt_'*(up-sol0.up./Dsc)-ds;
        elseif parm==:arclength
            # dup[end] = norm((up-sol0.up)./Dsc)^2-ds^2;
            dup[end] = norm(up-sol0.up./Dsc)-ds;
        end
    end
    if Jf !== nothing
        if parm==:riks
            tgt_ = normalize(sol0.dupds./Dsc);
    	    Jf[end, :] = (tgt_)';
        elseif parm==:arclength
            # Jf[end, :] = 2((up-sol0.up)./Dsc.^2)';
            Jf[end, :] = (up-sol0.up./Dsc)'/norm(up-sol0.up./Dsc);
        end
    end
    return nothing;
    
end

# ** Continuation Routine

"""
   CONTINUATE(u0, fun, ps, dp; kwargs...)

Continuation routine. Solves the bordered problem with residue drawn from `EXTRESFUN!`.

# Arguments
- `u0::Vector{Float64}`                          : Initial guess for first point
- `fun`                                          : `NonlinearFunction` object. Will call:
  - `fun.f(du, u, p)` for residue r. p must be scalar.
  - `fun.jac(J, u, p)` for jacobian dr/du. Uses `ForwardDiff.jacobian!` if `fun.jac` is `nothing`.
  - `fun.paramjac(Jp, u, p)` for jacobian dr/dp. Uses `ForwardDiff.jacobian!` if `fun.paramjac` is `nothing`.
- `ps::Vector{Float64}`                          : Specify range of p for continuation.
- `dp::Float64`                                  : Specify first step in Δp units. (will be rescaled if necessary)
## Optional Arguments
- `parm::Symbol`                                 : (default `:arclength`) Arclength parameterization. See `EXTRESFUN!`.
- `nmax::Int64`                                  : (default `1000`) Maximum number of steps.
- `dpbnds::Union{Nothing,Vector{Float64}}`       : (default `[dp/5,5dp]`) Bounds for the step length dp. (Rescaled as appropriate)
- `save_jacs::Bool`                              : (default `false`) Specify whether or not to save the Jacobians in the output. Only solution and (unit) tangent are saved if false.
- `verbosity::Int`                               : (default 1) Verbosity levels:
  - 0: Suppress all messages.
  - 1: Display Steps, Continuation Progress.
  - 2: Display Iteration Information for each step also.
### For Scaling of Unknowns (recommended to get evenly spaced points on response curve)  
    
- `Dsc::Union{Symbol,Nothing,Vector{Float64}}`   : (default `:auto`) "Dscaling" used to scale the unknowns. The arc length constraint is applied in the scaled space (`uₛ=uₚₕ./Dsc`).
  
  - If set to `:auto`, it uses the absolute of the first converged solution as the initial `Dsc` vector. Zero entries are replaced with `minDsc`.
  - If set to `:none`, it fixes Dsc to a vector of ones and doesn't dynamically adapt it. (this forces `DynScale` to `false`).
  - If set to `:ones`, it fixes Dsc to a vector of ones but dynamically continues to scale.
  
- `DynScale::Bool`                               : (default `true`) Whether or not to dynamically adapt the `Dsc` vector. Each entry is allowed to grow or shrink by a maximum factor of 2 in each step if `true`.
- `minDsc::Float64`                              : (default eps()^(4//5)=3e-13) Minimum value for Dscale.
### For Step Length Adaptation. Currently set as dsₙ = dsₒ * xi, where xi = clamp((itopt/itns)^nxi, xirange[1], xirange[2]).
- `itopt::Union{Symbol,Int}`                     : (default :auto) Optimal number of  
- `nxi::Float64`                                 : (default 0.0) Switches off adaptation by default. 
- `xirange::Vector{Float64}`                     : (default [0.5,2.])

# Returns
- `vector{myNLsoln}` representing solutions.
- `vector{Int}` representing iterations taken.
- `vector{Float64}` representing step sizes.
- `vector{Float64}` representing adaptation parameter ξ.
- `vector{Float64}` representing the final Dscale matrix.
"""
function CONTINUATE(u0::Vector{Float64}, fun, ps::Vector{Float64}, dp::Float64;
                    parm::Symbol=:arclength, nmax::Int64=1000,
                    dpbnds::Union{Nothing,Vector{Float64}}=nothing,
                    save_jacs::Bool=false, verbosity::Int=1,
                    Dsc::Union{Symbol,Nothing,Vector{Float64}}=:auto,
                    DynScale::Bool=true,
                    itopt::Union{Symbol,Int}=:auto,
                    nxi::Float64=0.5, xirange::Vector{Float64}=[0.5, 2.0],
                    minDsc::Float64=eps()^(4//5))
    
    if dpbnds === nothing
        dpbnds = [dp/5, 5dp];
    end

    # Temporary Variables
    N = length(u0);
    Rf = similar(u0, N+1);
    Jf = similar(u0, N+1,N+1);
    R = @view Rf[1:N];
    Rp = @view Rf[N+1];
    J = @view Jf[1:N,1:N];
    Jp = @view Jf[1:N,N+1];

    # Initialize Storers
    sols = myNLSoln[];
    its = Int[];
    dss = Float64[];
    xis = Float64[];

    # First Solution Point   
    prob0 = NonlinearProblem(fun, u0, ps[1]);
    if verbosity>0
        display(md"");
        display(md"# Correcting the Initial Point")
    end
    solp0 = solve(prob0, show_trace=Val(verbosity>0));
    if fun.jac !== nothing
        fun.jac(J, solp0.u, ps[1]);
    else
        ForwardDiff.jacobian!(J, (R,u)->fun.f(R,u,ps[1]), R, solp0.u);
    end
    if fun.paramjac !== nothing
        fun.paramjac(Jp, solp0.u, ps[1]);
    else
        ForwardDiff.jacobian!(Jp, (R,p)->fun.f(R,solp0.u,p), R, [ps[1]]);
    end

    push!(sols, myNLSoln([solp0.u;ps[1]]; J=copy(J), Jp=copy(Jp), save_jacs=save_jacs));
    sols[end].dupds .*= sign(sols[end].dupds[end]*(ps[2]-ps[1]));
    push!(its, solp0.stats.nsteps)

    if Dsc==:auto
        # minDsc = minimum(abs.(sols[end].up[sols[end].up.!=0]))/100;
        Dsc = max.(abs.(sols[end].up), minDsc);

        # Dsc = abs.(sols[end].up);
        # Dsc[isapprox.(sols[end].up, 0.0, atol=eps()^(4/5))] .= 1.0;

        # un = abs.(normalize(sols[end].up[1:end-1]));
        # pn = abs(sols[end].up[end]);
    elseif Dsc==:none
        Dsc = ones(length(sols[end].up));
        DynScale = false;
    elseif Dsc==:ones
        Dsc = ones(length(sols[end].up));        
    end

    # Recontextualize dp (such that first step is as requested)
    ds = dp/Dsc[end]normalize(sols[end].dupds./Dsc)[end]sign(ps[2]-ps[1]);
    push!(dss,  ds);
    push!(xis,  1.0);
    dsbnds = dpbnds/dp*ds;

    # Setup Extended Problem
    exfun = NonlinearFunction((du,up,p)->EXTRESFUN!(up, fun, p[1], p[2];
                                                    parm=parm, Dsc=p[3], dup=du,
                                                    Ralsc=p[4]),
                              jac=(J,up,p)->EXTRESFUN!(up, fun, p[1], p[2];
                                                       parm=parm, Dsc=p[3], Jf=J,
                                                       Ralsc=p[4]));
    
    if itopt == :auto # Automatic itopt
        # itopt as no. of iterations required for first point.
        tgt_ = Dsc.*normalize(sols[end].dupds./Dsc);
        up0_ = sols[end].up + dss[end]tgt_;

        # Get Ralsc
        exfun.f(Rf, up0_, (sols[end], dss[end], Dsc, 1.0));
        Ralsc_ = norm(R)/dss[end];
        
        prob_ = NonlinearProblem(exfun, up0_, (sols[end], dss[end], Dsc, Ralsc_));
        solp_ = solve(prob_);
        # itopt = max(solp_.stats.nsteps, 3);
        itopt = solp_.stats.nsteps;
    else
        up0_ = copy(sols[end].up);
        prob_ = NonlinearProblem(exfun, up0_, (sols[end], dss[end], Dsc, 1.0));
    end
    
    if verbosity>0
        display(md"# Starting Continuation (_itopt = $(itopt), nxi = $(nxi)_)")
    end
    
    while sols[end].up[end]sign(ps[2]-ps[1])<ps[2]sign(ps[2]-ps[1]) && length(sols)<=nmax
    	# Tangent Predictor
        Dsc_ = Dsc;
        tgt = Dsc_.*normalize(sols[end].dupds./Dsc_);
        up0 = sols[end].up + dss[end]tgt;

        # Get Ralsc
        exfun.f(Rf, up0, (sols[end], dss[end], Dsc, 1.0));
        Ralsc = norm(R)/dss[end];

        # Constrained Corrector
        prob = remake(prob_; u0=up0, p=(sols[end], dss[end], Dsc, Ralsc));
        solp = solve(prob);
        prob.f.jac(Jf, solp.u, prob.p);

        # Push to storers
        push!(its, solp.stats.nsteps);
        push!(sols, myNLSoln(solp.u; J=Jf[1:end-1,1:end-1], Jp=Jf[1:end-1,end],
                             save_jacs=save_jacs));
        # Fix New Tangent Sign
        sols[end].dupds .*= sign((tgt./Dsc)'normalize(sols[end].dupds./Dsc));

        # Print out Message
        if verbosity>0
            println(@sprintf("%d. %.2f with step %.4f (%.4f) converged in %d iterations.",
                             length(sols), sols[end].up[end], dss[end]tgt[end], dss[end],
                             its[end]))
        end

        # Step Length Adaptation
        push!(xis, clamp((itopt/its[end])^nxi, xirange[1], xirange[2]));
        push!(dss, clamp(xis[end]dss[end], dsbnds[1], dsbnds[2]));

        # Dynamical Scaling of Dsc
        if DynScale # Rescale
            rat = clamp.(abs.(sols[end].up)./Dsc, 0.5, 2.0);
            Dsc .*= rat;
            Dsc = max.(Dsc, minDsc);

            # Dsc[isapprox.(Dsc, 0.0, atol=eps()^(4//5))] .= 1.0;
        end
    end

    if verbosity>0
        if sols[end].up[end]sign(ps[2]-ps[1])>=ps[2]sign(ps[2]-ps[1])
            display(md"_Parameter End-Point Reached in $(length(sols)) points. Terminating._")
        else
            display(md"_Max. Points Exceeded. Terminating._")
        end
    end

    return sols, its, dss, xis, Dsc;
end
