using NonlinearSolve
using NonlinearSolve: ForwardDiff
using LinearAlgebra, Printf
using GLMakie, LaTeXStrings, Revise
using BSON: @save, @load
using CairoMakie  # Required for PDF export

include("./CONTINUATION.jl")

# ---------------------- CONFIGURATION ----------------------
save_figures = false   # Set true to export figure as PDF
save_data     = true   # Set true to save BSON result
analyze       = true   # Set false to skip simulation and load previous results

datafile = "data/a1_FR_continue.bson"
figfile  = "figures/a1_FR_continue.pdf"

# ---------------------- Duffing Residual Function ----------------------
function duffresfun!(uOm, p; du=nothing, J=nothing, Jp=nothing)
    (; m, w, k, a, c, F) = p
    A, B, W = uOm

    if du !== nothing
        du[1] = ((3*B^3 + 3*A^2*B)*a - 4*B*m*W^2 - 4*A*c*W + 4*B*k) / (8*m*W)
        du[2] = -((3*A^3 + 3*B^2*A)*a - 4*A*m*W^2 + 4*B*c*W + 4*A*k - 4*F) / (8*m*W)
    end

    if J !== nothing
        J[1, 1] = (6*A*B*a - 4*c*W) / (8*m*W)
        J[1, 2] = ((9*B^2 + 3*A^2)*a - 4*m*W^2 + 4*k) / (8*m*W)
        J[2, 1] = -((3*B^2 + 9*A^2)*a - 4*m*W^2 + 4*k) / (8*m*W)
        J[2, 2] = -(6*A*B*a + 4*c*W) / (8*m*W)
    end

    if Jp !== nothing
        Jp[1] = (4*A*c)/(8*m*W) - ((3*B^3 + 3*A^2*B)*a - 4*B*m*W^2 - 4*A*c*W + 4*B*k)/(8*m*W^2) + (8*B*m*W)/(8*m*W)
        Jp[2] = -((8*A*m*W - 4*B*c)/(8*m*W) - ((3*A^3 + 3*B^2*A)*a - 4*A*m*W^2 + 4*B*c*W + 4*A*k - 4*F)/(8*m*W^2))
    end

    return nothing
end

# ---------------------- PARAMETERS ----------------------
ξ = 1e3
m = 1.0
w = 2.0
k = m * w^2
a = 0.1 * ξ^2
c = 0.5e-2
F_force = 0.1 / ξ
pars = (m=m, w=w, k=k, a=a, c=c, F=F_force)

Om0 = 0.85 * w
Om1 = 1.65 * w
dOm = 0.01 * w

# Linear guess
Alin = F_force / (k - m*Om0^2 + 2im*c*Om0)
AB0 = [real(Alin), imag(Alin)]

# Setup nonlinear function
funduff = NonlinearFunction((du, u, Om) -> duffresfun!([u; Om], pars; du=du))
funduffJ = NonlinearFunction(
    (du, u, Om) -> duffresfun!([u; Om], pars; du=du),
    jac = (J, u, Om) -> duffresfun!([u; Om], pars; J=J),
    paramjac = (JOm, u, Om) -> duffresfun!([u; Om], pars; Jp=JOm)
)

# ---------------------- CONTINUATION ----------------------
if analyze
    println("Performing continuation from ω = $(Om0) to ω = $(Om1)...")
    sols, its, dss, xis, Dsc = CONTINUATE(AB0, funduff, [Om0, Om1], dOm)
    println("Done. Found $(length(sols)) solution points.")

    if !isempty(sols)
        frequencies = [s.up[end] for s in sols]
        A_vals = [s.up[1] for s in sols]
        B_vals = [s.up[2] for s in sols]
        amplitudes = sqrt.(A_vals.^2 + B_vals.^2)

        if save_data
            @save datafile frequencies A_vals B_vals amplitudes
        end
    else
        println("No solutions found.")
        frequencies, A_vals, B_vals, amplitudes = Float64[], Float64[], Float64[], Float64[]
    end
else
    println("Loading previous results from BSON...")
    @load datafile frequencies A_vals B_vals amplitudes
end

# ---------------------- PLOTTING ----------------------
fig = Figure(fontsize=18)
ax = Axis(fig[1, 1],
    xlabel = "Excitation Frequency (rad/s)",
    ylabel = "Response Amplitude (m)",
    title = "Frequency Response Curve")

if !isempty(frequencies)
    lines!(ax, frequencies, amplitudes,
        color = :blue, linewidth = 3, label = "Analytical (Continuation)")
end

axislegend(ax, position = :rt)

if save_figures
    save(figfile, fig)
end

if isdefined(Main, :GLMakie)
    if !isdefined(Main, :scr2)
        scr2 = GLMakie.Screen()
    end
    display(scr2, fig)
else
    display(fig)
end

# ---------------------- SUMMARY ----------------------
if !isempty(frequencies)
    println("\n=== Solution Summary ===")
    println("Number of points: $(length(frequencies))")
    println("Frequency range: $(round(minimum(frequencies), digits=3)) – $(round(maximum(frequencies), digits=3)) rad/s")
    println("Max amplitude: $(round(maximum(amplitudes), digits=6)) m")
    println("Min amplitude: $(round(minimum(amplitudes), digits=6)) m")
    
    max_idx = argmax(amplitudes)
    println("Resonance at ω = $(round(frequencies[max_idx], digits=3)) rad/s with amplitude = $(round(amplitudes[max_idx], digits=6)) m")
end
