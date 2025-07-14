using DifferentialEquations, LinearAlgebra
using GLMakie, CairoMakie
using BSON: @save, @load

# ---------------------- CONFIGURATION ----------------------
save_figures = false
save_data    = true
analyze      = true  # <- set to true for full re-run at least once

datafile = "data/a2_avg_all.bson"
figfile  = "figures/a2_avg_all.pdf"

save_figures ? CairoMakie.activate!() : GLMakie.activate!()
π
# ---------------------- HELPER FUNCTIONS ----------------------
safe_acos(x) = abs(x) < 1 ? acos(x) : π

function steadystate(p)
    sol = solve(SteadyStateProblem(avglag_old!, [0.0, 0.0], p), SSRootfind())
    return sqrt(sol.u[1]^2 + sol.u[2]^2)
end

function find_w0(p_base)
    ω_range = 0.25:0.01:5.0
    max_amp = zeros(length(ω_range))
    for (i, w) in enumerate(ω_range)
        p_try = [p_base[1:6]... , w]
        max_amp[i] = steadystate(p_try)
    end
    return ω_range[argmax(max_amp)]
end

# ---------------------- SYSTEM DEFINITIONS ----------------------
function jerkins!(du, u, p, t)
    m, c, k, kt, F, fs, Om = p
    du[1] = u[2]
    du[2] = -(c*u[2] + k*u[1] + kt*(u[1]-u[3]) - F*cos(Om*t)) / m
    du[3] = abs(kt*(u[1]-u[3]) * sign(u[2])) <= fs ? 0.0 : u[2]
end

function avglag_new!(du, u, p, t)
    m, c, k, kt, F, fs, Ω, w0 = p
    Uc, Us = u
    τst = safe_acos(1 - (2fs) / (kt * sqrt(Uc^2 + Us^2)))

    du[1] = -((Us*kt*Ω^2*sin(2*τst))/(4*π*Ω^2-4*π*w0^2)-(Us*kt*w0^2*sin(2*τst))/(4*π*Ω^2-4*π*w0^2)-(Uc*kt*Ω^2*cos(2*τst))/(4*π*Ω^2-4*π*w0^2)+(Uc*kt*w0^2*cos(2*τst))/(4*π*Ω^2-4*π*w0^2)-(2*Us*kt*Ω^2*τst)/(4*π*Ω^2-4*π*w0^2)+(2*Us*kt*w0^2*τst)/(4*π*Ω^2-4*π*w0^2)+(2*F*w0^2*cos((2*π*Ω)/w0))/(4*π*Ω^2-4*π*w0^2)+(2*π*Uc*c*w0*Ω^2)/(4*π*Ω^2-4*π*w0^2)+(Uc*kt*Ω^2)/(4*π*Ω^2-4*π*w0^2)-(2*π*Uc*c*w0^3)/(4*π*Ω^2-4*π*w0^2)-(Uc*kt*w0^2)/(4*π*Ω^2-4*π*w0^2)-(2*F*w0^2)/(4*π*Ω^2-4*π*w0^2)+(Us*m*w0^2)/2-(Us*k)/2)/(m*w0)

    du[2] = ((Uc*kt*Ω^2*sin(2*τst))/(4*π*Ω^2-4*π*w0^2)-(Uc*kt*w0^2*sin(2*τst))/(4*π*Ω^2-4*π*w0^2)+(Us*kt*Ω^2*cos(2*τst))/(4*π*Ω^2-4*π*w0^2)-(Us*kt*w0^2*cos(2*τst))/(4*π*Ω^2-4*π*w0^2)-(2*Uc*kt*Ω^2*τst)/(4*π*Ω^2-4*π*w0^2)+(2*Uc*kt*w0^2*τst)/(4*π*Ω^2-4*π*w0^2)+(2*F*w0*Ω*sin((2*π*Ω)/w0))/(4*π*Ω^2-4*π*w0^2)-(2*π*Us*c*w0*Ω^2)/(4*π*Ω^2-4*π*w0^2)-(Us*kt*Ω^2)/(4*π*Ω^2-4*π*w0^2)+(2*π*Us*c*w0^3)/(4*π*Ω^2-4*π*w0^2)+(Us*kt*w0^2)/(4*π*Ω^2-4*π*w0^2)+(Uc*m*w0^2)/2-(Uc*k)/2)/(m*w0)
end

function avglag_old!(du, u, p, t)
    m, c, k, kt, F, fs, Om = p
    Uc, Us = u
    τst = safe_acos(1 - (2fs) / (kt * sqrt(Uc^2 + Us^2)))
    du[1] = -(((Us*kt*sin(2*τst))/(4π) - (Uc*kt*cos(2*τst))/(4π) - (Us*kt*τst)/(2π) + (Us*m*Om^2)/2 + (Uc*c*Om)/2 + (Uc*kt)/(4π) - (Us*k)/2) / (m*Om))
    du[2] =  ((Uc*kt*sin(2*τst))/(4π) + (Us*kt*cos(2*τst))/(4π) - (Uc*kt*τst)/(2π) + (Uc*m*Om^2)/2 - (Us*c*Om)/2 - (Us*kt)/(4π) - (Uc*k)/2 + F/2) / (m*Om)
end

# ---------------------- PARAMETERS ----------------------
m, c, k = 1.0, 0.1, 4.0
kt, F, fs, Om = 5.0, 0.5, 1.0, 2.0

w₀ = find_w0([m, c, k, kt, F, fs, Om])
println(w₀)
p = [m, c, k, kt, F, fs, Om, w₀]

tspan = (0.0, 100.0)
dt = 0.01
tpoints = collect(tspan[1]:dt:tspan[2])
u0_jerkins = [0.01, 0.0, 0.0]
u0_avglag  = [0.01, 0.0]

# ---------------------- SOLVE / LOAD ----------------------
if analyze
    sol_jerkins = solve(ODEProblem(jerkins!, u0_jerkins, tspan, p[1:7]), Tsit5(), saveat=tpoints)
    sol_avglag  = solve(ODEProblem(avglag_new!, u0_avglag, tspan, p), Tsit5(), saveat=tpoints)

    Uc = [u[1] for u in sol_avglag.u]
    Us = [u[2] for u in sol_avglag.u]
    x_avglag = Uc .* cos.(w₀ .* tpoints) .+ Us .* sin.(w₀ .* tpoints)

    x_jerkins = [u[1] for u in sol_jerkins.u]


    if save_data
        @save datafile tpoints x_jerkins dx_jerkins x_avglag dx_avglag w_jerkins friction_num friction_al
    end
else
    @load datafile tpoints x_jerkins dx_jerkins x_avglag dx_avglag w_jerkins friction_num friction_al
end

# ---------------------- PLOTTING ----------------------
fig = Figure(resolution = (1200, 800))

ax1 = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Displacement (m)", title = "Displacement vs Time")
lines!(ax1, tpoints, x_jerkins, color = :blue, label = "Jerkins")
lines!(ax1, tpoints, x_avglag, color = :red, linestyle = :dash, label = "AvgLag")
axislegend(ax1, position = :rb)


if save_figures
    save(figfile, fig)
end

display(fig)
