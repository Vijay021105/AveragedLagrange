using DifferentialEquations, LinearAlgebra
using GLMakie, CairoMakie
using BSON: @save, @load

# ---------------------- CONFIGURATION ----------------------
save_figures = true
save_data     = true
analyze       = true

datafile = "data/c2_avg_all.bson"
figfile  = "figures/c2_avg_all.pdf"

save_figures ? CairoMakie.activate!() : GLMakie.activate!()

# ---------------------- SYSTEM DEFINITIONS ----------------------
function jerkins!(du, u, p, t)
    m, c, k, kt, F, fs, Om = p
    du[1] = u[2]
    du[2] = -(c*u[2] + k*u[1] + kt*(u[1]-u[3]) - F*cos(Om*t)) / m
    du[3] = (kt*(u[1]-u[3]) * sign(u[2])) <= fs ? 0.0 : u[2]
end

function safe_acos(x)
    abs(x) < 1 ? acos(x) : π
end

function avglag!(du, u, p, t)
    m, c, k, kt, F, fs, Om = p
    Uc, Us = u
    τst = safe_acos(1 - (2*fs) / (kt * sqrt(Uc^2 + Us^2)))
    du[1] = -(((Us*kt*sin(2*τst))/(4π) - (Uc*kt*cos(2*τst))/(4π) - (Us*kt*τst)/(2π) + (Us*m*Om^2)/2 + (Uc*c*Om)/2 + (Uc*kt)/(4π) - (Us*k)/2) / (m*Om))
    du[2] =  ((Uc*kt*sin(2*τst))/(4π) + (Us*kt*cos(2*τst))/(4π) - (Uc*kt*τst)/(2π) + (Uc*m*Om^2)/2 - (Us*c*Om)/2 - (Us*kt)/(4π) - (Uc*k)/2 + F/2) / (m*Om)
end

function steadystate(p, t)
    sol = solve(SteadyStateProblem(avglag!, [0.0, 0.0], p), SSRootfind())
    return sol.u[1] * cos(p[7] * t) + sol.u[2] * sin(p[7] * t)
end

# ---------------------- PARAMETERS ----------------------
m, c, k = 1.0, 0.1, 4.0
kt = 5.0
F, fs, Om = 0.5, 1.0, 0.5
p = [m, c, k, kt, F, fs, Om]

tspan = (0.0, 100.0)
dt = 0.01
tpoints = collect(tspan[1]:dt:tspan[2])

u0_jerkins = [0.01, 0.0, 0.0]
u0_avglag  = [0.01, 0.0]

# ---------------------- SOLVE / LOAD ----------------------
if analyze
    sol_jerkins = solve(ODEProblem(jerkins!, u0_jerkins, tspan, p), Tsit5(), saveat=tpoints)
    sol_avglag  = solve(ODEProblem(avglag!,  u0_avglag,  tspan, p), Tsit5(), saveat=tpoints)

    Uc = [u[1] for u in sol_avglag.u]
    Us = [u[2] for u in sol_avglag.u]
    U_amp = sqrt.((Uc).^2 .+ (Us).^2)
    x_avglag = Uc .* cos.(Om .* tpoints) .+ Us .* sin.(Om .* tpoints)
    dx_avglag = -(Uc .* Om .*sin.(Om.*tpoints)) + dUs.*sin.(Om.*tpoints) + Us.*Om.*cos.(Om.*tpoints) +dUc.*cos.(Om.*tpoints)

    x_jerkins = [u[1] for u in sol_jerkins.u]
    dx_jerkins = [u[2] for u in sol_jerkins.u]
    w_jerkins = [u[3] for u in sol_jerkins.u]

    friction_num = kt .* (x_jerkins .- w_jerkins)
    τst = safe_acos.(1 .- (2fs) ./ (kt .* sqrt.(Uc.^2 .+ Us.^2)))
    Fc = kt .* (Uc .* (2 .* τst .- sin.(2 .* τst)) .+ Us .* (1 .- cos.(2 .* τst))) ./ (2π)
    Fs = kt .* (-Uc .* (1 .- cos.(2 .* τst)) .+ Us .* (2 .* τst .- sin.(2 .* τst))) ./ (2π)
    friction_al = Fc .* cos.(Om .* tpoints) .+ Fs .* sin.(Om .* tpoints)

    if save_data
        @save datafile tpoints x_jerkins dx_jerkins x_avglag dx_avglag w_jerkins friction_num friction_al
    end
else
    @load datafile tpoints x_jerkins dx_jerkins x_avglag dx_avglag w_jerkins friction_num friction_al
end

# ---------------------- PLOTTING ----------------------
fig = Figure(resolution = (1200, 800))

# fig[1, 1] - Displacement
ax1 = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Displacement (m)", title = "Displacement vs Time")
lines!(ax1, tpoints, x_jerkins, color = :blue, label = "Numerical")
lines!(ax1, tpoints, x_avglag, color = :red, linestyle = :dash, label = "AvgLag")

# fig[1, 2] - Velocity
ax2 = Axis(fig[1, 2], xlabel = "Time (s)", ylabel = "Velocity (m/s)", title = "Velocity vs Time")
lines!(ax2, tpoints, dx_jerkins, color = :blue, label = "Numerical")
lines!(ax2, tpoints, dx_avglag, color = :red, linestyle = :dash, label = "AvgLag")

# fig[2, 1] - Slider displacement (w)
ax3 = Axis(fig[2, 1], xlabel = "Time (s)", ylabel = "Slider Displacement w (m)", title = "Slider Position")
lines!(ax3, tpoints, w_jerkins, color = :green, label = "w (Numerical)")

# fig[2, 2] - Friction force vs displacement
ax4 = Axis(fig[2, 2], xlabel = "Displacement (u)", ylabel = "Frictional Force (kt(u-w))",
           title = "Frictional Force vs Displacement")
lines!(ax4, x_avglag, friction_al, color = :red,linestyle = :dot ,label = "AvgLag", linewidth=2)
lines!(ax4, x_jerkins, friction_num, color = :blue, label = "Numerical", linewidth=2)

# Legends
axislegend(ax1, position = :rb)
axislegend(ax2, position = :rb)
axislegend(ax3, position = :rb)
axislegend(ax4, position = :rt)

if save_figures
    save(figfile, fig)
end

display(fig)
