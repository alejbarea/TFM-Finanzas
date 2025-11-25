using Random
using Plots

# Parameters
n_trajectories = 20
n_steps = 100
T = 1.0
dt = T / n_steps
t = range(0, T, length=n_steps+1)

# Generate Brownian trajectories
trajectories = zeros(n_trajectories, n_steps+1)
for i in 1:n_trajectories
    dW = sqrt(dt) * randn(n_steps)
    trajectories[i, 2:end] = cumsum(dW)
end

# Compute Monte Carlo average
mean_trajectory = mean(trajectories, dims=1) |> vec

gen_dir = joinpath(@__DIR__, "..", "generated")
mkpath(gen_dir)

# Gráfico
plt = plot()
for i in 1:n_trajectories
    plot!(plt, t, trajectories[i, :], 
        color=:blue, 
        linestyle=:dash, 
        label=(i==1 ? "Muestra" : ""), 
        linewidth=1.5)
end
plot!(plt, t, mean_trajectory, 
    color=:red, 
    linestyle=:solid, 
    linewidth=3, 
    label="Monte Carlo")

# Etiquetas en español
xlabel!("Tiempo")
ylabel!("Valor")
title!("Trayectorias de Movimiento Browniano y Monte Carlo")

# Forzar límite inferior en y a 0 (si prefieres no cortar valores negativos, quita esta línea)
top = maximum([maximum(trajectories), maximum(mean_trajectory)])


# Mostrar leyenda en la esquina superior derecha y guardar
plot!(plt, legend=:topright)
savefig(joinpath(gen_dir, "brownian_mc.png"))
println("Guardado ", joinpath(gen_dir, "brownian_mc.png"))