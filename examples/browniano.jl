# Julia code: Random Walk vs. Brownian Motion Convergence Visualization

using Random
using Plots

function random_walk(n_steps, dt)
    # Generate a random walk with step size sqrt(dt)
    steps = sqrt(dt) .* randn(n_steps)
    return cumsum(steps)
end

function plot_convergence(time_steps, T)
    plot(title="De un Paseo Aleatorio al Movimiento Browniano", xlabel="Tiempo", ylabel="Posición", legend=:topleft)
        gen_dir = joinpath(@__DIR__, "..", "generated")
        mkpath(gen_dir)
    for dt in time_steps
        n_steps = Int(T/dt)
        t = range(0, T, length=n_steps)
        walk = random_walk(n_steps, dt)
        plot!(t, walk, label="dt = $(round(dt, digits=3))")
    end
        savefig(joinpath(gen_dir, "randomwalk_to_brownian.png"))
        println("Guardado ", joinpath(gen_dir, "randomwalk_to_brownian.png"))
end

# Parameters
T = 1.0
time_steps = [0.1, 0.05, 0.01, 0.001]

plot_convergence(time_steps, T)