using Random
using Distributions
using Plots

# Parameters
μ = 0.1           # Drift
σ = 0.2           # Volatility
S0 = 100.0        # Initial value
T = 1.0           # Time horizon (years)
N = 252           # Number of time steps
M = 10            # Number of trajectories

dt = T / N
t = range(0, T, length=N+1)

# Function to simulate GBM paths
function simulate_gbm_paths(μ, σ, S0, T, N, M)
    dt = T / N
    S = zeros(N+1, M)
    S[1, :] .= S0
    for m in 1:M
        for n in 1:N
            dW = sqrt(dt) * randn()
            S[n+1, m] = S[n, m] * exp((μ - 0.5*σ^2)*dt + σ*dW)
        end
    end
    return S
end

# Simulate
S = simulate_gbm_paths(μ, σ, S0, T, N, M)

gen_dir = joinpath(@__DIR__, "..", "generated")
mkpath(gen_dir)

# Plot
plot(t, S, legend=false, xlabel="Tiempo", ylabel="S(t)", title="Trayectorias de Movimiento Browniano Geométrico")
savefig(joinpath(gen_dir, "geometric_paths.png"))
println("Guardado ", joinpath(gen_dir, "geometric_paths.png"))