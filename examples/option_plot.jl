# lsm_option.jl
# GitHub Copilot
#
# Least-Squares Monte Carlo (LSM) for American option pricing (Put/Call)
# Simulates GBM, computes LSM exercise policy, returns price and plots
# evolution of the option value from issuance (t=0) to maturity.
#
# Usage example at the bottom — edit parameters as needed.

using Random
using LinearAlgebra
using Statistics
using Distributions
using Plots

# Simulate GBM paths under risk-neutral drift r
function simulate_gbm(S0, r, σ, T, N, M; seed=1234)
    Random.seed!(seed)
    dt = T / N
    sqrt_dt = sqrt(dt)
    S = zeros(M, N+1)
    S[:,1] .= S0
    for j in 2:N+1
        Z = randn(M)
        S[:,j] = S[:,j-1] .* exp.((r - 0.5*σ^2)*dt .+ σ*sqrt_dt.*Z)
    end
    return S, dt
end

# Build polynomial basis matrix (columns: 1, S, S^2, ..., S^deg)
function poly_basis(Scol, deg)
    M = length(Scol)
    X = ones(M, deg+1)
    for k in 1:deg
        X[:,k+1] = Scol .^ k
    end
    return X
end

# Least-Squares Monte Carlo for American option
# Returns: price_at_t0, times, values_over_time (vector length N+1), exercise_indices, payoff_at_exercise
function lsm_american(S0, K, r, σ, T, N; M=20000, option_type=:put, polydeg=2, seed=1234)
    S, dt = simulate_gbm(S0, r, σ, T, N, M; seed=seed)
    times = (0:N) .* dt

    # payoff matrix
    if option_type == :put
        payoff = max.(K .- S, 0.0)
    elseif option_type == :call
        payoff = max.(S .- K, 0.0)
    else
        error("option_type must be :put or :call")
    end

    # exercise index j in 1:(N+1), initialize to maturity (N+1)
    exercise_idx = fill(N+1, M)
    payoff_at_ex = payoff[:,N+1]  # payoff at exercise time (updated when exercised)

    # Backward induction
    for j in N:-1:1
        active = exercise_idx .> j              # not exercised before time j
        itm = payoff[:,j] .> 0.0                # in-the-money at time j
        idxs = findall(active .& itm)
        if !isempty(idxs)
            # Discount future realized payoff back to time j
            steps_to_ex = exercise_idx[idxs] .- j
            Y = payoff_at_ex[idxs] .* exp.(-r * dt .* steps_to_ex)

            # Basis at time j
            X = poly_basis(S[idxs, j], polydeg)

            # Solve least-squares: coefficients ≈ X \ Y
            coeffs = X \ Y
            cont = X * coeffs

            # Exercise if immediate payoff > continuation estimate
            exercise_now = payoff[idxs, j] .> cont
            if any(exercise_now)
                sel = idxs[exercise_now]
                exercise_idx[sel] .= j
                payoff_at_ex[sel] .= payoff[sel, j]
            end
        end
    end

    # Present value at t=0: discount payoff_at_ex back to time 0
    steps_to_ex0 = exercise_idx .- 1
    discounted_payoff0 = payoff_at_ex .* exp.(-r * dt .* steps_to_ex0)
    price0 = mean(discounted_payoff0)

    # Evolution of option value at each time j (expected remaining value discounted to time j)
    values_over_time = zeros(N+1)
    for j in 1:N+1
        # For each path, if already exercised before j => remaining value = 0
        # else remaining value = payoff_at_ex * exp(-r * dt * (exercise_idx - j))
        still = exercise_idx .>= j
        steps_to_ex_j = max.(0, exercise_idx .- j)
        remaining = payoff_at_ex .* exp.(-r * dt .* steps_to_ex_j) .* (still .== 1)
        values_over_time[j] = mean(remaining)
    end

    return price0, times, values_over_time, exercise_idx, payoff_at_ex
end

# Quick plotting helper
function plot_evolution(times, values; title_text="Evolución del valor de la opción (LSM)", savepath=nothing)
    plt = plot(times, values,
        xlabel="Tiempo",
        ylabel="Valor de la opción (valor esperado restante en el tiempo t)",
        title=title_text,
        legend=false,
        lw=2,
        size=(900,620),        # increase canvas height so labels don't get clipped
        xguidefont=font(9),    # smaller x-axis label font
        yguidefont=font(10),   # y-axis label font
        tickfont=font(8))      # smaller tick labels to give more room
    if savepath !== nothing
        savefig(plt, savepath)
    end
    return plt
end

# new: plot two evolutions (call + put) on the same axes
function plot_two_evolutions(times, values1, values2; labels=("Call","Put"), title_text="Evolución (Call vs Put)", savepath=nothing)
    plt = plot(times, values1,
        label=labels[1],
        xlabel="Tiempo",
        ylabel="Valor de la opción (valor esperado restante en el tiempo t)",
        title=title_text,
        lw=2,
        size=(900,620),
        xguidefont=font(9),
        yguidefont=font(10),
        tickfont=font(8))
    plot!(plt, times, values2, label=labels[2], lw=2, linestyle=:dash)
    # do not call display() to avoid inline rendering padding issues; caller saves the figure
    if savepath !== nothing
        savefig(plt, savepath)
    end
    return plt
end

    # Parameters (change as needed)
    S0 = 446.74                     # initial underlying price
    K = 402.5                      # strike price
    r = 0.03                        # risk-free rate (annual)
    σ = 0.25                        # volatility (annual)
    T = 1.0                         # maturity in years
    N = 50                          # time steps
    M = 500000                      # Monte Carlo paths (increase for accuracy)

    println("Parameters: S0=$(S0), K=$(round(K, digits=2)), r=$(r), σ=$(σ), T=$(T), N=$(N), M=$(M)")

    # compute call and put using the same settings
    price_call, times, values_call, _, _ = lsm_american(S0, K, r, σ, T, N; M=M, option_type=:call, polydeg=3, seed=42)
    price_put,  _,     values_put,  _, _ = lsm_american(S0, K, r, σ, T, N; M=M, option_type=:put,  polydeg=3, seed=43)

    println("Estimated call price at t=0: $(round(price_call, digits=4))")
    println("Estimated put  price at t=0: $(round(price_put,  digits=4))")

    plt = plot_two_evolutions(times, values_call, values_put,
        labels=("Call (LSM)","Put (LSM)"),
        title_text="Evoluciones Call vs Put (K=$(round(K, digits=2)))",
        savepath="lsm_call_put_evolution.png")
    println("Plot saved to lsm_call_put_evolution.png")
