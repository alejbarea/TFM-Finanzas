module TFMFinanzas

"""Basic financial mathematics helpers and small plotting utilities.

Exports:
- bond_price
- black_scholes_price
- mc_option_price
- plot_yield_curve
- plot_option_vs_strike
- plot_mc_paths
"""

using Distributions
using Random
using Plots

export bond_price, black_scholes_price, mc_option_price,
       plot_yield_curve, plot_option_vs_strike, plot_mc_paths

"""Compute the present price of a fixed-rate bond.

Arguments
- face: Face/par value (e.g., 1000)
- coupon_rate: Annual coupon rate (decimal, e.g., 0.05)
- ytm: Annual yield-to-maturity (decimal)
- n_periods: number of years to maturity
- freq: coupons per year (default 2)
"""
function bond_price(face::Real, coupon_rate::Real, ytm::Real, n_periods::Int; freq::Int=2)
    c = coupon_rate * face / freq
    r = ytm / freq
    n = n_periods * freq
    pv_coupons = sum(c / (1 + r)^t for t in 1:n)
    pv_face = face / (1 + r)^n
    return pv_coupons + pv_face
end

"""Black-Scholes European option price (call or put).

S: spot price
K: strike
r: risk-free rate (annual)
sigma: volatility (annual)
T: time to maturity (years)
option: :call or :put
"""
function black_scholes_price(S::Real, K::Real, r::Real, sigma::Real, T::Real; option::Symbol=:call)
    if T <= 0
        payoff = max(option==:call ? S - K : K - S, zero(S))
        return payoff
    end
    d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    nd = Normal()
    if option == :call
        return S * cdf(nd, d1) - K * exp(-r*T) * cdf(nd, d2)
    else
        return K * exp(-r*T) * cdf(nd, -d2) - S * cdf(nd, -d1)
    end
end

"""Simple Monte Carlo European option pricing (geometric Brownian motion).

npaths: number of simulated paths (default 10000)
seed: optional RNG seed for reproducibility
"""
function mc_option_price(S0::Real, K::Real, r::Real, sigma::Real, T::Real; npaths::Int=10000, seed::Union{Nothing,Int}=nothing, option::Symbol=:call)
    if seed !== nothing
        Random.seed!(seed)
    end
    dt = T
    Z = randn(npaths)
    ST = S0 .* exp.((r - 0.5*sigma^2)*dt .+ sigma*sqrt(dt).*Z)
    if option == :call
        payoffs = max.(ST .- K, 0)
    else
        payoffs = max.(K .- ST, 0)
    end
    return exp(-r*T) * mean(payoffs)
end

"""Plot a yield curve for a set of maturities and yields.
Returns the plot object.
"""
function plot_yield_curve(maturities::AbstractVector, yields::AbstractVector; title::String="Curva de rendimientos")
    p = plot(maturities, yields, marker=:circle, xlabel="Madurez (años)", ylabel="Rendimiento", title=title, legend=false)
    # Forzar límite inferior en y a 0
    ylims!(p, 0, maximum(yields) * 1.05)
    return p
end

"""Plot option price vs strike for given parameters."""
function plot_option_vs_strike(S::Real, Ks::AbstractVector, r::Real, sigma::Real, T::Real; option::Symbol=:call, title::String="Precio de opción vs Strike")
    prices = [black_scholes_price(S, K, r, sigma, T; option=option) for K in Ks]
    p = plot(Ks, prices, xlabel="Strike", ylabel="Precio", title=title, marker=:auto)
    # Forzar límite inferior en y a 0
    ylims!(p, 0, maximum(prices) * 1.05)
    return p
end

"""Simulate a few Monte Carlo GBM paths and return a plot object."""
function plot_mc_paths(S0::Real, r::Real, sigma::Real, T::Real; npaths::Int=10, steps::Int=100, seed::Union{Nothing,Int}=nothing, title::String="MC paths")
    if seed !== nothing
        Random.seed!(seed)
    end
    dt = T / steps
    t = collect(0:dt:T)
    paths = zeros(npaths, length(t))
    for i in 1:npaths
        paths[i,1] = S0
        for j in 2:length(t)
            Z = randn()
            paths[i,j] = paths[i,j-1] * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        end
    end
    p = plot()
    for i in 1:npaths
        plot!(p, t, paths[i,:], label = i == 1 ? "paths" : "", alpha=0.7)
    end
    xlabel!(p, "Tiempo")
    ylabel!(p, "Precio")
    title!(p, title)
    # Forzar límite inferior en y a 0
    ylims!(p, 0, maximum(paths) * 1.05)
    return p
end

end # module
