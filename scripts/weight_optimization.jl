module PortfolioOptimizer

# 1. IMPORTS (Must be inside the module)
using LinearAlgebra
using Statistics
using Random
using Distributions
using DataFrames
using Polynomials

# 2. EXPORTS
export PortfolioPolicy, calibrate_portfolio_model, calculate_rebalancing_weights

# ==========================================
# Data Structures
# ==========================================
struct PortfolioPolicy
    dt::Float64
    r_rf::Float64 # Calibrated Risk-Free Rate (Annualized)
    strikes::Matrix{Float64}
    lsm_coeffs::Dict{Int, Vector{Vector{Float64}}}
    return_models::Dict{Int, Matrix{Float64}}
    risk_models::Dict{Int, Matrix{Float64}}
end

# ==========================================
# Helper Functions
# ==========================================
basis_features(S_t) = hcat(ones(length(S_t)), S_t, S_t.^2)
prediction_features(S_t) = hcat(ones(size(S_t, 1)), S_t)

# ==========================================
# Calibration (Training) Function
# ==========================================
function calibrate_portfolio_model(S0, μ, σ, Rho, T, N_steps, N_paths)
    println(">>> Phase 1: Starting Model Calibration...")
    dt = T / N_steps
    N_stocks = length(S0)
    
    # A. Generate Training Data
    # ---------------------------------------
    println("   Generating $N_paths training paths...")
    L = cholesky(Rho).L
    paths = zeros(Float64, N_steps + 1, N_paths, N_stocks)
    for i in 1:N_stocks; paths[1, :, i] .= S0[i]; end
    
    for t in 1:N_steps
        Z = randn(N_paths, N_stocks)
        X = Z * L'
        for i in 1:N_stocks
            drift = (μ[i] - 0.5 * σ[i]^2) * dt
            diff = σ[i] * sqrt(dt)
            paths[t+1, :, i] = paths[t, :, i] .* exp.(drift .+ diff .* X[:, i])
        end
    end

    # B. Calibrate Option Pricing (LSM)
    # ---------------------------------------
    println("   Calibrating American Option Pricing Surfaces...")
    Strikes = reshape([0.9, 1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0, 1.1] .* S0[1], N_stocks, 3)
    N_options_total = N_stocks * 3
    
    option_cashflows = zeros(Float64, N_steps + 1, N_paths, N_options_total)
    lsm_coeffs = Dict{Int, Vector{Vector{Float64}}}()
    
    # Terminal Payoff
    for s in 1:N_stocks, (o, K) in enumerate(Strikes[s, :])
        idx = (s-1)*3 + o
        option_cashflows[end, :, idx] .= max.(K .- paths[end, :, s], 0.0)
    end

    for t in N_steps:-1:2
        step_coeffs = Vector{Vector{Float64}}(undef, N_options_total)
        df = exp(-0.02 * dt)
        
        for s in 1:N_stocks
            S_vec = paths[t, :, s]
            for (o, K) in enumerate(Strikes[s, :])
                idx = (s-1)*3 + o
                Y = df .* option_cashflows[t+1, :, idx]
                h_t = max.(K .- S_vec, 0.0)
                itm = findall(h_t .> 0)
                
                beta = zeros(3)
                if !isempty(itm)
                    X_itm = basis_features(S_vec[itm])
                    beta = X_itm \ Y[itm]
                    cont = X_itm * beta
                    exercise = h_t[itm] .> cont
                    
                    current_vals = copy(Y)
                    current_vals[itm[exercise]] = h_t[itm[exercise]]
                    option_cashflows[t, :, idx] = current_vals
                else
                    option_cashflows[t, :, idx] = Y
                end
                step_coeffs[idx] = beta
            end
        end
        lsm_coeffs[t] = step_coeffs
    end

    # C. Calibrate Return & Risk Models
    # ---------------------------------------
    println("   Calibrating Return & Risk Models...")
    return_models = Dict{Int, Matrix{Float64}}()
    risk_models = Dict{Int, Matrix{Float64}}()
    
    for t in 1:(N_steps-1)
        # Using cashflows as proxy for market prices to train return model
        vals_t = option_cashflows[t, :, :]
        vals_t1 = option_cashflows[t+1, :, :]
        
        # --- STABILIZE RETURNS ---
        safe_denom = max.(vals_t, 0.05) 
        R_opt = (vals_t1 .- vals_t) ./ safe_denom
        R_opt = clamp.(R_opt, -1.0, 5.0)
        
        R_stk = (paths[t+1, :, :] .- paths[t, :, :]) ./ paths[t, :, :]
        R_all = hcat(R_stk, R_opt)
        
        Z = prediction_features(paths[t, :, :])
        Gamma = Z \ R_all
        return_models[t] = Gamma
        
        Mu_hat = Z * Gamma
        Eps = R_all - Mu_hat
        Sigma_raw = cov(Eps)
        
        # --- SHRINKAGE ---
        delta = 0.5 
        Target = Diagonal(diag(Sigma_raw))
        Sigma_clean = (1-delta)*Sigma_raw + delta*Target
        risk_models[t] = Sigma_clean
    end
    
    println(">>> Calibration Complete.")
    return PortfolioPolicy(dt, 0.02, Strikes, lsm_coeffs, return_models, risk_models)
end

# ==========================================
# Execution (Rebalancing) Function
# ==========================================
"""
    calculate_rebalancing_weights(model, current_S, t; risk_aversion=3.0, max_leverage=2.0, current_r_rf=nothing)

Calculates optimal portfolio weights for risky assets.
Any weight NOT allocated to risky assets is automatically allocated to the Risk-Free Asset.

- `risk_aversion`: Higher values reduce risky exposure, increasing the Risk-Free Asset allocation.
- `current_r_rf`: (Optional) If provided, overrides the calibrated risk-free rate for this month's decision.
"""
function calculate_rebalancing_weights(model::PortfolioPolicy, current_S::Vector{Float64}, t::Int; 
                                     risk_aversion::Float64=3.0, 
                                     max_leverage::Float64=2.0,
                                     current_r_rf::Union{Float64, Nothing}=nothing)
    N_stocks = length(current_S)
    N_options = N_stocks * 3
    N_total = N_stocks + N_options
    
    # Use current market rate if provided, otherwise model rate
    r_safe = isnothing(current_r_rf) ? model.r_rf : current_r_rf
    
    # Predict Expected Returns
    Z_current = vcat(1.0, current_S)'
    
    if !haskey(model.return_models, t)
        println("Warning: No predictive model for step $t. Allocating 100% to Risk-Free Asset.")
        return zeros(N_total)
    end
    
    Gamma = model.return_models[t]
    Sigma = model.risk_models[t]
    
    μ_hat = vec(Z_current * Gamma)
    # Calculate Excess Returns (Return of Asset - Return of Risk Free)
    er = μ_hat .- (r_safe * model.dt)
    
    # --- OPTIMIZATION SOLVER ---
    w_raw = zeros(N_total)
    try
        # Standard Mean-Variance Solver
        w_raw = Sigma \ er
    catch e
        w_raw = pinv(Sigma) * er
    end
    
    # Clean numerical noise
    replace!(w_raw, NaN => 0.0)
    replace!(w_raw, Inf => 0.0)
    
    # --- APPLY RISK AVERSION ---
    # Formula: w = (1 / γ) * Σ^-1 * (μ - r)
    # The remainder (1 - sum(w)) is implicitly the Risk-Free Asset.
    w_raw = w_raw ./ risk_aversion
    
    # --- CHECK LEVERAGE CAP ---
    leverage = sum(abs.(w_raw))
    if leverage > max_leverage
        w_raw = w_raw .* (max_leverage / leverage)
    elseif leverage == 0.0
        return w_raw
    end
    
    # Debug info (Optional: remove in production)
    # rf_weight = 1.0 - sum(w_raw)
    # println("Allocation :: Risky: $(round(sum(w_raw)*100, digits=1))% | Risk-Free ($r_safe): $(round(rf_weight*100, digits=1))%")
    
    return w_raw
end

end # End of Module