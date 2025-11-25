# ==========================================
# User Workflow
# ==========================================
include("weight_optimization.jl")

# 2. Load the module (Note the DOT before the name)
using .PortfolioOptimizer
# 1. Configuration (Same as before)
S0_config = [100.0, 100.0, 100.0, 100.0, 100.0]
μ_drift = [0.08, 0.09, 0.07, 0.10, 0.08]
σ_vol = [0.20, 0.25, 0.15, 0.30, 0.22]
Rho = [
    1.0 0.4 0.3 0.2 0.4;
    0.4 1.0 0.3 0.2 0.3;
    0.3 0.3 1.0 0.2 0.2;
    0.2 0.2 0.2 1.0 0.3;
    0.4 0.3 0.2 0.3 1.0
]

# 2. TRAIN THE MODEL (Do this once, it takes a few seconds)
# We simulate 1 year, 12 steps (months)
calibration_paths = 5000
trained_model = calibrate_portfolio_model(S0_config, μ_drift, σ_vol, Rho, 1.0, 12, calibration_paths)

println("\nModel Trained! Ready for manual monthly inputs.\n")

# 3. MONTHLY REBALANCING LOOP (User Input)
# -------------------------------------------------

# Let's pretend we are in Month 1. 
# The user provides NEW prices for the 5 stocks.
# (Example: Market crashed slightly)
new_prices_month_1 = [98.5, 99.0, 101.2, 95.0, 97.0]
step_idx = 1

weights = calculate_rebalancing_weights(trained_model, new_prices_month_1, step_idx)

# Display Results
println("Optimization for Month $step_idx:")
println("Input Prices: $new_prices_month_1")
println("-----------------------------------")
println("Optimal Weights (Top 5 Assets shown):")
for i in 1:5
    println("Stock $i: $(round(weights[i]*100, digits=2))%")
end
# Show first option of Stock 1
println("Stock 1 Put Option (ITM): $(round(weights[6]*100, digits=2))%")
println("Total Leverage: $(round(sum(abs.(weights)), digits=2))")