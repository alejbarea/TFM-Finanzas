using Random
using Statistics
using LinearAlgebra
using Distributions
using DataFrames
using Dates
using CSV
using JuMP
using Ipopt
using Printf
using Plots

# Use GR backend for plotting
gr()

# ==========================================
# 1. STRUCTS AND CONFIGURATION
# ==========================================

struct Stock
    id::Int
    current_price::Float64
    mu::Float64
    sigma::Float64
end

struct Option
    id::String
    underlying_idx::Int
    strike::Float64
    is_call::Bool
    maturity::Float64 # in years
    held::Bool        # false if already exercised
end

# Simulation Constants
const T_YEAR = 1.0
const N_MONTHS = 12
const DT = T_YEAR / N_MONTHS
const RISK_FREE_R = 0.0475  # 4.75% anual
const N_PATHS = 2000     
const BASIS_DEGREE = 3   

# Constraints for Optimization
const MAX_OPTION_ALLOCATION = 0.10  # Max 10% del portafolio en opciones
const MAX_SINGLE_ASSET = 0.30       # Max 30% por activo

if !isdefined(@__MODULE__, :MONTH_MAP)
    const MONTH_MAP = Dict(
        "Ene" => 1, "Feb" => 2, "Mar" => 3, "Abr" => 4, "May" => 5, "Jun" => 6,
        "Jul" => 7, "Ago" => 8, "Sep" => 9, "Oct" => 10, "Nov" => 11, "Dic" => 12
    )
end

# ==========================================
# 2. DATA LOADERS
# ==========================================

function load_nasdaq_monthly_returns(path::AbstractString)
    df = CSV.read(path, DataFrame)
    raw_names = names(df)
    cleaned = Dict{Symbol,Symbol}()
    for n in raw_names
        key = Symbol(replace(lowercase(String(n)), " " => "", "/" => "_", "\ufeff" => "", "á" => "a", "é" => "e", "í" => "i", "ó" => "o", "ú" => "u", "ñ" => "n"))
        cleaned[key] = Symbol(n)
    end
    rename!(df, Dict(values(cleaned) .=> keys(cleaned)))
    if !haskey(cleaned, :fecha) || !haskey(cleaned, Symbol("cerrar_ultimo"))
        error("Faltan columnas fecha o cerrar_ultimo en NASDAQ.csv")
    end
    df.fecha = Date.(df.fecha, dateformat"mm/dd/yyyy")
    sort!(df, :fecha)
    years = year.(df.fecha)
    year_target = maximum(years)
    month_end_price = Dict{Int,Float64}()
    prev_dec_key = 0
    for row in eachrow(df)
        y = year(row.fecha)
        m = month(row.fecha)
        if y == year_target
            month_end_price[m] = row.cerrar_ultimo
        elseif y == year_target - 1 && m == 12
            month_end_price[prev_dec_key] = row.cerrar_ultimo
        end
    end
    rets = fill(0.0, 12)
    for m in 1:12
        prev_key = (m == 1) ? prev_dec_key : m - 1
        prev_p = get(month_end_price, prev_key, NaN)
        curr_p = get(month_end_price, m, NaN)
        if !isnan(prev_p) && !isnan(curr_p) && prev_p != 0
            rets[m] = (curr_p - prev_p) / prev_p
        else
            rets[m] = 0.0
        end
    end
    return rets
end

function load_asset_monthly_stats(path::AbstractString)
    df = CSV.read(path, DataFrame)
    raw_names = names(df)
    cleaned = Dict{Symbol,Symbol}()
    for n in raw_names
        key = Symbol(replace(strip(lowercase(String(n))), '\ufeff' => ""))
        cleaned[key] = Symbol(n)
    end
    required_cols = [:activo, :mes, :media_anualizada, :volatilidad_anualizada, :precio_cierre_anchor]
    missing = [c for c in required_cols if !haskey(cleaned, c)]
    if !isempty(missing)
        error("Missing columns $(missing) in stats file: $path")
    end
    rename!(df, Dict(values(cleaned) .=> keys(cleaned)))
    activos = unique(df.activo)
    stats = Dict{String, NamedTuple}()
    for a in activos
        prices = fill(NaN, 12)
        mus = fill(NaN, 12)
        sigmas = fill(NaN, 12)
        sdf = filter(row -> row.activo == a, df)
        for row in eachrow(sdf)
            m_idx = get(MONTH_MAP, String(row.mes), nothing)
            if m_idx === nothing || m_idx < 1 || m_idx > 12
                continue
            end
            prices[m_idx] = Float64(row.precio_cierre_anchor)
            mus[m_idx] = Float64(row.media_anualizada)
            sigmas[m_idx] = Float64(row.volatilidad_anualizada)
        end
        stats[String(a)] = (prices = prices, mus = mus, sigmas = sigmas)
    end
    return stats
end

function load_monthly_covariances(base_dir::AbstractString)
    covs = Dict{Int, Matrix{Float64}}()
    for m in 1:12
        fname = @sprintf("covarianzas_2024_%02d.csv", m)
        fpath = joinpath(base_dir, fname)
        if !isfile(fpath)
            continue
        end
        df = CSV.read(fpath, DataFrame)
        assets = df.asset
        n = length(assets)
        mat = Matrix{Float64}(undef, n, n)
        for (j, name) in enumerate(assets)
            mat[:, j] = Float64.(df[!, name])
        end
        covs[m] = mat
    end
    return covs
end

function resolve_asset_key(name::AbstractString, asset_keys::Vector{String})
    up = uppercase(strip(name))
    mapping = Dict(
        "NVIDIA" => "NVDA", "TESLA" => "TSLA", "PALANTIR" => "PLTR",
        "SENTINELONE" => "S", "MICROSTRATEGY" => "MSTR"
    )
    target = haskey(mapping, up) ? mapping[up] : up
    idx = findfirst(x -> uppercase(x) == target, asset_keys)
    if idx === nothing
        error("No se encontro el activo '" * name * "' en asset_keys")
    end
    return asset_keys[idx], idx
end

function load_options_from_csv(path::AbstractString, asset_keys::Vector{String})
    df = CSV.read(path, DataFrame)
    raw_names = names(df)
    cleaned = Dict{Symbol,Symbol}()
    for n in raw_names
        key = Symbol(replace(lowercase(String(n)), " " => "_", "/" => "_", "\ufeff" => "", "á" => "a", "é" => "e", "í" => "i", "ó" => "o", "ú" => "u", "ñ" => "n"))
        cleaned[key] = Symbol(n)
    end
    rename!(df, Dict(values(cleaned) .=> keys(cleaned)))
    opts = Option[]
    for row in eachrow(df)
        asset_name, idx = resolve_asset_key(row.asset, asset_keys)
        t = lowercase(strip(row.type_of_option))
        is_call = occursin("call", t)
        cat = occursin("atm", t) ? "CALL_ATM" : (occursin("otm", t) ? "CALL_OTM" : (occursin("itm", t) && occursin("put", t) ? "PUT_ITM" : (is_call ? "CALL" : "PUT")))
        strike_val = Float64(row.strike)
        opt_id = string(cat, "_", asset_name)
        push!(opts, Option(opt_id, idx, strike_val, is_call, T_YEAR, true))
    end
    return opts
end

function choose_value(vals::Vector{Float64}, idx::Int)
    if idx <= length(vals) && !isnan(vals[idx]) return vals[idx] end
    for v in vals
        if !isnan(v) return v end
    end
    return 0.0
end

function apply_month_snapshot!(stocks::Vector{Stock}, stats::Dict{String,NamedTuple}, asset_keys::Vector{String}, month_idx::Int)
    for i in 1:length(stocks)
        key = asset_keys[i]
        entry = stats[key]
        price = choose_value(entry.prices, month_idx)
        mu = choose_value(entry.mus, month_idx)
        sigma = choose_value(entry.sigmas, month_idx)
        stocks[i] = Stock(stocks[i].id, price, mu, sigma)
    end
end

# ==========================================
# 3. HELPER FUNCTIONS: SIMULATION & MATH
# ==========================================

function simulate_gbm(stocks::Vector{Stock}, T::Float64, dt::Float64, n_paths::Int; cov_matrix::Union{Matrix{Float64},Nothing}=nothing)
    n_steps = Int(round(T / dt))
    if n_steps == 0 return zeros(1, n_paths, length(stocks)) end
    
    n_stocks = length(stocks)
    paths = zeros(Float64, n_steps + 1, n_paths, n_stocks)
    L = nothing
    if cov_matrix !== nothing
        L = cholesky(Symmetric(cov_matrix)).L
    end
    
    for (i, s) in enumerate(stocks)
        paths[1, :, i] .= s.current_price
    end
    for t in 1:n_steps
        if cov_matrix === nothing
            for i in 1:n_stocks
                s = stocks[i]
                drift = (s.mu - 0.5 * s.sigma^2) * dt
                vol = s.sigma * sqrt(dt)
                z = randn(n_paths)
                paths[t+1, :, i] = paths[t, :, i] .* exp.(drift .+ vol .* z)
            end
        else
            z = randn(n_paths, n_stocks) * L'
            for i in 1:n_stocks
                s = stocks[i]
                drift = (s.mu - 0.5 * s.sigma^2) * dt
                vol = s.sigma * sqrt(dt)
                paths[t+1, :, i] = paths[t, :, i] .* exp.(drift .+ vol .* z[:, i])
            end
        end
    end
    return paths
end

function payoff(price::Float64, strike::Float64, is_call::Bool)
    return is_call ? max(price - strike, 0.0) : max(strike - price, 0.0)
end

function bs_price(S, K, T, r, sigma, is_call)
    if T <= 0 return payoff(S, K, is_call) end
    d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if is_call
        return S * cdf(Normal(), d1) - K * exp(-r*T) * cdf(Normal(), d2)
    else
        return K * exp(-r*T) * cdf(Normal(), -d2) - S * cdf(Normal(), -d1)
    end
end

function bs_delta(S, K, T, r, sigma, is_call)
    if T <= 0 return is_call ? (S>K ? 1.0 : 0.0) : (S<K ? -1.0 : 0.0) end
    d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    return is_call ? cdf(Normal(), d1) : cdf(Normal(), d1) - 1.0
end

# ==========================================
# 4. HELPER FUNCTIONS: LSM VALUATION
# ==========================================

function lsm_valuation_data(option::Option, stock_paths::Matrix{Float64}, dt::Float64, r::Float64)
    n_steps, n_paths = size(stock_paths)
    cashflows = zeros(Float64, n_paths)
    
    for p in 1:n_paths
        cashflows[p] = payoff(stock_paths[end, p], option.strike, option.is_call)
    end
    
    plot_x_prices = Float64[]
    plot_y_continuation = Float64[]
    plot_y_intrinsic = Float64[]
    
    for t in (n_steps-1):-1:2
        prices_t = stock_paths[t, :]
        itm_indices = findall(x -> payoff(x, option.strike, option.is_call) > 0, prices_t)
        
        if isempty(itm_indices)
            cashflows .*= exp(-r * dt)
            continue
        end
        
        X = prices_t[itm_indices]
        Y = cashflows[itm_indices] .* exp(-r * dt)
        
        A = zeros(length(X), BASIS_DEGREE + 1)
        for d in 0:BASIS_DEGREE
            A[:, d+1] = X .^ d
        end
        
        beta = A \ Y
        continuation_values = A * beta
        exercise_values = payoff.(X, option.strike, option.is_call)
        
        if t == Int(round(n_steps / 2))
             plot_x_prices = X
             plot_y_continuation = continuation_values
             plot_y_intrinsic = exercise_values
        end

        for (i, idx) in enumerate(itm_indices)
            if exercise_values[i] > continuation_values[i]
                cashflows[idx] = exercise_values[i]
            else
                cashflows[idx] *= exp(-r * dt)
            end
        end
        
        non_itm = setdiff(1:n_paths, itm_indices)
        cashflows[non_itm] .*= exp(-r * dt)
    end
    
    present_values = cashflows .* exp(-r * dt)
    return mean(present_values), plot_x_prices, plot_y_continuation, plot_y_intrinsic
end

# ==========================================
# 5. OPTIMIZATION (UPDATED: TRACE FRONTIER)
# ==========================================

function optimize_portfolio(stocks::Vector{Stock}, options::Vector{Option}, r::Float64, dt::Float64, cov_override::Union{Matrix{Float64},Nothing}=nothing, generate_frontier::Bool=false)
    
    n_sim_paths = 1000
    paths_next = simulate_gbm(stocks, dt, dt, n_sim_paths; cov_matrix=cov_override)
    
    active_assets = String[]
    returns_matrix = zeros(n_sim_paths, 0)
    expected_returns = Float64[]
    
    # 1. Stocks
    for (i, s) in enumerate(stocks)
        push!(active_assets, "Stock_$(s.id)")
        R = (paths_next[2, :, i] ./ s.current_price) .- 1.0
        returns_matrix = hcat(returns_matrix, R)
        push!(expected_returns, exp((s.mu) * dt) - 1)
    end
    
    # 2. Options
    for opt in options
        if opt.held
            S_now = stocks[opt.underlying_idx].current_price
            S_next = paths_next[2, :, opt.underlying_idx]
            
            T_rem = max(0.001, opt.maturity - dt)
            V_next = [bs_price(s, opt.strike, T_rem, r, stocks[opt.underlying_idx].sigma, opt.is_call) for s in S_next]
            
            V_now = bs_price(S_now, opt.strike, opt.maturity, r, stocks[opt.underlying_idx].sigma, opt.is_call)
            if V_now < 0.01 V_now = 0.01 end
            
            R_opt = (V_next .- V_now) ./ V_now
            returns_matrix = hcat(returns_matrix, R_opt)
            push!(active_assets, opt.id)
            push!(expected_returns, -0.01) # Penalizacion theta
        end
    end
    
    n_assets = length(active_assets)
    
    # Covariance Construction
    if cov_override !== nothing
        n_stocks = length(stocks)
        if size(cov_override, 1) == n_stocks && size(cov_override, 2) == n_stocks
            n_opts = count(opt -> opt.held, options)
            cov_matrix = zeros(n_assets, n_assets)
            cov_matrix[1:n_stocks, 1:n_stocks] .= cov_override
            if n_opts > 0
                opt_var = 0.10^2
                for k in 1:n_opts
                    idx = n_stocks + k
                    cov_matrix[idx, idx] = opt_var
                end
            end
        else
            cov_matrix = Matrix(cov(returns_matrix))
        end
    else
        cov_matrix = Matrix(cov(returns_matrix))
    end

    if cov_matrix isa Number
        cov_matrix = fill(cov_matrix, n_assets, n_assets)
    elseif cov_matrix isa UniformScaling
        cov_matrix = fill(cov_matrix.λ, n_assets, n_assets)
    end
    cov_matrix = Matrix(cov_matrix)
    cov_matrix[diagind(cov_matrix)] .+= 1e-6
    
    # Identify Option Indices for Constraints
    is_option_idx = [!occursin("Stock", name) for name in active_assets]
    option_indices = findall(is_option_idx)
    
    excess_mu = expected_returns .- (r * dt)

    # --- MAIN MODEL (Find Max Sharpe) ---
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, w[1:n_assets] >= 0)
    
    @constraint(model, sum(w) <= 1.0)
    
    if !isempty(option_indices)
        @constraint(model, sum(w[i] for i in option_indices) <= MAX_OPTION_ALLOCATION)
    end

    for i in 1:n_assets
        @constraint(model, w[i] <= MAX_SINGLE_ASSET)
    end
    
    @expression(model, port_ret, sum(w[i] * excess_mu[i] for i in 1:n_assets))
    @expression(model, port_var, sum(w[i] * cov_matrix[i,j] * w[j] for i in 1:n_assets, j in 1:n_assets))
    
    @NLobjective(model, Max, port_ret / (sqrt(port_var) + 1e-6))
    
    optimize!(model)
    opt_weights = value.(w)
    
    # --- FRONTIER GENERATION (Trace the line) ---
    frontier_x = Float64[]
    frontier_y = Float64[]
    
    if generate_frontier
        # Start from min excess return to max excess return
        min_exc = minimum(excess_mu)
        max_exc = maximum(excess_mu)
        
        # Create grid of target returns
        targets = range(min_exc, max_exc, length=30)
        
        for t_ret in targets
            f_model = Model(Ipopt.Optimizer)
            set_silent(f_model)
            @variable(f_model, fw[1:n_assets] >= 0)
            
            # Constraints (same as main)
            @constraint(f_model, sum(fw) <= 1.0)
            if !isempty(option_indices)
                @constraint(f_model, sum(fw[i] for i in option_indices) <= MAX_OPTION_ALLOCATION)
            end
            for i in 1:n_assets
                @constraint(f_model, fw[i] <= MAX_SINGLE_ASSET)
            end
            
            # Force target return
            @constraint(f_model, sum(fw[i] * excess_mu[i] for i in 1:n_assets) >= t_ret)
            
            # Minimize Variance
            @expression(f_model, f_var, sum(fw[i] * cov_matrix[i,j] * fw[j] for i in 1:n_assets, j in 1:n_assets))
            @objective(f_model, Min, f_var)
            
            optimize!(f_model)
            
            if termination_status(f_model) == MOI.LOCALLY_SOLVED || termination_status(f_model) == MOI.OPTIMAL
                fw_val = value.(fw)
                p_std = sqrt(value(f_var))
                # Store full expected return (Excess + Rf)
                p_mu_cons = sum(fw_val .* excess_mu) + r*dt
                
                push!(frontier_x, p_std)
                push!(frontier_y, p_mu_cons)
            end
        end
        
        opt_mu = sum(opt_weights .* excess_mu) + r*dt
        opt_std = sqrt(sum(opt_weights[i] * cov_matrix[i,j] * opt_weights[j] for i in 1:n_assets, j in 1:n_assets))
        
        # Sort for clean line plot
        perm = sortperm(frontier_x)
        return active_assets, opt_weights, (frontier_x[perm], frontier_y[perm], opt_std, opt_mu)
    end
    
    return active_assets, opt_weights, nothing
end

# ==========================================
# 6. MAIN LOOP
# ==========================================

function run_strategy(; rng_seed::Union{Int,Nothing}=nothing)
    if rng_seed !== nothing
        Random.seed!(rng_seed)
    end
    println("--- Starting Simulation ---")
    
    stats_path = joinpath(@__DIR__, "..", "generated", "estadisticas_mensuales_2024.csv")
    cov_dir = joinpath(@__DIR__, "..", "generated")
    nasdaq_path = joinpath(@__DIR__, "..", "data", "NASDAQ.csv")
    if !isfile(stats_path)
        error("No se encontro el archivo de estadisticas: $stats_path")
    end
    stats = load_asset_monthly_stats(stats_path)
    covs = load_monthly_covariances(cov_dir)
    nasdaq_rets = isfile(nasdaq_path) ? load_nasdaq_monthly_returns(nasdaq_path) : fill(0.0, 12)
    asset_keys = sort(collect(keys(stats)))
    
    # Initialize "Reality"
    stocks_reality = [Stock(i, 0.0, 0.0, 0.0) for (i, _) in enumerate(asset_keys)]
    apply_month_snapshot!(stocks_reality, stats, asset_keys, 1)

    opt_path = joinpath(@__DIR__, "..", "data", "options.csv")
    options = (isfile(opt_path) && !isempty(stocks_reality)) ? load_options_from_csv(opt_path, asset_keys) : Option[]
    opt_start = fill(0, length(options))
    opt_end = fill(N_MONTHS, length(options))
    opt_exercised = fill(false, length(options))
    
    wealth_algo = 500000.0
    wealth_benchmark = 500000.0
    
    hist_months = 0:N_MONTHS
    hist_wealth_algo = [wealth_algo]
    hist_wealth_bench = [wealth_benchmark]
    hist_delta = [0.0] 
    
    asset_names_map = vcat("Libre de Riesgo", ["Stock_$(i)" for i in 1:length(stocks_reality)], [opt.id for opt in options])
    hist_weights = zeros(N_MONTHS+1, length(asset_names_map))
    hist_weights[1, 1] = 1.0 
    
    frontier_data_por_mes = Dict{Int,Tuple{Vector{Float64},Vector{Float64},Float64,Float64}}()
    lsm_plot_data_por_mes = Dict{Int, Vector{Tuple{Vector{Float64},Vector{Float64},Vector{Float64},String}}}()

    # --- Loop ---
    for month in 0:(N_MONTHS-1)
        curr_t = month * DT
        rem_t = T_YEAR - curr_t

        month_idx = min(month + 1, 12)
        
        # Beliefs
        stocks_belief = deepcopy(stocks_reality)
        prev_idx = max(month_idx - 1, 1)
        apply_month_snapshot!(stocks_belief, stats, asset_keys, prev_idx)
        
        # Reality
        apply_month_snapshot!(stocks_reality, stats, asset_keys, month_idx)
        
        # Benchmark
        if month > 0
            prev_month_idx = month_idx - 1
            naive_returns = Float64[]
            for key in asset_keys
                entry = stats[key]
                price_prev = choose_value(entry.prices, prev_month_idx)
                price_curr = choose_value(entry.prices, month_idx)
                if price_prev != 0 && !isnan(price_prev) && !isnan(price_curr)
                    push!(naive_returns, (price_curr - price_prev) / price_prev)
                else
                    push!(naive_returns, 0.0)
                end
            end
            if !isempty(naive_returns)
                avg_return = sum(naive_returns) / length(naive_returns)
                wealth_benchmark = wealth_benchmark * (1.0 + avg_return)
            end
        end
        
        # LSM Valuation
        sim_paths = simulate_gbm(stocks_reality, rem_t, DT, N_PATHS)
        for (i, opt) in enumerate(options)
            if !opt.held continue end
            u_paths = sim_paths[:, :, opt.underlying_idx]
            val, lx, ly_cont, ly_int = lsm_valuation_data(opt, u_paths, DT, RISK_FREE_R)
            
            push!(get!(lsm_plot_data_por_mes, month_idx, Vector{Tuple{Vector{Float64},Vector{Float64},Vector{Float64},String}}()), (lx, ly_cont, ly_int, opt.id))
            
            intrinsic = payoff(stocks_reality[opt.underlying_idx].current_price, opt.strike, opt.is_call)
            if intrinsic > val && intrinsic > 0
                options[i] = Option(opt.id, opt.underlying_idx, opt.strike, opt.is_call, opt.maturity, false)
                opt_end[i] = month + 1
                opt_exercised[i] = true
            end
        end
        
        # Optimize
        is_snapshot_month = true 
        cov_override = (month > 0 && haskey(covs, month_idx - 1)) ? covs[month_idx - 1] : nothing
        
        active_assets, weights, f_data = optimize_portfolio(stocks_belief, options, RISK_FREE_R, DT, cov_override, is_snapshot_month)
        
        if is_snapshot_month && f_data !== nothing
            frontier_data_por_mes[month_idx] = f_data
        end
        
        # Record Weights
        current_port_delta = 0.0
        rf_w = 1.0 - sum(weights)
        hist_weights[month+2, 1] = rf_w 
        
        for (i, name) in enumerate(active_assets)
            col_idx = findfirst(x -> x == name, asset_names_map)
            if col_idx !== nothing
                hist_weights[month+2, col_idx] = weights[i]
            end
            if occursin("Stock", name)
                current_port_delta += weights[i] * 1.0
            else
                opt_obj = filter(x -> x.id == name, options)[1]
                stk = stocks_reality[opt_obj.underlying_idx]
                d = bs_delta(stk.current_price, opt_obj.strike, rem_t, RISK_FREE_R, stk.sigma, opt_obj.is_call)
                current_port_delta += weights[i] * d * (stk.current_price / wealth_algo) 
            end
        end
        push!(hist_delta, current_port_delta)
        
        # Step Forward
        portfolio_return_factor = 0.0

        for (i, name) in enumerate(active_assets)
            weight = weights[i]
            if weight <= 1e-5 continue end

            if occursin("Stock", name)
                s_id = parse(Int, split(name, "_")[2])
                entry = stats[asset_keys[s_id]]
                price_prev = choose_value(entry.prices, month_idx)
                price_curr = choose_value(entry.prices, min(month_idx + 1, 12))
                if price_prev != 0 && !isnan(price_prev) && !isnan(price_curr)
                    r_asset = (price_curr - price_prev) / price_prev
                    portfolio_return_factor += weight * r_asset
                end
            else
                opt_obj = filter(x -> x.id == name, options)[1]
                s_id = opt_obj.underlying_idx
                entry = stats[asset_keys[s_id]]
                price_prev = choose_value(entry.prices, month_idx)
                price_curr = choose_value(entry.prices, min(month_idx + 1, 12))
                if price_prev != 0 && !isnan(price_prev) && !isnan(price_curr)
                    p_opt_old = bs_price(price_prev, opt_obj.strike, rem_t, RISK_FREE_R, stocks_reality[s_id].sigma, opt_obj.is_call)
                    if p_opt_old < 0.01 p_opt_old = 0.01 end
                    T_new = (month_idx == 12) ? 0.0 : max(0.0001, rem_t - DT)
                    p_opt_new = bs_price(price_curr, opt_obj.strike, T_new, RISK_FREE_R, stocks_reality[s_id].sigma, opt_obj.is_call)
                    r_asset = (p_opt_new - p_opt_old) / p_opt_old
                    portfolio_return_factor += weight * r_asset
                end
            end
        end

        portfolio_return_factor += rf_w * (RISK_FREE_R * DT)
        wealth_algo = wealth_algo * (1.0 + portfolio_return_factor)
        
        push!(hist_wealth_algo, wealth_algo)
        push!(hist_wealth_bench, wealth_benchmark)
        
        for i in 1:length(options)
            options[i] = Option(options[i].id, options[i].underlying_idx, options[i].strike, options[i].is_call, options[i].maturity - DT, options[i].held)
        end
    end
    
    # ==========================================
    # 7. PLOTTING & METRICS (UPDATED: ANNUALIZED)
    # ==========================================
    println("Generando graficos y metricas...")

    port_rets = [ (hist_wealth_algo[m+1] / hist_wealth_algo[m]) - 1.0 for m in 1:N_MONTHS ]
    bench_rets = [ (hist_wealth_bench[m+1] / hist_wealth_bench[m]) - 1.0 for m in 1:N_MONTHS ]
    port_curve = ones(N_MONTHS+1)
    idx_curve = ones(N_MONTHS+1)
    for m in 1:N_MONTHS
        port_curve[m+1] = port_curve[m] * (1.0 + port_rets[m])
        idx_curve[m+1] = idx_curve[m] * (1.0 + nasdaq_rets[m])
    end

    rf_m = RISK_FREE_R * DT
    
    function safe_std(x)
        s = std(x)
        return s < 1e-12 ? 1e-12 : s
    end
    
    function compute_beta(ret, mkt)
        v = var(mkt)
        return v < 1e-12 ? 0.0 : cov(ret, mkt) / v
    end
    
    function compute_metrics(name, ret, mkt)
        # 1. Monthly (Raw) Data
        excess = ret .- rf_m
        rel = ret .- mkt
        
        mean_excess = mean(excess)
        vol_monthly = safe_std(excess) 
        
        beta = compute_beta(ret, mkt)
        
        # 2. Annualization
        vol_annual = vol_monthly * sqrt(12)
        sharpe_annual = (mean_excess / vol_monthly) * sqrt(12)
        ir_annual = (mean(rel) / safe_std(rel)) * sqrt(12)
        
        # Alpha/Treynor scale linearly
        alpha_monthly = mean_excess - beta * mean(mkt .- rf_m)
        alpha_annual = alpha_monthly * 12
        treynor_annual = (mean_excess * 12) / (beta == 0 ? 1.0 : beta)
        
        total_return = prod(1 .+ ret) - 1
        
        println("\n--- Metricas Anualizadas: " * name * " ---")
        @printf("Sharpe Ratio:       %.4f\n", sharpe_annual)
        @printf("Information Ratio:  %.4f\n", ir_annual)
        @printf("Beta:               %.4f\n", beta)
        @printf("Treynor Ratio:      %.4f\n", treynor_annual)
        @printf("Alpha (Jensen):     %.4f\n", alpha_annual)
        @printf("Volatilidad Anual:  %.2f%%\n", vol_annual * 100)
        @printf("Retorno Total:      %.2f%%\n", total_return * 100)
    end

    compute_metrics("Portafolio LSM", port_rets, nasdaq_rets)
    compute_metrics("Benchmark 50/50", bench_rets, nasdaq_rets)

    @printf("\nRiqueza final Portafolio LSM: %.2f\n", hist_wealth_algo[end])
    @printf("Riqueza final Benchmark 50/50: %.2f\n", hist_wealth_bench[end])

    p1 = plot(hist_months, hist_wealth_algo, label="Estrategia Dinamica LSM", lw=2, color=:blue, legend=:outerright)
    plot!(p1, hist_months, hist_wealth_bench, label="Benchmark 50/50", lw=2, linestyle=:dash, color=:gray)
    title!(p1, "Evolucion de la riqueza")
    xlabel!(p1, "Mes")
    ylabel!(p1, "Riqueza (\$)")
    savefig(p1, "figura1_evolucion_riqueza.png")
    
    asset_labels_es = vcat("Libre de Riesgo", ["Accion $(k)" for k in asset_keys], [opt.id for opt in options])
    meses_plot = hist_months[2:end] 
    pesos_plot = hist_weights[2:end, :] 
    p2 = areaplot(meses_plot, pesos_plot, label=reshape(asset_labels_es, 1, :), alpha=0.6, legend=:outerright)
    title!(p2, "Asignacion de activos en el tiempo")
    xlabel!(p2, "Mes")
    ylabel!(p2, "Peso del portafolio")
    savefig(p2, "figura2_asignacion_activos.png")
    
    if !isempty(frontier_data_por_mes)
        for (mes, data) in sort(collect(frontier_data_por_mes); by=first)
            (fx, fy, opt_std, opt_mu) = data
            # Use line plot for the frontier instead of scatter
            p3 = plot(fx, fy, label="Frontera Eficiente Real", lw=2, color=:grey, legend=:outerright)
            scatter!(p3, [opt_std], [opt_mu], label="Eleccion optima", color=:red, markersize=8, shape=:star5)
            title!(p3, "Frontera eficiente (Mes $(mes))")
            xlabel!(p3, "Riesgo (Std)")
            ylabel!(p3, "Retorno Esperado")
            savefig(p3, @sprintf("figura3_frontera_eficiente_mes%02d.png", mes))
        end
    end
    
    if !isempty(lsm_plot_data_por_mes)
        for (mes, vecdata) in sort(collect(lsm_plot_data_por_mes); by=first)
            for tup in vecdata
                (lx, ly_cont, ly_int, opt_id) = tup
                perm = sortperm(lx)
                p4 = plot(lx[perm], ly_cont[perm], label="Valor continuacion", lw=2, color=:blue, legend=:outerright)
                plot!(p4, lx[perm], ly_int[perm], label="Valor ejercicio", lw=2, color=:green, linestyle=:dash)
                title!(p4, "Decision LSM $(opt_id) (Mes $(mes))")
                fname = @sprintf("figura4_decision_lsm_%s_mes%02d.png", opt_id, mes)
                savefig(p4, fname)
            end
        end
    end

    if !isempty(options)
        categorias = [
            ("CALL_ATM", "Lineas de vida - Calls ATM", "figura5_linea_opciones_call_atm.png"),
            ("CALL_OTM", "Lineas de vida - Calls OTM", "figura5_linea_opciones_call_otm.png"),
            ("PUT_ITM",  "Lineas de vida - Puts ITM",  "figura5_linea_opciones_put_itm.png")
        ]
        for (tag, titulo, fname) in categorias
            idxs = [i for (i, opt) in enumerate(options) if occursin(tag, uppercase(opt.id))]
            if isempty(idxs) continue end
            p5 = plot(title=titulo, xlabel="Mes", ylabel="Opcion", legend=false)
            y_ticks = collect(1:length(idxs))
            for (j, iopt) in enumerate(idxs)
                y = y_ticks[j]
                c = opt_exercised[iopt] ? :red : :blue
                plot!(p5, [opt_start[iopt], opt_end[iopt]], [y, y], lw=6, color=c)
            end
            yticks!(p5, y_ticks, [options[i].id for i in idxs])
            savefig(p5, fname)
        end
    end

    p6 = plot(hist_months, port_curve, label="Portafolio (LSM)", lw=2, color=:blue, legend=:outerright)
    plot!(p6, hist_months, idx_curve, label="NASDAQ acumulado", lw=2, color=:orange, linestyle=:dash)
    title!(p6, "Portafolio vs NASDAQ")
    savefig(p6, "figura6_portafolio_vs_nasdaq.png")

    println("Graficos guardados en archivos individuales.")
end

run_strategy(rng_seed=42)