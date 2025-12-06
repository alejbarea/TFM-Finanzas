using CSV
using DataFrames
using Dates
using Statistics
using Plots
using ColorSchemes

# Paths
DATA_DIR = joinpath(@__DIR__, "..", "data")
OUTPUT_DIR = joinpath(@__DIR__, "..", "generated")
mkpath(OUTPUT_DIR)

# Parameters
target_year = 2024
window_days = 30  # approximately one-month window
anchor_day = 15   # aim for the same day each month
DATE_FORMAT = "mm/dd/yyyy"
TRADING_DAYS = 252

month_labels = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# Pick the trading day in the month closest to the anchor day
function pick_anchor_date(dates::Vector{Date}, m::Int)
    cand = filter(d -> year(d) == target_year && month(d) == m, dates)
    if isempty(cand)
        return nothing
    end
    anchor_target = Date(target_year, m, min(anchor_day, day(lastdayofmonth(Date(target_year, m, 1)))) )
    idx = argmin(abs.(cand .- anchor_target))
    return cand[idx]
end

files = readdir(DATA_DIR)
csv_files = filter(f -> endswith(f, ".csv"), files)

println("Computing monthly mean and volatility (one-month windows) for $target_year")
println("----------------------------------------------------------------")

gr()

monthly_means = Dict{String, Vector{Float64}}()
monthly_vols  = Dict{String, Vector{Float64}}()
stats_rows = NamedTuple[]
returns_by_month = Dict(m => Dict{String,DataFrame}() for m in 1:12)

for file in csv_files
    if file == "daily-treasury-rates.csv"
        println("Skipping $file: Treasury rates handled separately.")
        println("----------------------------------------------------------------")
        continue
    end

    file_path = joinpath(DATA_DIR, file)
    df = CSV.read(file_path, DataFrame)

    if !("Date" in names(df) && "Close" in names(df))
        println("Skipping $file: Missing 'Date' or 'Close' column.")
        println("----------------------------------------------------------------")
        continue
    end

    try
        df.Date = Date.(df.Date, DATE_FORMAT)
    catch e
        println("Skipping $file: Could not parse dates with format $DATE_FORMAT. Error: $e")
        println("----------------------------------------------------------------")
        continue
    end

    sort!(df, :Date)
    dates = collect(df.Date)
    prices = Float64.(df.Close)

    means = fill(NaN, 12)
    vols = fill(NaN, 12)
    anchor_prices = fill(NaN, 12)

    for m in 1:12
        anchor = pick_anchor_date(dates, m)
        if anchor === nothing
            continue
        end
        anchor_idx = findfirst(==(anchor), dates)
        anchor_price = anchor_idx === nothing ? NaN : prices[anchor_idx]
        anchor_prices[m] = anchor_price
        window_start = anchor - Day(window_days)
        window_idx = findall(d -> d > window_start && d <= anchor, dates)
        if length(window_idx) < 2
            continue
        end
        window_prices = prices[window_idx]
        rets = diff(log.(window_prices))
        if isempty(rets)
            continue
        end
        means[m] = mean(rets)
        vols[m] = std(rets)
    end

    monthly_means[file] = means
    monthly_vols[file] = vols

    for m in 1:12
        if isnan(means[m]) || isnan(vols[m])
            continue
        end
        mu_ann = means[m] * TRADING_DAYS
        sig_ann = vols[m] * sqrt(TRADING_DAYS)
        push!(stats_rows, (
            activo = replace(file, ".csv" => ""),
            mes = month_labels[m],
            media_diaria = means[m],
            volatilidad_diaria = vols[m],
            media_anualizada = mu_ann,
            volatilidad_anualizada = sig_ann,
            precio_cierre_anchor = anchor_prices[m],
        ))

        # Store returns for covariance calculation
        anchor = pick_anchor_date(dates, m)
        if anchor !== nothing
            window_start = anchor - Day(window_days)
            window_idx = findall(d -> d > window_start && d <= anchor, dates)
            if length(window_idx) >= 2
                window_prices = prices[window_idx]
                rets = diff(log.(window_prices))
                if !isempty(rets)
                    ret_dates = dates[window_idx][2:end]
                    returns_by_month[m][replace(file, ".csv" => "")] = DataFrame(Date = ret_dates, ret = rets)
                end
            end
        end
    end

    valid_pts = count(!isnan, means)
    println("Archivo: $file -> meses con datos: $valid_pts")
    println("----------------------------------------------------------------")
end

stats_df = DataFrame(stats_rows)
if !isempty(stats_df)
    csv_path = joinpath(OUTPUT_DIR, "estadisticas_mensuales_2024.csv")
    CSV.write(csv_path, stats_df)
    println("Datos tabulados guardados en '$csv_path'")
end

# Compute and save annualized covariance matrices per month
for m in 1:12
    asset_map = returns_by_month[m]
    if length(asset_map) < 2
        continue
    end
    # common dates across all assets
    common_dates = reduce(intersect, [df.Date for df in values(asset_map)])
    if length(common_dates) < 2
        continue
    end
    sort!(common_dates)
    assets = sort(collect(keys(asset_map)))
    R = Array{Float64}(undef, length(common_dates), length(assets))
    for (j, name) in enumerate(assets)
        date_to_ret = Dict(asset_map[name].Date .=> asset_map[name].ret)
        R[:, j] = [date_to_ret[d] for d in common_dates]
    end
    cov_mat = cov(R) * TRADING_DAYS
    cov_df = DataFrame(asset = assets)
    for (j, name) in enumerate(assets)
        cov_df[!, name] = cov_mat[:, j]
    end
    fname = "covarianzas_$(target_year)_" * lpad(string(m), 2, '0') * ".csv"
    CSV.write(joinpath(OUTPUT_DIR, fname), cov_df)

    # Heatmap de covarianza para el mes m
    p_cov = heatmap(
        assets,
        assets,
        cov_mat,
        c = :balance,
        title = "Covarianza anualizada mes $(month_labels[m])",
        xlabel = "Activo",
        ylabel = "Activo",
        colorbar_title = "Cov",
    )
    savefig(p_cov, joinpath(OUTPUT_DIR, @sprintf("covarianza_heatmap_%s_%02d.png", string(target_year), m)))
end

x_vals = 1:12

p_means = plot(
    title = "Media de retornos (ventana mensual, $target_year)",
    xlabel = "Mes",
    ylabel = "Media diaria",
    legend = :outerright,
    xticks = (x_vals, month_labels),
)
for (file, vals) in monthly_means
    plot!(p_means, x_vals, vals, label = replace(file, ".csv" => ""))
end
savefig(p_means, joinpath(OUTPUT_DIR, "medias_mensuales_2024.png"))

p_vols = plot(
    title = "Volatilidad (ventana mensual, $target_year)",
    xlabel = "Mes",
    ylabel = "Volatilidad diaria",
    legend = :outerright,
    xticks = (x_vals, month_labels),
)
for (file, vals) in monthly_vols
    plot!(p_vols, x_vals, vals, label = replace(file, ".csv" => ""))
end
savefig(p_vols, joinpath(OUTPUT_DIR, "volatilidades_mensuales_2024.png"))

println("Figuras guardadas en 'generated/medias_mensuales_2024.png' y 'generated/volatilidades_mensuales_2024.png'")
