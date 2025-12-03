using CSV
using DataFrames
using Dates
using Statistics

# Configuration
data_dir = joinpath(@__DIR__, "..", "data")
cutoff_date = Date(2024, 12, 31) # You can change this date
date_format = "mm/dd/yyyy"

# Get list of CSV files
files = readdir(data_dir)
csv_files = filter(f -> endswith(f, ".csv"), files)

println("Computing statistics (Mean and Volatility) for data up to $cutoff_date")
println("----------------------------------------------------------------")

for file in csv_files
    file_path = joinpath(data_dir, file)
    
    # Read CSV
    # Silence warnings about missing columns if any, though we expect standard format
    df = CSV.read(file_path, DataFrame)
    
    # Check if it has 'Date' and 'Close' columns (Stock data)
    if "Date" in names(df) && "Close" in names(df)
        # Parse dates
        # The date format in the files is mm/dd/yyyy (e.g. 12/31/2024)
        try
            df.Date = Date.(df.Date, date_format)
        catch e
            println("Skipping $file: Could not parse dates with format $date_format. Error: $e")
            continue
        end
        
        # Filter by date
        df_filtered = filter(row -> row.Date <= cutoff_date, df)
        sort!(df_filtered, :Date)
        
        if nrow(df_filtered) < 2
            println("Skipping $file: Not enough data points before cutoff.")
            continue
        end
        
        # Compute Log Returns: ln(P_t / P_{t-1})
        prices = df_filtered.Close
        # Ensure prices are Float64
        prices = Float64.(prices)
        
        # diff(log.(prices)) calculates log(P_t) - log(P_{t-1}) = log(P_t / P_{t-1})
        returns = diff(log.(prices))
        
        # Compute Statistics
        mu_daily = mean(returns)
        sigma_daily = std(returns)
        
        # Annualization (assuming 252 trading days)
        mu_annual = mu_daily * 252
        sigma_annual = sigma_daily * sqrt(252)
        
        println("File: $file")
        println("  Data points: $(length(returns))")
        println("  Daily Mean Return: $(round(mu_daily, digits=6))")
        println("  Daily Volatility:  $(round(sigma_daily, digits=6))")
        println("  Annualized Mean:   $(round(mu_annual, digits=4))")
        println("  Annualized Vol:    $(round(sigma_annual, digits=4))")
        println("----------------------------------------------------------------")
        
    elseif file == "daily-treasury-rates.csv"
         println("Skipping $file: Treasury rates file (treated differently than stock prices).")
         println("----------------------------------------------------------------")
    else
        println("Skipping $file: Missing 'Date' or 'Close' column.")
        println("----------------------------------------------------------------")
    end
end
