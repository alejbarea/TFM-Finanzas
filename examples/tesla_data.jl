using CSV
using DataFrames
using PlotlyJS
using Dates

# Path to the TSLA.csv file
csv_path = joinpath(@__DIR__, "..", "data", "TSLA.csv")

# Read the CSV file into a DataFrame
tsla_df = CSV.read(csv_path, DataFrame)

# Parse Date column if not already Date type (format in CSV is mm/dd/yyyy)
if !(eltype(tsla_df.Date) <: Date)
    tsla_df.Date = Date.(tsla_df.Date, dateformat"mm/dd/yyyy")
end

# Sort by Date ascending so the chart is chronological
sort!(tsla_df, :Date)

# Extract series
date = tsla_df.Date
open = tsla_df.Open
high = tsla_df.High
low = tsla_df.Low
close = tsla_df.Close

# Create candlestick plot (PlotlyJS)
trace = candlestick(
    x = date,
    open = open,
    high = high,
    low = low,
    close = close
)

plt = Plot(trace, Layout(title="TSLA Candlestick Chart"))

# Ensure generated directory exists and save interactive HTML
gen_dir = joinpath(@__DIR__, "..", "generated")
mkpath(gen_dir)
html_path = joinpath(gen_dir, "tsla_candlestick.html")
PlotlyJS.savefig(plt, html_path)
println("Guardado ", html_path)

display(plt)