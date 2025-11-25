using CSV
using DataFrames

# Path to the CSV file in the data folder
csv_path = joinpath(@__DIR__, "..", "data", "daily-treasury-rates.csv")

using Plots
using Measures
using Dates

# Read the CSV file into a DataFrame
df = CSV.read(csv_path, DataFrame)

# Select only columns ending with "COUPON EQUIVALENT" plus Date
coupon_cols = filter(col -> endswith(col, "COUPON EQUIVALENT") || col == "Date", names(df))
df = df[:, coupon_cols]

# Parse the Date column to Date type (adapt format if needed)
df.Date = Date.(df.Date, dateformat"mm/dd/yyyy")

# Ensure dates are sorted ascending so time goes left-to-right
sort!(df, :Date)

# Display the column names and first rows
println("Column names: ", names(df))
println(first(df, 5))

# Ensure generated folder exists
gen_dir = joinpath(@__DIR__, "..", "generated")
mkpath(gen_dir)

# Convert the numeric columns (exclude Date) to a Matrix for plotting
Y = Matrix(df[:, Not(:Date)])
labels = string.(["4 Semanas", "8 Semanas", "13 Semanas", "17 Semanas", "26 Semanas", "52 Semanas"])

# Plot each COUPON EQUIVALENT column as a separate curve — one label per curva
p = plot(size=(1400,550), titlefont=font(18), guidefont=font(14), tickfont=font(11), legendfontsize=11, xrotation=45,
    left_margin=8mm, bottom_margin=18mm, right_margin=60mm, top_margin=8mm)
for (i, col) in enumerate(eachcol(Y))
    plot!(p, df.Date, col; label=labels[i], lw=2)
end
xlabel!(p, "Fecha")
ylabel!(p, "Rentabilidad (%)")
title!(p, "Rentabilidades vs Fecha")
# Reduce number of x-ticks to improve readability and avoid clipping
n_dates = length(df.Date)
n_ticks = min(12, n_dates)
step = max(1, Int(floor(n_dates / n_ticks)))
xt = df.Date[1:step:end]
xt_labels = Dates.format.(xt, dateformat"yyyy-mm")

# Apply xticks and legend placement
plot!(p, xticks=(xt, xt_labels), xrotation=45, legend=:outerright)

# Save the plot to the generated folder
savefig(p, joinpath(gen_dir, "bond_coupon_equivalents.png"))
println("Guardado ", joinpath(gen_dir, "bond_coupon_equivalents.png"))

# Save the DataFrame to the generated/ folder for consistency with other examples
CSV.write(joinpath(gen_dir, "bond_data_out.csv"), df)
println("Guardado ", joinpath(gen_dir, "bond_data_out.csv"))