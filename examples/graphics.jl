using Distributions
using Plots

# Parámetros para la distribución normal
μ = 1.0
σ = 2.0
dist = Normal(μ, σ)

# Rango x para graficar
x = range(μ - 4σ, μ + 4σ, length=500)
y = pdf.(dist, x)

# Área a sombrear: dentro de 1 desviación estándar (μ - σ a μ + σ)
x_fill = range(μ - σ, μ + σ, length=200)
y_fill = pdf.(dist, x_fill)

gen_dir = joinpath(@__DIR__, "..", "generated")
mkpath(gen_dir)

# Gráfico
plot(x, y, label="Densidad normal", linewidth=2, legend=:topright)
plot!(x_fill, y_fill, fillrange=0, fillalpha=0.4, color=:orange, label="μ ± σ")
vline!([μ], color=:red, linestyle=:dash, label="μ")

xlabel!("x")
ylabel!("Densidad")
title!("Distribución normal (μ = $(μ), σ = $(σ))")
# Forzar límite inferior en y a 0
ylims!(0, maximum(y)*1.05)
savefig(joinpath(gen_dir, "normal_density.png"))
println("Guardado ", joinpath(gen_dir, "normal_density.png"))