# script: plot_bid_ask.jl
# Genera una gráfica enseñable del precio de una acción con su spread bid-ask.
# Todo el texto está en español.

using Pkg
try
    @eval using Plots
    @eval using Random
catch
    println("Instalando paquetes necesarios (Plots, Random)...")
    Pkg.add("Plots")
    Pkg.add("Random")
    using Plots
    using Random
end

# Parámetros y generación de datos sintéticos (modelo sencillo)
n = 240                      # número de puntos temporales (p. ej. minutos u horas)
t = collect(1:n)             # eje temporal
rng = MersenneTwister(42)    # semilla para reproducibilidad

μ = 0.0006                   # deriva media de los retornos
σ = 0.008                    # volatilidad de los retornos
retornos = μ .+ σ .* randn(rng, n)
precio_mid = 100.0 .* exp.(cumsum(retornos))  # precio medio (mid)

# Spread variable (porcentaje del precio) -> halfspread en unidades de precio
spread_pct = 0.001 .+ 0.1 .* abs.(randn(rng, n))
halfspread = precio_mid .* spread_pct ./ 2.0

# Bid y Ask
precio_bid = precio_mid .- halfspread
precio_ask = precio_mid .+ halfspread

# Crear la gráfica
gr()  # backend GR
p = plot(t, precio_mid;
    label = "Precio medio (mid)",
    lw = 2,
    color = :black)

# Área sombreada del spread (bid-ask) usando ribbon
plot!(p, t, precio_mid; ribbon = halfspread, fillalpha = 0.18, color = :blue, label = "Spread (bid-ask)")

# Líneas para bid y ask
plot!(p, t, precio_bid; label = "Bid (oferta)", lw = 1, ls = :dash, color = :red)
plot!(p, t, precio_ask; label = "Ask (demanda)", lw = 1, ls = :dash, color = :green)

# Anotaciones didácticas (en español)

# Etiquetas y leyenda en español
xlabel!(p, "Tiempo")
ylabel!(p, "Precio")
title!(p, "Evolución del precio con spread Bid-Ask")
plot!(p, legend=:topright)

# Guardar y mostrar
savefig(p, "precio_bid_ask.png")
println("Gráfica generada y guardada en 'precio_bid_ask.png'.")