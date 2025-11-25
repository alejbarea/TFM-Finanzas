# Demo script showing bond pricing, Black-Scholes, and plots
using Pkg
Pkg.activate("..")
Pkg.instantiate()

using TFMFinanzas
using Plots

# 1) Ejemplo de valoración de bono
face = 1000
coupon = 0.05
ytm = 0.04
years = 5
price = bond_price(face, coupon, ytm, years)
println("Precio del bono (valor nominal=",face,", cupón=",coupon,", ytm=",ytm,") = ", round(price, digits=2))

# 2) Ejemplo Black-Scholes
S = 100.0
K = 110.0
r = 0.01
sigma = 0.2
T = 0.5
call_price = black_scholes_price(S, K, r, sigma, T; option=:call)
put_price = black_scholes_price(S, K, r, sigma, T; option=:put)
println("Precio call = ", round(call_price, digits=4), ", Precio put = ", round(put_price, digits=4))

# 3) Gráficos
# Curva de rendimientos
maturities = [0.5, 1, 2, 3, 5, 7, 10]
yields = [0.005, 0.008, 0.012, 0.015, 0.02, 0.025, 0.03]
gen_dir = joinpath(@__DIR__, "..", "generated")
mkpath(gen_dir)
py = plot_yield_curve(maturities, yields)
savefig(py, joinpath(gen_dir, "curva_rendimientos.png"))
println("Guardado ", joinpath(gen_dir, "curva_rendimientos.png"))

# Precio de opción vs strike
Ks = collect(50:5:150)
pk = plot_option_vs_strike(S, Ks, r, sigma, T; option=:call)
savefig(pk, joinpath(gen_dir, "precio_opcion_vs_strike.png"))
println("Guardado ", joinpath(gen_dir, "precio_opcion_vs_strike.png"))

# Caminos Monte Carlo
pmc = plot_mc_paths(S, r, sigma, 1.0; npaths=8, steps=200, seed=42)
savefig(pmc, joinpath(gen_dir, "caminos_mc.png"))
println("Guardado ", joinpath(gen_dir, "caminos_mc.png"))
