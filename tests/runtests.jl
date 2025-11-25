using Test
using TFMFinanzas

@testset "TFMFinanzas basic tests" begin
    @test isapprox(bond_price(1000, 0.05, 0.04, 1; freq=1), 1000*0.05/(1+0.04) + 1000/(1+0.04), atol=1e-6)
    # Black-Scholes: for zero volatility, call payoff = max(S*exp(-rT)-K*exp(-rT),0) simpler test
    price = black_scholes_price(100.0, 100.0, 0.0, 1e-8, 1.0; option=:call)
    @test price ≈ max(100.0 - 100.0, 0.0)
    mc = mc_option_price(100.0, 100.0, 0.01, 0.2, 0.1; npaths=5000, seed=123, option=:call)
    @test mc >= 0
end
