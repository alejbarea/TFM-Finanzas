# ipm_visualization.jl
# Script de Julia que visualiza el Método de Punto Interior (camino central + contornos de barrera)
# Requiere: Plots, Optim, ForwardDiff

import Pkg
for pkg in ("Plots", "Optim", "ForwardDiff")
    try
        @eval using $(Symbol(pkg))
    catch
        Pkg.add(pkg)
        @eval using $(Symbol(pkg))
    end
end

using Plots, Optim, ForwardDiff

# Problema: minimizar f(x) sujeto a x1 >= 0, x2 >= 0, x1 + x2 <= 1
f(x) = (x[1] - 0.2)^2 + (x[2] - 0.7)^2

# Restricciones en forma g_i(x) <= 0 (interior: g_i(x) < 0)
g1(x) = x[1] + x[2] - 1   # <= 0  (x1 + x2 <= 1)
g2(x) = -x[1]             # <= 0  (x1 >= 0)
g3(x) = -x[2]             # <= 0  (x2 >= 0)
gs(x) = (g1(x), g2(x), g3(x))

# Objetivo con barrera para parámetro mu > 0:
function barrier_obj(x, μ)
    s1 = 1 - x[1] - x[2]
    s2 = x[1]
    s3 = x[2]
    # devuelve un valor muy grande si está fuera de la región factible para mantener interior
    if s1 <= 0 || s2 <= 0 || s3 <= 0
        # keep ForwardDiff-friendly penalty outside the interior
        return 1e12 + sum(max.(0.0, -[s1, s2, s3]).^2)
    end
    return f(x) - μ*(log(s1) + log(s2) + log(s3))  # f + μ*(-sum log(s_i))
end

function run_ipm_visualization()
    # Resolver la minimización con barrera para una secuencia decreciente de μ -> camino central
    mus = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    x0 = [0.2, 0.2]  # punto inicial interior
    central_points = Vector{Vector{Float64}}(undef, length(mus))

    for (i, μ) in enumerate(mus)
        # usar BFGS con autodiff en modo forward
        res = optimize(x -> barrier_obj(x, μ), x0, BFGS(); autodiff = :forward)
        x_star = Optim.minimizer(res)
        central_points[i] = x_star
        # arrancar la siguiente μ desde la solución actual
        x0 = x_star
    end

    # Preparar la malla para los contornos (solo dentro del triángulo factible)
    nx = 200
    xs = range(1e-3, stop = 1 - 1e-3, length = nx)
    ys = range(1e-3, stop = 1 - 1e-3, length = nx)
    Z = Dict{Float64, Array{Float64,2}}()
    mask = Array{Bool}(undef, nx, nx)

    for (i, xi) in enumerate(xs)
        for (j, yj) in enumerate(ys)
            mask[i,j] = (xi + yj) < 1.0
        end
    end

    # calcular la grilla del objetivo con barrera para cada μ (enmascarando lo infactible)
    for μ in mus
        arr = fill(NaN, nx, nx)
        for (i, xi) in enumerate(xs)
            for (j, yj) in enumerate(ys)
                if mask[i,j]
                    arr[i,j] = barrier_obj([xi, yj], μ)
                end
            end
        end
        Z[μ] = arr
    end

    # Configuración del gráfico
    palette = cgrad(:viridis, length(mus))
    plt = plot(title="Punto interior: barrera y camino central",
               xlabel="x₁", ylabel="x₂", aspect_ratio=1, legend=false)

    # Polígono de la región factible
    poly_x = [0.0, 1.0, 0.0, 0.0]
    poly_y = [0.0, 0.0, 1.0, 0.0]
    plot!(plt, poly_x, poly_y, lw=2, lc=:black, label="Región factible (cierre)")

    # Contorno relleno para la μ más pequeña para mostrar la cubeta
    μ0 = mus[end]
    contour!(plt, xs, ys, Z[μ0]', levels=20, fill=true, alpha=0.5, c=:blues, label="Contornos de barrera (μ=$(μ0))")

    # Superponer contornos para el resto de μ con diferentes colores
    for (k, μ) in enumerate(mus[1:end-1])
        contour!(plt, xs, ys, Z[μ]', levels=10, linewidth=1.2, c=palette[k], alpha=0.8)
    end

    # Puntos del camino central y su conexión
    cx = [p[1] for p in central_points]
    cy = [p[2] for p in central_points]
    scatter!(plt, cx, cy, marker=:circle, ms=6, c=:red, label="Puntos del camino central")
    plot!(plt, cx, cy, lw=2, lc=:red, label="Camino central")

    # Flechas entre puntos sucesivos para indicar la dirección (μ decreciente)
    for i in 1:(length(cx)-1)
        xstart, ystart = cx[i], cy[i]
        xend, yend = cx[i+1], cy[i+1]
        dx, dy = xend - xstart, yend - ystart
        quiver!(plt, [xstart], [ystart], quiver=([dx], [dy]), arrowsize=0.15, lc=:red)
    end

    # Marcar el minimizador sin restricciones como referencia
    # Nota: el minimizador sin restricciones de f está en (0.2,0.7) pero es infactible (fuera de x1+x2<=1)
    scatter!(plt, [0.2], [0.7], ms=6, mc=:magenta, markerstrokecolor=:black, label="Minimizador sin restricciones")

    # Guardar y mostrar
    savefig(plt, "ipm_central_path.png")
    println("Visualización guardada en ipm_central_path.png")
    display(plt)
end

run_ipm_visualization()