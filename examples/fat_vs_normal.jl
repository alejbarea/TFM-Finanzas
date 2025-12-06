# examples/fat_vs_normal.jl
# Dibuja la PDF de una Normal estándar y una distribución de cola gruesa (t de Student)
using Plots
using Distributions

# Datos
x = -8:0.01:8
dist_normal = Normal(0, 1)
dist_t = TDist(1)   # cola más gruesa al tener pocos grados de libertad

pdf_normal = pdf.(dist_normal, x)
pdf_t = pdf.(dist_t, x)

# Gráfica
default(size=(800,500))
p = plot(x, pdf_normal;
    label="Normal N(0,1)",
    lw=3,
    color=:blue)
plot!(p, x, pdf_t;
    label="Cola gruesa — t(ν=3)",
    lw=3,
    color=:red,
    linestyle=:dash)

xlabel!("Valor (x)")
ylabel!("Densidad de probabilidad")
title!("Normal vs. Distribución de Cola Gruesa")
plot!(p, legend=:topright, grid=true)

# Anotaciones en español
annotate!(2.5, pdf_t[x .== 2.5][1] + 0.01, text("Cola más pesada", :red, 10))
annotate!(-2.5, pdf_normal[x .== -2.5][1] + 0.01, text("Cola ligera (Normal)", :blue, 10))

# Guardar imagen
savefig("fat_vs_normal.png")

# Mostrar en pantalla (si el entorno lo permite)
display(p)