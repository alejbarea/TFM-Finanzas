# TFMFinanzas

Un pequeño proyecto en Julia para matemáticas financieras y visualización (punto de partida para un TFM o flujo de trabajo).

Inicio rápido (PowerShell):

```powershell
# Desde la raíz del proyecto
julia --project=. scripts/setup.jl
# Ejecutar el demo que crea varias imágenes en examples/
julia --project=. examples/demo.jl
# Ejecutar pruebas
julia --project=. -e "using Pkg; Pkg.test()"
```

Paquetes recomendados usados
- Plots, StatsPlots
- Distributions
- DataFrames, CSV (para tareas con datos)

Notas
- El proyecto está preparado para Julia 1.9+ (ver `Project.toml`)
- Usa la extensión de Julia para VS Code y configura el entorno al proyecto (el archivo `.vscode/settings.json` establece `julia.environment` a `project`).

Archivos de interés
- `src/TFMFinanzas.jl` — módulo principal con funciones y utilidades de graficado
- `examples/demo.jl` — demo que produce tres PNGs en `examples/` (nombres de archivos en español)
- `tests/runtests.jl` — pruebas básicas
- `scripts/setup.jl` — instancia el entorno

Si quieres, puedo añadir una carpeta `docs/` o cuadernos interactivos (Pluto) para exploración interactiva.
