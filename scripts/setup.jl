# Setup script to prepare the project environment
using Pkg
Pkg.activate(".")
Pkg.instantiate()
println("Project environment instantiated. Use `julia --project=. examples/demo.jl` to run the demo.")
