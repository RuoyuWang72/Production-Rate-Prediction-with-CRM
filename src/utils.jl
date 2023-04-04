function create_nn(activation::Function, layer_sizes::Int...)
    layer_sizes = collect(layer_sizes)
    layers = []
    for i in 1:length(layer_sizes)-1
        push!(layers, Lux.Dense(layer_sizes[i], layer_sizes[i+1], activation))
    end
    push!(layers, Lux.Dense(1, 1, softplus))
    return Lux.Chain(layers...)
end