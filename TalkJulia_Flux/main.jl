using Flux
using Plots
using RDatasets

# Target Function
target_function(x) = 4x + 2 + (rand()*rand(1:5))^(rand(1:2))

# Create Artificial Data
x_train, x_test = hcat(0:5...), hcat(6:10...)
y_train, y_test = target_function.(x_train), target_function.(x_test)

# Define a simple model with one input node
model = Dense(1, 1)

println("model weight variable = $(model.W)")
println("model bias variable = $(model.b)")

# Define Mean Squared Error Loss Function
loss(x, y) = Flux.Losses.mse(model(x), y)

println("Initial Loss: $(loss(x_train, y_train))")

# Define Gradient Descent Optimizer
opt = Flux.Descent()

# Format your Data
data = [(x_train, y_train)]

# Collect weights and bias for your Models
parameters = Flux.params(model)

# Now train over 1_000 epochs
for epoch in 1:1_000
    Flux.train!(loss, parameters, data, opt)
end

println("Loss after training: $(loss(x_train, y_train))")

scatter(x_test, y_test, color = "magenta")
domain = LinRange(6, 10, 100)

plot!(domain, domain .* model.W .+ model.b, legend = false)
plot!()