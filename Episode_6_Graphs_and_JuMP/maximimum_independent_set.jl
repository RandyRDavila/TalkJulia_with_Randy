using JuMP
using GLPK
using Graphs
using GraphPlot
using GraphIO


function maximum_independent_set(graph::SimpleGraph{Int64}; show_model = false)
    model = Model(GLPK.Optimizer)
    @variable(model, x[1:Graphs.nv(graph)], Bin)
    @objective(model, Max, sum(x))
    for e in Graphs.edges(graph)
        @constraint(model, x[Graphs.src(e)] + x[Graphs.dst(e)] <= 1)
    end
    optimize!(model)
    if show_model
        println(model)
    end
    return [xi for xi in x if value(xi) == 1.0]
end

# Read in the Petersen graph edge list text file 
graph = Graphs.loadgraph("PetersenGraph.txt", "graph_key", GraphIO.EdgeListFormat())
# Convert the directed graph to a simple graph
graph = Graphs.SimpleGraph(graph)

# Solve the maximimum independent set problem on the Petersen graph
max_independent_set = maximum_independent_set(graph, show_model = true)

println(max_independent_set)