# Load X and y variable
using JLD
data = load("../data/gradChain.jld")
(p1,pt) = (data["p1"],data["pt"])
k = length(p1)

println("############ 1.1.1 ############")

# use MC estimate to generate 50000 samples
include("sampleAncestral.jl")
sample = zeros(Int8, 50000)
for i = 1:50000
	sample[i] = sampleAncestral(p1, pt, 50)
end

using StatsBase, Printf
count = counts(sample)
marginal_MC = count / 50000
for i = 1:k
	@printf("marginal of state %d at time 50 using MC is %f\n", i, marginal_MC[i])
end

println("\n############ 1.1.2 ############")

include("marginalCK.jl")
exact_marginal = marginalCK(p1, pt, 50, k)

for i = 1:k
	@printf("exact marginal of state %d at time 50 is %f\n", i, exact_marginal[i, 50])
end

println("\n############ 1.1.3 ############")

include("viterbiDecode.jl")
time = 50
println("time is: ", time)
decoded = viterbiDecode(p1, pt, time, k)

println("Viterbi Decoded sequences: ")
for i = 1:time
	print(decoded[i])
end
println("")

time = 100
println("time is: ", time)
decoded = viterbiDecode(p1, pt, time, k)

println("Viterbi Decoded sequences: ")
for i = 1:time
	print(decoded[i])
end
println("")

# state sequences by maximizing marginal
println("state sequences obtained by maximizing marginal")
M = marginalCK(p1, pt, time, k)
for i = 1:time
	print(argmax(M[:, i]))
end
println("")

println("\n############ 1.1.4 ############")

init = zeros(Int8, k)
init[3] = 1 # start for grad school
conditional = marginalCK(init, pt, 50, k)
for i = 1:k
	@printf("exact marginal of state %d at time 50 starting from grad school is %f\n", i, conditional[i, 50])
end