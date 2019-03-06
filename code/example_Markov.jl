# Load X and y variable
using JLD
data = load("../data/gradChain.jld")
(p1,pt) = (data["p1"],data["pt"])
k = length(p1)

println("############ 1.1.1 ############")

# use MC estimate to generate 10000 samples
include("sampleAncestral.jl")
sample = zeros(Int8, 10000)
for i = 1:10000
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

println("\n############ 1.2.1 ############")
# use MC estimate to generate 10000 samples
n_experiments = 10000
sample = zeros(Int8, (n_experiments, 2))
for i = 1:n_experiments
	sample[i, :] = sampleConditioning(p1, pt, 5, 10)
end

x_10 = 6
sample_accepted = sample[sample[:, 2].==x_10, 1]
@printf("The number of sample accepted is %d.\n", size(sample_accepted)[1])
for i = 1:k
	@printf("P(X5= %d|X10=6) using MC with rejection is %f\n", i, sum(sample_accepted .== i)/size(sample_accepted)[1])
end

println("\n############ 1.2.2 ############")
# use backward sampling to generate 10000 samples
n_experiments = 10000
sample = zeros(Int8, n_experiments)
x_10 = 6
for i = 1:n_experiments
	sample[i] = sampleBackwards(p1, pt, x_10, 5, 10, k)
end

for i = 1:k
	@printf("P(X5= %d|X10=6) using MC with backward sampling is %f\n", i, sum(sample .== i)/size(sample)[1])
end

println("\n############ 1.2.3 ############")
#conditionals = forwardBackwards(p1, pt, x_10, 5, 10, k)
conditionals = forwardBackwards(p1, pt, 3, 50, 1, k)
## p(x_5|x_10=6)

for i = 1:k
	@printf("P(x_5=%i|x_10=6)= %f\n", i, conditionals[i])
end
