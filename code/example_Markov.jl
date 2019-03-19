# Load X and y variable
using JLD, Printf
data = load("../data/gradChain.jld")
(p1,pt) = (data["p1"],data["pt"])
k = length(p1)

println("############ 1.1.1 ############")

# use MC estimate to generate 50000 samples
include("sampleAncestral.jl")
acc = zeros(k)
for i = 1:50000
	sample = sampleAncestral(p1, pt, 50)[50]
	acc[sample] += 1
end

marginal_MC = acc / 50000
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
# use MC estimate to generate 50000 samples
include("sampleAncestral.jl")
joint_count = zeros(k)
accepted = 0
for i = 1:10000
	sample = sampleAncestral(p1, pt, 10)
	if sample[10] == 6
		global accepted += 1
		joint_count[sample[5]] += 1
	end
end
marginal_cond = joint_count ./ accepted
@printf("Number of accepted samples: %d\n", accepted)
for i = 1:k
	@printf("Using rejection sampling, MCMC conditional marginal of state %d is %f\n", i, marginal_cond[i])
end

println("\n############ 1.2.2 ############")
include("marginalCK.jl")
margin = marginalCK(p1, pt, 10, k)
# construct reverse transition matrix
pr = zeros(10-5, k, k)
for time = 1:5
	for from = 1:k
		for to = 1:k
			pr[time, from, to] = pt[to, from] * margin[to, 10-time] / margin[from, 11-time]
		end
	end
end

# print(sum(pr, dims=3)) # sanity check

acc = zeros(k) # accumulator for counting
for i = 1:10000
init = zeros(Int8, k)
init[6] = 1 # start for grad school
	for j = 1:5
		global bk_sample = sampleAncestral(init, pr[j, :, :], 2)[2]
		# reset init
		init = zeros(Int8, k)
		init[bk_sample] = 1
		# print(typeof(sample))
	end
	# println(bk_sample)
	acc[bk_sample] += 1
end

backward_cond = acc ./ 10000.0
for i = 1:k
	@printf("Using backward sampling, MCMC conditional marginal of state %d is %f\n", i, backward_cond[i])
end

println("\n############ 1.2.3 ############")
# exact univariate conditional 
uni_cond = forwardBackwards(p1, pt, 5, 10, 6, k)
for i = 1:k
	@printf("Using forwardBackwards, exact univariate conditionals of state %d is %f\n", i, uni_cond[i])
end