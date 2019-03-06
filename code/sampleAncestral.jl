include("misc.jl")

function sampleAncestral(p1, pt, d)

	x = zeros(Int8, d)
	# at time 1
	x[1] = sampleDiscrete(p1)
	for i in 2:d
		x[i] = sampleDiscrete(pt[x[i-1], :])
	end
	return x[d]
end

function sampleConditioning(p1, pt, m, d) # m: intermediate state

	x = zeros(Int8, d)
	# at time 1
	x[1] = sampleDiscrete(p1)
	for i in 2:d
		x[i] = sampleDiscrete(pt[x[i-1], :])
	end
	return [x[m], x[d]]
end

function sampleBackwards(p1, pt, xd, i, d, k)
#pt: transition probability
#xd: final state
#i: initial time, d: final time
	x = zeros(Int8, d-i+1)
	M = marginalCK(p1, pt, d, k)
	# at time 1
	x[d-i+1] = xd
	for j in d-1:-1:i
		x[j-i+1] = sampleDiscrete(pt[:, x[j+1-i+1]].*M[:, j]./M[x[j+1-i+1],j+1])
	end
	return x[1]
end

function marginalCK(p1, pt, d, k)
	M = zeros(k, d)
	M[:, 1] = p1
	for i = 2:d
		M[:, i] = pt' * M[:, i-1]
	end
	return M
end
