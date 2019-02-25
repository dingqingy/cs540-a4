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

function marginalCK(p1, pt, d, k)
	M = zeros(k, d)
	M[:, 1] = p1
	for i = 2:d
		M[:, i] = pt' * M[:, i-1]
	end
	return M
end