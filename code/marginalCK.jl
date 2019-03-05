function marginalCK(p1, pt, d, k)
	M = zeros(k, d)
	M[:, 1] = p1
	for i = 2:d
		M[:, i] = pt' * M[:, i-1]
	end
	return M
end

function marginalBackward(pd, pt, d, k)
	# 
	V = zeros(k, d)
	V[:, d] = pd
	for i = d:-1:2
		V[:, i-1] = pt * V[:, i]
	end
	return V
end

# compute prob(x_j | x_d = c)
function forwardBackwards(p1, pt, j, d, c, k)
	#forward
	M = marginalCK(p1, pt, d, k)

	# backward
	pd = zeros(k)
	pd[c] = 1
	V = marginalBackward(pd, pt, d, k)
	result = M[:, j] .* V[:, j]
	return 1 ./ sum(result) * result
end