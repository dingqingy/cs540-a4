function marginalCK(p1, pt, d, k)
	M = zeros(k, d)
	M[:, 1] = p1
	for i = 2:d
		M[:, i] = pt' * M[:, i-1]
	end
	return M
end

function marginalBackward(pt, pd, d, k)
	M = zeros(k, d)
	M[:, d] = pd
	pt_column_normalized = pt./sum(pt, dims=1)
	for i = d-1:-1:1
		M[:, i] = pt_column_normalized * M[:, i+1]
	end
	return M
end

function forwardBackwards(pt, c, j, d, k)
# compute p(x_j|x_d=c)
	init_distribution = zeros(k)
	init_distribution[c]=1
	if (j>=d)
		result = marginalCK(init_distribution, pt, j-d+1, k)[:, j-d+1]
	else
		result = marginalBackward(pt, init_distribution, d-j+1, k)[:, 1]
	end

	return result
end
