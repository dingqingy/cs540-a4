function marginalCK(p1, pt, d, k)
	M = zeros(k, d)
	M[:, 1] = p1
	for i = 2:d
		M[:, i] = pt' * M[:, i-1]
	end
	return M
end