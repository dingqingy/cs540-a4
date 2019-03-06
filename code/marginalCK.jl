function marginalCK(p1, pt, d, k)
	M = zeros(k, d)
	M[:, 1] = p1
	for i = 2:d
		M[:, i] = pt' * M[:, i-1]
	end
	return M
end

function forwardBackwards(p1, pt, c, j, d, k)
# compute p(x_j|x_d=c)
	l = max(j, d)
	M = zeros(k, l)
	V = zeros(k, l)

	M[:, 1] = p1

	if d == 1
		M[:, 1] = zeros(k, 1)
		M[c, 1] = 1
	end

	for i = 2:l
		M[:, i] = pt' * M[:, i-1]
		if i == d
			M[:, i] = zeros(k, 1)
			M[c, i] = 1
		end
	end

	if j > d
		V[:, j]=ones(k, 1)
	else
		V[c, d]=1
		for i = d-1:-1:j
			V[:, i]= pt * V[:, i+1]
		end
	end

	P = M[:, j] .* V[:, j]
	P_sum = sum(P)
	return P/P_sum
end
