function viterbiDecode(p1, pt, d, k)
	#construct M table
	M = zeros(k, d)
	B = zeros(Int8, k, d)

	M[:, 1] = p1
	for i in 2:d
		for c in 1:k
			joint = M[:, i-1] .* pt[:, c]
			M[c, i] = maximum(joint)
			B[c, i] = argmax(joint)
		end
	end

	# return decoding
	decoded = zeros(Int8, d)
	decoded[d] = argmax(M[:, d])
	for i in d:-1:2
		decoded[i-1] = B[decoded[i], i]
	end
	return decoded
end