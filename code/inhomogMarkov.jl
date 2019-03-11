include("misc.jl")

function inhomogMarkov(prev,curr)
	n = size(prev,1)

	# Compute the frequencies in the training data
	# (The below is *not* efficient in time or space)
	D =  Dict()
	for i in 1:n
		key = [curr[i];prev[i]]
		if haskey(D,key)
			D[key] += 1
		else
			D[key] = 1
		end
	end

	# Sample function
	function sampleFunc(xtilde)
		key0 = [0;xtilde]
		key1 = [1;xtilde]
		p0 = 0
		if haskey(D,key0)
			p0 = D[key0]
		end
		p1 = 0
		if haskey(D,key1)
			p1 = D[key1]
		end
		
		if p0+p1 == 0
			# Probability is undefined, go random
			return rand() < .5
		else
			return rand() < p1/(p0+p1)
		end
	end
	# Return model
	return SampleModel(sampleFunc)
end

function DAG(tile) # 3 by 3 pixel containing parent and child
	n = size(tile,3)

	# Compute the frequencies in the training data
	# (The below is *not* efficient in time or space)
	D =  Dict()
	for i in 1:n
		key = tile[:, :, i][:]
		if haskey(D,key)
			D[key] += 1
		else
			D[key] = 1
		end
	end

	# Sample function
	function sampleFunc(xtilde)
		key0 = [xtilde;0]
		key1 = [xtilde;1]
		p0 = 0
		if haskey(D,key0)
			p0 = D[key0]
		end
		p1 = 0
		if haskey(D,key1)
			p1 = D[key1]
		end
		
		if p0+p1 == 0
			# Probability is undefined, go random
			return rand() < .5
		else
			return rand() < p1/(p0+p1)
		end
	end
	# Return model
	return SampleModel(sampleFunc)
end