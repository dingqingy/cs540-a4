# Load X and y variable
using JLD, PyPlot
data = load("../data/MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

m = size(X,1)
n = size(X,3)

include("logReg.jl")
subModel = Array{SampleModel}(undef,m,m)

# Train a sigmoid belief network
for i in 2:m
    for j in 2:m
        Xtrain = reshape(X[1:i, 1:j, :], (:, n))
        # println(size(Xtrain))
        # println(size(Xtrain[1:end-1, :]))
        # println(size(Xtrain[end, :]))
        subModel[i,j] = logReg(Xtrain[1:end-1, :]', Xtrain[end, :])
    end
end

# Fill-in some random test images
t = size(Xtest,3)
figure(1)
for image in 1:4
    subplot(2,2,image)

    # Grab a random test example
    ind = rand(1:t)
    I = Xtest[:,:,ind]

    # Fill in the bottom half using the model
    init = zeros(m)
    for i in 1:m
        for j in 1:m
            if isnan(I[i,j])
                if j != 1
                    tile = I[1:i,1:j]
                    I[i,j] = subModel[i,j].sample(tile[:][1:end-1])
                else
                    I[i,j] = 0
                end
            end
        end
    end
    imshow(I)
end

