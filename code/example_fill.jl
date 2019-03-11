# Load X and y variable
using JLD, PyPlot
data = load("../data/MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

m = size(X,1)
n = size(X,3)

include("inhomogMarkov.jl")
subModel = Array{SampleModel}(undef,m,m)

# Train an inhomog Markov chain
for i in 1:m
    for j in 1:m
        if j != 1
            subModel[i,j] = inhomogMarkov(X[i,j-1,:], X[i,j,:])
        else
            subModel[i,1] = inhomogMarkov(zeros(n), X[i,1,:])
        end
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
                    I[i,j] = subModel[i,j].sample(I[i,j-1])
                else
                    I[i,j] = subModel[i,j].sample(init[i])
                end
            end
        end
    end
    imshow(I)
end
