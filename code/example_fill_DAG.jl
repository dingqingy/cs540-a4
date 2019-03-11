# Load X and y variable
using JLD, PyPlot
data = load("../data/MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

m = size(X,1)
n = size(X,3)

include("inhomogMarkov.jl")
subModel = Array{SampleModel}(undef,m,m)

# Train a DAG model using tabular approach
for i in 3:m
    for j in 3:m
        subModel[i,j] = DAG(X[i-2:i,j-2:j,:])
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
                if j != 1 && j != 2
                    tile = I[i-2:i,j-2:j]
                    I[i,j] = subModel[i,j].sample(tile[:][1:end-1])
                else
                    I[i,j] = 0
                end
            end
        end
    end
    imshow(I)
end

