using Flux
using DataFrames
using DataFramesMeta
using Random
using Statistics

function __init__()
    """
    Initialization function which would initialize the LVQ model with given parameters. Other than that,
        this function would be used to raise exceptions
    """
end

function train(training_data,training_labels)
    """
    This function is used to train model using the training data.
    """
end

function test(testing_data,testing_labels)
    """
    This function is used to test the accuracy of LVQ model using the testing data.
    """
end

function euclidean_distance(row1, row2)
    """
    This function calculates the Euclidean distance between the two datapoints
    """
    distance = 0.0
	for i in range(len(row1)-1)
		distance += (row1[i] - row2[i])^2
	    return sqrt(distance)
    end
end


function get_best_matching_unit(codebooks, test_row)
    """
    This function helps locate the best matching unit
    """
    distances = Any[]
	for codebook in codebooks
		dist = euclidean_distance(codebook, test_row)
		append!(distances,[codebook, dist])
    end
	distances.sort()
	return distances
end
