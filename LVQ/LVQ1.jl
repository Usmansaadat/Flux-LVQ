using Flux
using DataFrames
using DataFramesMeta
using Random
using Statistics
using MLBase

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
    for i in 1:length(row1)
        distance += (row1[i] - row2[i])^2
        return sqrt(distance)
    end
end

function accuracy_metric(actual, predicted)
    """
    This function is used to calculate the accuracy percentage
    """
    correct = 0
	for i in 1:length(actual)
		if actual[i] == predicted[i]
			correct += 1
        end
    end
	return correct / float(length(actual)) * 100.0
end

function get_best_matching_unit(codebooks, test_row)
    """
    This function helps locate the best matching unit
    """
    distances = Any[]
    for codebook in codebooks
        dist = euclidean_distance(codebook, test_row)
        push!(distances,[codebook, dist])
    end
    sort(distances[1])
    return distances[1][1]
end


function predict(codebooks, test_row)
    """
    This function helps make a prediction with codebook vectors
    """

    bmu = get_best_matching_unit(codebooks, test_row)
	return last(bmu)
end

function cross_validation_split(dataset, n_folds)
    dataset_split = Vector{Any}
    dataset_copy = collect(Iterators.flatten(dataset))
    fold_size = Int(length(dataset) / n_folds)
    for i=1:n_folds
        fold = Vector{Any}
        while length(fold) < fold_size
            index = rand((1:length(dataset_copy)))
            push!(fold, dataset_copy[index])
        end
    end
    push!(dataset_split,fold)
    return dataset_split
end




function random_codebook(train)
    """
    This function helps create the random codebook vector
    """
    n_features = length(train[1])
    n_records = length(train)
    codebook = [train[rand(1:n_records)][i] for i in 1:n_features]
    return codebook
end


function train_codebooks(train, n_codebooks, lrate, epochs)
    """
    This function trains a set of codebook vectors
    """
    codebooks = [random_codebook(train) for i in 1:n_codebooks]
    for epoch in 1:epochs
        rate = lrate * (1.0-(epoch/float(epochs)))
        for row in train
            bmu = get_best_matching_unit(codebooks, row)
            for i in 1:length(row)
                error = row[i] - bmu[i]
                if last(bmu) == last(row)
                    bmu[i] += rate * error
                else
                    bmu[i] -= rate * error
                end
            end
        end
    end
    return codebooks
end


function learning_vector_quantization(train, test, n_codebooks, lrate, epochs)
	"""
    This function is the main LVQ algorithm
    """
    codebooks = train_codebooks(train, n_codebooks, lrate, epochs)
	predictions = Any[]
	for row in test
		output = predict(codebooks, row)
		push!(predictions,output)
    end
	return(predictions)
end
