using DataFrames
using CSV
using DataStructures


struct KNN
    """
    At first, a type KNN is been made
    """
    x::DataFrames.DataFrame
    y::DataFrames.DataFrame
end

function predict(data::KNN, testData::DataFrames.DataFrame, k=5)
    """
    This function works as the fit phase. Here, the distances between target points and
    train data points are calculated. By the nearest k point’s label, we can decide the target's label
    on majority rule.
    The function used within it sortperm(), doesn't have intuitive name.
    This function is to sort the data and return the index of sorted data..
    """
    predictedLabels = []
    for i in 1:size(testData, 1)
        sourcePoint = Array(testData[i,:])
        distances = []
        for j in 1:size(data.x, 1)
            destPoint = Array(data.x[j,:])
            distance = calcDist(sourcePoint, destPoint)
            push!(distances, distance)
        end
        sortedIndex = sortperm(distances)
        targetCandidates = Array(data.y)[sortedIndex[1:k]]
        predictedLabel = extractTop(targetCandidates)
        push!(predictedLabels, predictedLabel)
    end
    return predictedLabels
end



function calcDist(sourcePoint, destPoint)
    """
    This function is used to calculate the distance between two points
    """
    sum = 0
    for i in 1:length(sourcePoint)
        sum += (destPoint[i] - sourcePoint[i]) ^ 2
    end
    dist = sqrt(sum)
    return dist
end

function extractTop(targetCandidates)
    """
    This function is used to get the frequency of labels and the top frequency's label.
    """
    targetFrequency = counter(targetCandidates)

    normValue = 0
    normKey = "hoge"

    for key in keys(targetFrequency)
        if targetFrequency[key] > normValue
            normKey = key
            normValue = targetFrequency[key]
        end
    end
    return normKey
end

function splitTrainTest(data, at = 0.7)
    """
    This function is used to split the data into train and test set
    """
    n = nrow(data)
    ind = shuffle(1:n)
    train_ind = view(ind, 1:floor(Int, at*n))
    test_ind = view(ind, (floor(Int, at*n)+1):n)
    return data[train_ind,:], data[test_ind,:]
end
