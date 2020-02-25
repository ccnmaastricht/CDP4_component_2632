load training_data
load training_labels
training_labels = categorical(training_labels');
inputSize = [150 150 3];
filterSize = 5;
numFilters = 10;
numHiddenUnits = 300;
numClasses = 10;
layers = [ ...
    sequenceInputLayer(inputSize,'Name','input')
    sequenceFoldingLayer('Name','fold')
    convolution2dLayer(filterSize,numFilters,'Name','conv')
    batchNormalizationLayer('Name','bn')
    reluLayer('Name','relu')
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    lstmLayer(numHiddenUnits,'OutputMode','last','Name','lstm')
    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];
lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph,'fold/miniBatchSize','unfold/miniBatchSize');
options = trainingOptions('sgdm', 'InitialLearnRate', 1e-2 , 'MaxEpochs', 50 ...
    , 'Shuffle', 'every-epoch', 'Verbose',true, 'Plots', 'training-progress','MiniBatchSize',256);
trained_netowrk = trainNetwork(training_data,training_labels,lgraph,options)
