function [Theta1, Theta2] = nnTrain(X,y,flagPlot)
% function [Theta1, Theta2] = nnTrain(X,y,flagPlot)
% train neural network
% input:
%    X: input samples packed in a matrix of (nSamples X sampleSize)
%    y: labels of input samples
%    flagPlot: if 1 plot history for debug)
%output: 
%    Theta1, Theta2: network parameters
%=========================================


%general training parameters
nIters =150; 
validationSize =0;

% get network parameters
inSize  = size(X,2);  % input dimention
hiddenLayerSize = 25;   % 25 hidden units
nLabels = length(unique(y)); 

% randomly initiate parameters
initialTheta1 = randInitializeWeights(inSize, hiddenLayerSize);
initialTheta2 = randInitializeWeights(hiddenLayerSize, nLabels);
initialParams = [initialTheta1(:) ; initialTheta2(:)];

% run gradient descent
[theta, Jh] = GradientDescent(X, y, initialParams,hiddenLayerSize, nIters,validationSize);

%% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(theta(1:hiddenLayerSize * (inSize + 1)), ...
                 hiddenLayerSize, (inSize + 1));

Theta2 = reshape(theta((1 + (hiddenLayerSize * (inSize + 1))):end), ...
                 nLabels, (hiddenLayerSize + 1));
if flagPlot
    plot(Jh,'.-');
end
