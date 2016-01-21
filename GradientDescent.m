function [theta, Jh,Jv] = GradientDescent(X, y, theta,hiddenLayerSize, nIters,validationSize)
% function [theta, Jh,Jv] = GradientDescent(X, y, theta, nIters,validationSize)
% run gradient descent algorithm to find local minimum of cost function
% inputs:
%    X: input samples packed in a matrix of (nSamples X sampleSize)
%    y: labels of input samples
%    theta: initial network parameters
%    hiddenLayerSize: # neurons in middle level
%    nIters: # of iterations to run
%    validationSize: size of validation set, used for debug
% output:
%    theta: network parameters after optimization
%    Jh,Jv: cost history for training set and validation set
%=========================================================

%general parameters:
stepSize     =10; 
lambda    = 0.1; 

assert(validationSize< length(y));
trainSize =  length(y)-validationSize;% number of training examples
Jh = zeros(nIters, 1);
Jv = zeros(nIters, 1);
  

%cut to train set and validation set
inds = randperm(length(y));
Xt = X(sort(inds(1:trainSize)),:);
yt = y(sort(inds(1:trainSize)));
Xv = X(inds(trainSize+1:end),:);
yv = y(inds(trainSize+1:end));

%run algorithm steps:
[J ,grad] = nnCostFunction(theta,hiddenLayerSize,Xt, yt, lambda);

for iter = 1:nIters
    %try default step size
    currStep = stepSize;
    while 1
        %walk one step in gradient direction
        newTheta = theta - currStep*grad;
        [Jn] = nnCostFunction(newTheta,hiddenLayerSize,Xt, yt, lambda);
        if Jn<J %succsess
            theta = newTheta;
            break
        else %step too big
           currStep = currStep/5;
        end
    end
    %get new gradient
    [J,grad] = nnCostFunction(newTheta,hiddenLayerSize,Xt, yt, lambda);
    %save cost history
    Jh(iter) = J;
    if validationSize>0
        Jv(iter) = nnCostFunction(theta,hiddenLayerSize,Xv, yv, 0);
    end
      
    if ~mod(iter,20)
        disp(['iteration ' num2str(iter) ', cost = ' num2str(J)]);
    end

end

end