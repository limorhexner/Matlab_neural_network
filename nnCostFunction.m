function [J ,grad] = nnCostFunction(nnParams,hiddenLayerSize, X, y, lambda)
% [J ,grad] = nnCostFunction(nnParams,hiddenLayerSize, X, y, lambda)
% for corrent network parameters, calculate cost function ang gradient
% input:
%    nnParams: network parameters
%    hiddenLayerSize: # neurons in middle level
%    X: input samples packed in a matrix of (nSamples X sampleSize)
%    y: labels of input samples
%    lambda: regularization term
% output:
%    J: cost function at corrent parameters
%    grad: cost function gradient at current parameters
%==========================================================

%get info
[nSamples,inSize] = size(X); 
nLabels = length(unique(y));    

% Reshape nnParams into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nnParams(1:hiddenLayerSize * (inSize + 1)), ...
                 hiddenLayerSize, (inSize + 1));

Theta2 = reshape(nnParams((1 + (hiddenLayerSize * (inSize + 1))):end), ...
                 nLabels, (hiddenLayerSize + 1));

%% 
% Feedforward the neural network and return the cost 
a1 = [ones(nSamples, 1) X];
z2 = a1*Theta1';
a2 = [ones(nSamples, 1) sigmoid(z2)];
z3 = a2*Theta2';
a3 =  sigmoid(z3);
hTheta =a3;
yMat = zeros(nSamples,nLabels);
yMat(sub2ind(size(yMat),1:nSamples,y')) = 1;

cost = -yMat(:)'*log(hTheta(:)) -(1-yMat(:))'*log(1-hTheta(:)) ;
J = cost/nSamples + lambda/(2*nSamples)*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
if nargout<2
    return
end

%%
% Backpropagation algorithm to compute the gradients
delta3 = (a3-yMat)';
gTag2 = sigmoidGradient(z2');
delta2 = (Theta2'*delta3);
delta2 = delta2(2:end,:).*gTag2; %remove bias

s1 = inSize;
s2 = hiddenLayerSize;
s3 = nLabels;
Delta2 = zeros(s3,s2+1);
Delta1 = zeros(s2,s1+1);

for t=1:nSamples
  Delta1 = Delta1 + delta2(:,t)*a1(t,:);
  Delta2 = Delta2 + delta3(:,t)*a2(t,:);
end
Theta1_grad = 1/nSamples*Delta1;
Theta2_grad = 1/nSamples*Delta2;

%% 
%regularization 

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/nSamples*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/nSamples*Theta2(:,2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
