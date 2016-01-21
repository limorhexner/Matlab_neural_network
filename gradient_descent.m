function [theta, J_history,J_validation] = gradient_descent(X, y, theta, alpha,lambda, num_iters,validationSize)
assert(validationSize< length(y));
m =  length(y)-validationSize;% number of training examples
J_history = zeros(num_iters, 1);
J_validation = zeros(num_iters, 1);
input_layer_size  = size(X,2);  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = length(unique(y));    

%cut to train set and validation set
inds = randperm(length(y));
Xt = X(sort(inds(1:m)),:);
yt = y(sort(inds(1:m)));

Xv = X(inds(m+1:end),:);
yv = y(inds(m+1:end));
[J ,grad] = nnCostFunction(theta,input_layer_size,hidden_layer_size,num_labels,Xt, yt, lambda);
for iter = 1:num_iters
    %try big alpha
    alphaT = alpha;
    while 1
        newTheta = theta - alphaT*grad;
        [Jn] = nnCostFunction(newTheta,input_layer_size,hidden_layer_size,num_labels,Xt, yt, lambda);
        if Jn<J %succsess
            theta = newTheta;
            break
        else %step too big
           alphaT = alphaT/5;
        end
    end
    [J,grad] = nnCostFunction(newTheta,input_layer_size,hidden_layer_size,num_labels,Xt, yt, lambda);
    if validationSize>0
        J_validation(iter) = nnCostFunction(theta,input_layer_size,hidden_layer_size, ...
            num_labels,Xv, yv, 0);
    end
    % Save the cost J in every iteration    
    J_history(iter) = J;
    if ~mod(iter,20)
        disp(['iteration ' num2str(iter) ', cost = ' num2str(J)]);
    end

end

end