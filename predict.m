function p = predict(Theta1, Theta2, X)
% function p = predict(Theta1, Theta2, X)
% with given network parameters, run neural betwork to predict output
%=========================

m = size(X, 1);
%first layer
h1 = sigmoid([ones(m, 1) X] * Theta1');
%second layer
h2 = sigmoid([ones(m, 1) h1] * Theta2');
%get max output neuron
[~, p] = max(h2, [], 2);

end
