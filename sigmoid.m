function g = sigmoid(z)
% function g = sigmoid(z)
% computes the sigmoid of z.

g = 1.0 ./ (1.0 + exp(-z));
end
