function g = sigmoidGradient(z)
% function g = sigmoidGradient(z)
%compute gradient of the sigmoid function
sigz=sigmoid(z);
g = sigz.*(1-sigz);

end
