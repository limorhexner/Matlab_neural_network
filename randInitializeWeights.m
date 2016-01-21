function W = randInitializeWeights(lIn, lOut)
% function W = randInitializeWeights(lIn, lOut)
% randomize initial weights 
epsilon_init = 0.12;
W = rand(lOut, 1 + lIn) * 2 * epsilon_init - epsilon_init;

end
