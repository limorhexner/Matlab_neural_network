% use nural network for digits recognition
% reduce dimention
% train neural network
clear ; close all; clc

%% read data
X = csvread('input.csv');
y = csvread('learnOutput.csv')';
nSamples=size(X,1);

%% reduce dimention
% Run PCA
[U, S] = pca(X);
% take eigenvectors responsible to 99% of energy
s = diag(S);
nd = find(cumsum(s)/sum(s)>0.99,1);
% plot eigen vectors
figure;
DisplayData(U(:, 1:nd)');
title('Eigen Vectors')
% projection data
Ur = (U(:,1:nd));
% new input
Z = X*Ur;

%% run nuoron network learning
hidden_layer_size = 25;   % 25 hidden units
[Theta1, Theta2] = nnTrain(Z,y,0);

%% test results
testLength = 1000;
inds = randperm(nSamples);
Zt = Z(inds(1:testLength),:);
Yt = y(inds(1:testLength));
pred = predict(Theta1, Theta2, Zt);
errInd = (pred~=Yt);
disp(['error rate on test series: ' num2str(sum(errInd)/testLength)]);
figure;
Xt = Zt*Ur';
DisplayData(Xt(errInd,:));
title('error classification')
disp('wrong classification values:')
disp(mat2str(Yt(errInd,:)));
disp(mat2str(pred(errInd,:)));

%% display digite and its Z values:
%4 success samples and 4 error samples:
Xp = [Xt(find(~errInd,4),:);Xt(find(errInd,4),:)];
displayPCA(Xp,Ur');



