function [U, S] = pca(X)
%function [U, S] = pca(X)
%Run principal component analysis on the dataset X


[m, n] = size(X);
covMat = zeros(n);
for ii=1:m
    x=X(ii,:);
    covMat = covMat+x'*x/m;
end
[U,S]=svd(covMat);


end
