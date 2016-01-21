function displayPCA(X,U)
% function displayPCA(X,U)
% display largest components (PCA) of samples
% input:
%    X: samples matrix
%    U: PCA matrix
%=========================================

[nSamples,inSize]=size(X);
nEvPlot = 10;
outMat = zeros(nSamples*(nEvPlot+2),inSize);
inInd=1;
%for each sample:
for k=1:nSamples
    %calc principal components
    z=U*X(k,:)'; 
    %find biggest components
    [~,b]=sort(abs(z)); 
    maxInds = flip(b);
    maxInds = maxInds(1:nEvPlot);
    %stack image and its components to matrix
    outMat(inInd,:) = X(k,:);
    outMat(inInd+(1:nEvPlot),:) = U(maxInds,:);
    outMat(inInd+(1+nEvPlot),:) = (U(maxInds,:)')*z(maxInds);
    inInd = inInd+nEvPlot+2;   
end

DisplayData(outMat,nSamples,nEvPlot+2)

