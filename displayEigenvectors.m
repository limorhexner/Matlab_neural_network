function displayEigenvectors(X,U)
%get input images arrange in a matrix and display them with their 10 main
%eigenvectors

[nSamples,inSize]=size(X);
nEvPlot = 10;
outMat = zeros(nSamples*(nEvPlot+2),inSize);
inInd=1;
for k=1:nSamples
    z=U*X(k,:)'; %X projection on U space
    [a,b]=sort(abs(z)); %find biggest compunents
    maxInds = flip(b);
    maxInds = maxInds(1:nEvPlot);
    outMat(inInd,:) = X(k,:);
    outMat(inInd+(1:nEvPlot),:) = U(maxInds,:);
    outMat(inInd+(1+nEvPlot),:) = (U(maxInds,:)')*z(maxInds);
    inInd = inInd+nEvPlot+2;   
end

DisplayData(outMat, [],nSamples,nEvPlot+2)

