function h = DisplayData(X,displayRows,displayCols)
% function [h, display_array] = DisplayData(X, example_width,display_rows,display_cols)
% display 2D data in a grid
% input:
%   X: input samples packed in a matrix of (nSamples X sampleSize)
%   displayRows,displayCols: optional, how to arrange the grid
% output:
%   h: figure hanlde
%=========================================

colormap(gray);

% Compute rows, cols
[nSamples,inSize] = size(X);
exampleWidth = round(sqrt(size(X, 2)));
exampleHeight = (inSize / exampleWidth);

% Compute number of items to display
if ~exist('display_rows','var')
    displayRows = floor(sqrt(nSamples));
    displayCols = ceil(nSamples / displayRows);
end

% Between images padding
pad = 1;

% Setup blank display
displayArray = - ones(pad + displayRows * (exampleHeight + pad), ...
                       pad + displayCols * (exampleWidth + pad));

% Copy each example into a patch on the display array
currEx = 1;
for j = 1:displayRows
	for i = 1:displayCols
		if currEx > nSamples, 
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch
		max_val = max(abs(X(currEx, :)));
		displayArray(pad + (j - 1) * (exampleHeight + pad) + (1:exampleHeight), ...
		              pad + (i - 1) * (exampleWidth + pad) + (1:exampleWidth)) = ...
						reshape(X(currEx, :), exampleHeight, exampleWidth) / max_val;
		currEx = currEx + 1;
	end
	if currEx > nSamples, 
		break; 
	end
end

% Display Image
h = imagesc(displayArray, [-1 1]);

% Do not show axis
axis image off

drawnow;

end
