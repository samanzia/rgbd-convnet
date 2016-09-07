load('filters.mat');
filters = filters';
numFilters = size(filters,2);
dimFilters = sqrt(size(filters,1)/3);
filters = reshape(filters,[dimFilters dimFilters 3 numFilters]);
figure;
for x = 1:numFilters
    subplot(8,16,x);
    imagesc(filters(:,:,2,x));
    axis tight;
    axis off;
end