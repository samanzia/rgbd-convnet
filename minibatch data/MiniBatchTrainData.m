load TrainRGBSplit1.mat
load TrainRGBSplit1LabelsMat.mat

%^^ shuffle the ones for which you will be training, rather than just
%extracting features

%store the mean channel value (if needed) that was previously calculated in VGGforw.jl
meanRed = mean(mean(mean(TrainRGBSplit1(:,:,1,:))));
meanGreen = mean(mean(mean(TrainRGBSplit1(:,:,2,:))));
meanBlue = mean(mean(mean(TrainRGBSplit1(:,:,3,:))));

fileCount = size(TrainRGBSplit1, 4);

%change this to change to batch size
batchSize = 5000;

batchCounter = 1;
batches = round(fileCount/batchSize);
for i=1:batchSize:fileCount
    if(batchCounter == batches)
        trainData = TrainRGBSplit1(:,:,:,i:end);
        labels = TrainRGBSplit1LabelsMat(:,i:end);
    else
        trainData = TrainRGBSplit1(:,:,:,i:batchSize+i-1);
        labels = TrainRGBSplit1LabelsMat(:,i:batchSize+i-1);
    end
    
    fileName = (strcat(strcat('TrainRGBSplit1Batch',num2str(batchCounter)),'.mat'));
    save(fileName, 'trainData', 'labels');
    
    batchCounter = batchCounter+1;
end

save('MeanChannelPixelValues','meanRed','meanGreen','meanBlue');