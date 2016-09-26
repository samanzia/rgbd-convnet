load TrainRGBSplit1Features1;
load TrainRGBSplit1Labels;

TrainSplit1  = TrainRGBSplit1Features1;
TrainSplit1Labels = TrainRGBSplit1Labels; 

fileCountTrain = size(TrainSplit1,4);
x = randperm(fileCountTrain);
TrainSplit1Labels = TrainSplit1Labels(x);
TrainSplit1 = TrainSplit1(:,:,:,x);

TrainSplit1LabelsMat = full(sparse(TrainSplit1Labels+1,1:fileCountTrain,1,51,fileCountTrain));

save('TrainShuffleSplit1Features1.mat','TrainSplit1','-v7.3');
save('TrainShuffleSplit1LabelsFeatures1.mat','TrainSplit1Labels','-v7.3');
save('TrainShuffleSplit1LabelsMatFeatures1.mat','TrainSplit1LabelsMat');

load TestRGBSplit1Features1;
load TestRGBSplit1Labels;

TestSplit1  = TestRGBSplit1Features1;
TestSplit1Labels = TestRGBSplit1Labels; 

fileCountTest = size(TestSplit1,4);
x = randperm(fileCountTest);
TestSplit1Labels = TestSplit1Labels(x);
TestSplit1 = TestSplit1(:,:,:,x);

TestSplit1LabelsMat = full(sparse(TestSplit1Labels+1,1:fileCountTest,1,51,fileCountTest));

save('TestShuffleSplit1Features1.mat','TestSplit1','-v7.3');
save('TestShuffleSplit1LabelsFeatures1.mat','TestSplit1Labels','-v7.3');
save('TestShuffleSplit1LabelsMatFeatures1.mat','TestSplit1LabelsMat');

load DevRGBSplit1Features1;
load DevRGBSplit1Labels;

DevSplit1  = DevRGBSplit1Features1;
DevSplit1Labels = DevRGBSplit1Labels; 

fileCountDev = size(DevSplit1,4);
x = randperm(fileCountDev);
DevSplit1Labels = DevSplit1Labels(x);
DevSplit1 = DevSplit1(:,:,:,x);

DevSplit1LabelsMat = full(sparse(DevSplit1Labels+1,1:fileCountDev,1,51,fileCountDev));

save('DevShuffleSplit1Features1.mat','DevSplit1','-v7.3');
save('DevShuffleSplit1LabelsFeatures1.mat','DevSplit1Labels','-v7.3');
save('DevShuffleSplit1LabelsMatFeatures1.mat','DevSplit1LabelsMat');