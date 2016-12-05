load TrainRGBSplit1Features1;
load Train3DSplit1Labels;

TrainSplit1  = TrainRGBSplit1Features1;
TrainSplit1Labels = Train3DSplit1Labels; 

fileCountTrain = size(TrainSplit1,4);
x = randperm(fileCountTrain);
TrainSplit1Labels = TrainSplit1Labels(x);
TrainSplit1 = TrainSplit1(:,:,:,x);

TrainSplit1LabelsMat = full(sparse(TrainSplit1Labels+1,1:fileCountTrain,1,51,fileCountTrain));

save('TrainShuffle1Split1.mat','TrainSplit1','-v7.3');
save('TrainShuffle1Split1Labels.mat','TrainSplit1Labels','-v7.3');
save('TrainShuffle1Split1LabelsMat.mat','TrainSplit1LabelsMat');

load TestRGBSplit1Features1;
load Test3DSplit1Labels;

TestSplit1  = TestRGBSplit1Features1;
TestSplit1Labels = Test3DSplit1Labels; 

fileCountTest = size(TestSplit1,4);
x = randperm(fileCountTest);
TestSplit1Labels = TestSplit1Labels(x);
TestSplit1 = TestSplit1(:,:,:,x);

TestSplit1LabelsMat = full(sparse(TestSplit1Labels+1,1:fileCountTest,1,51,fileCountTest));

save('TestShuffle1Split1.mat','TestSplit1','-v7.3');
save('TestShuffle1Split1Labels.mat','TestSplit1Labels','-v7.3');
save('TestShuffle1Split1LabelsMat.mat','TestSplit1LabelsMat');

%load Dev3DSplit1Features1;
%load Dev3DSplit1Labels;

%DevSplit1  = Dev3DSplit1Features1;
%DevSplit1Labels = Dev3DSplit1Labels; 

%fileCountDev = size(DevSplit1,4);
%x = randperm(fileCountDev);
%DevSplit1Labels = DevSplit1Labels(x);
%DevSplit1 = DevSplit1(:,:,:,x);

%DevSplit1LabelsMat = full(sparse(DevSplit1Labels+1,1:fileCountDev,1,51,fileCountDev));

%save('DevShuffleSplit1Features1.mat','DevSplit1','-v7.3');
%save('DevShuffleSplit1LabelsFeatures1.mat','DevSplit1Labels','-v7.3');
%save('DevShuffleSplit1LabelsMatFeatures1.mat','DevSplit1LabelsMat');