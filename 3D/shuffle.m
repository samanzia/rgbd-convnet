load TrainSplit1;
load TrainSplit1Labels;

TrainSplit1  = TrainSplit1;
TrainSplit1Labels = TrainSplit1Labels; 

fileCountTrain = size(TrainSplit1,5);
x = randperm(fileCountTrain);
TrainSplit1Labels = TrainSplit1Labels(x);
TrainSplit1 = TrainSplit1(:,:,:,:,x);

TrainSplit1LabelsMat = full(sparse(TrainSplit1Labels+1,1:fileCountTrain,1,51,fileCountTrain));

disp(size(TrainSplit1))

save('TrainShuffleSplit1.mat','TrainSplit1','-v7.3');
save('TrainShuffleSplit1Labels.mat','TrainSplit1Labels','-v7.3');
save('TrainShuffleSplit1LabelsMat.mat','TrainSplit1LabelsMat');

load TestSplit1;
load TestSplit1Labels;

TestSplit1  = TestSplit1;
TestSplit1Labels = TestSplit1Labels; 

fileCountTest = size(TestSplit1,5);
x = randperm(fileCountTest);
TestSplit1Labels = TestSplit1Labels(x);
TestSplit1 = TestSplit1(:,:,:,:,x);

TestSplit1LabelsMat = full(sparse(TestSplit1Labels+1,1:fileCountTest,1,51,fileCountTest));

disp(size(TestSplit1))

save('TestShuffleSplit1.mat','TestSplit1','-v7.3');
save('TestShuffleSplit1Labels.mat','TestSplit1Labels','-v7.3');
save('TestShuffleSplit1LabelsMat.mat','TestSplit1LabelsMat');