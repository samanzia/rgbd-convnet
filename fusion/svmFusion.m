addpath('/mnt/kufs/scratch/szia13/liblinear-2.1/matlab/')
 

load('TrainRGBSplit1Features4.mat');
load('TrainSplit1Labels.mat');
load('TestRGBSplit1Features4.mat');
load('TestSplit1Labels.mat');

trainLabel = TrainSplit1Labels;
trainData = TrainRGBSplit1Features4;
trainData = trainData';
fileCount = size(trainLabel,1);
x = randperm(fileCount);
trainLabel1 = trainLabel(x,1);
trainData1 = trainData(x,:);
trainData1 = sparse(trainData1);

clear TrainRGBSplit1Features;


disp(size(trainData));
disp(size(trainLabel));


testLabel = TestSplit1Labels;
testData = TestRGBSplit1Features4;
testData = testData';

fileCount = size(testLabel,1);
x = randperm(fileCount);
testLabel = testLabel(x,1);
testData = testData(x,:);
testData = sparse(testData);

disp(size(testData));
disp(size(testLabel));

c=-20:1:20;
%% randomize and split data
acc = [];testdataVal = [];testdataValLabels = [];
testdataTest = [];testdataTestLabels = [];

valacc = [];testacc = [];bestC = 0;val = 0;cMat = [];

  
for x=1:size(c,2)
    model = train(trainLabel1, trainData1, ['-q -c ',num2str(2^(c(x)))]);
	[predictedlabels,accuracy, prob]  = predict(trainLabel1,trainData1, model);
	[predictedlabels,valacc1, prob]  = predict(testLabel,testData, model);
%	[predictedlabels,valacc1, prob]  = predict(devLabel,devData, model);
%	disp(valacc1);
	disp(accuracy);
	disp(c(x));
   valacc = [valacc; valacc1];
end

[best, val] = max(valacc);
rp = 2^(c(val));
tic;
model = train(trainLabel1, trainData1, ['-q -c ',num2str(rp)]);
disp(toc);
[predictedlabels,accuracy, prob]  = predict(testLabel,testData, model);
disp(c(val));
disp('Testing accuracy: ');
disp(accuracy);
[predictedlabels,accuracy, prob]  = predict(trainLabel1,trainData1, model);
disp('Training accuracy: ');
disp(accuracy);
