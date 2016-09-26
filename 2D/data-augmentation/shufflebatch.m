load TrainRGBSplit1;
load TrainRGBSplit1Labels;

TrainSplit1  = TrainRGBSplit1;
TrainSplit1Labels = TrainRGBSplit1Labels;

clear TrainSplit1, TrainSplit1Labels;

fileCountTrain = size(TrainSplit1,4);
x = randperm(fileCountTrain);
TrainSplit1Labels = TrainSplit1Labels(x);
TrainSplit1 = TrainSplit1(:,:,:,x);

TrainSplit1LabelsMat = full(sparse(TrainSplit1Labels+1,1:fileCountTrain,1,51,fileCountTrain));

disp(size(TrainSplit1))

TrainSplit1Batch1 = TrainSplit1(:,:,:,1:40000);
TrainSplit1Batch2 = TrainSplit1(:,:,:,40001:80000);
TrainSplit1Batch3 = TrainSplit1(:,:,:,80001:120000);
TrainSplit1Batch4 = TrainSplit1(:,:,:,120001:160000);
TrainSplit1Batch5 = TrainSplit1(:,:,:,160001:200000);
TrainSplit1Batch6 = TrainSplit1(:,:,:,200001:240000);
TrainSplit1Batch7 = TrainSplit1(:,:,:,240001:280000);
TrainSplit1Batch8 = TrainSplit1(:,:,:,280001:320000);
TrainSplit1Batch9 = TrainSplit1(:,:,:,320001:36000);
TrainSplit1Batch10 = TrainSplit1(:,:,:,360001:400000);
TrainSplit1Batch11 = TrainSplit1(:,:,:,400001:440000);
TrainSplit1Batch12 = TrainSplit1(:,:,:,440001:480000);
TrainSplit1Batch13 = TrainSplit1(:,:,:,520001:560000);
TrainSplit1Batch14 = TrainSplit1(:,:,:,560001:600000);
TrainSplit1Batch15 = TrainSplit1(:,:,:,600001:640000);
TrainSplit1Batch16 = TrainSplit1(:,:,:,640001:680000);
TrainSplit1Batch17 = TrainSplit1(:,:,:,680001:720000);
TrainSplit1Batch18 = TrainSplit1(:,:,:,720001:760000);
TrainSplit1Batch19 = TrainSplit1(:,:,:,800001:840000);
TrainSplit1Batch20 = TrainSplit1(:,:,:,840001:880000);
TrainSplit1Batch21 = TrainSplit1(:,:,:,880001:920000);
TrainSplit1Batch22 = TrainSplit1(:,:,:,920001:960000);
TrainSplit1Batch23 = TrainSplit1(:,:,:,1000001:1040000);
TrainSplit1Batch24 = TrainSplit1(:,:,:,1040001:end);

save('TrainShuffleSplit1Batch1.mat','TrainSplit1Batch1','-v7.3');
save('TrainShuffleSplit1Batch2.mat','TrainSplit1Batch2','-v7.3');
save('TrainShuffleSplit1Batch3.mat','TrainSplit1Batch3','-v7.3');
save('TrainShuffleSplit1Batch4.mat','TrainSplit1Batch4','-v7.3');
save('TrainShuffleSplit1Batch5.mat','TrainSplit1Batch5','-v7.3');
save('TrainShuffleSplit1Batch6.mat','TrainSplit1Batch6','-v7.3');
save('TrainShuffleSplit1Batch7.mat','TrainSplit1Batch7','-v7.3');
save('TrainShuffleSplit1Batch8.mat','TrainSplit1Batch8','-v7.3');
save('TrainShuffleSplit1Batch9.mat','TrainSplit1Batch9','-v7.3');
save('TrainShuffleSplit1Batch10.mat','TrainSplit1Batch10','-v7.3');
save('TrainShuffleSplit1Batch11.mat','TrainSplit1Batch11','-v7.3');
save('TrainShuffleSplit1Batch12.mat','TrainSplit1Batch12','-v7.3');
save('TrainShuffleSplit1Batch13.mat','TrainSplit1Batch13','-v7.3');
save('TrainShuffleSplit1Batch14.mat','TrainSplit1Batch14','-v7.3');
save('TrainShuffleSplit1Batch15.mat','TrainSplit1Batch15','-v7.3');
save('TrainShuffleSplit1Batch16.mat','TrainSplit1Batch16','-v7.3');
save('TrainShuffleSplit1Batch17.mat','TrainSplit1Batch17','-v7.3');
save('TrainShuffleSplit1Batch18.mat','TrainSplit1Batch18','-v7.3');
save('TrainShuffleSplit1Batch19.mat','TrainSplit1Batch19','-v7.3');
save('TrainShuffleSplit1Batch20.mat','TrainSplit1Batch20','-v7.3');
save('TrainShuffleSplit1Batch21.mat','TrainSplit1Batch21','-v7.3');
save('TrainShuffleSplit1Batch22.mat','TrainSplit1Batch22','-v7.3');
save('TrainShuffleSplit1Batch23.mat','TrainSplit1Batch23','-v7.3');
save('TrainShuffleSplit1Batch24.mat','TrainSplit1Batch24','-v7.3');

save('TrainShuffleSplit1Labels.mat', 'TrainSplit1LabelsMat');

