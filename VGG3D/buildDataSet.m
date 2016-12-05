path = ('/mnt/kufs/scratch/szia13/VGG3D/depth-6-background/'); %this is where your files from voxelize.m are stored
fileList = dir(fullfile(path,'*.mat'));
%fileCount =34855;%34855; %7033 %6982
dim1 = 224;
dim2 = 224;
dim3 = 6;
Train3DRGBSplit1 = (zeros(dim1,dim2,dim3,3,10));
Test3DRGBSplit1 = (zeros(dim1,dim2,dim3,3,10));
fileCounterTest = 1;
fileCounterTrain = 1;
label = 0;
load splits.mat;
foldnum = 1;

for z = 1:size(fileList,1)
	load(strcat(path,fileList(z).name));
    flag = (splits(label+1,foldnum) == str2num(instance));
	disp(fileList(z).name);
	
	if(flag)
		 Test3DSplit1(:,:,:,:,fileCounterTest) = output;
		 Test3DSplit1Labels(fileCounterTest) = label;
		 fileCounterTest = fileCounterTest+1;
	else
		Train3DSplit1(:,:,:,:,fileCounterTrain) = output;	
		Train3DSplit1Labels(fileCounterTrain) = label;
		fileCounterTrain = fileCounterTrain+1;
	end
			
			
     

    label = label+1;
    
end


Train3DSplit1Labels = Train3DSplit1Labels';
Train3DSplit1LabelsMat = full(sparse(Train3DSplit1Labels+1,1:fileCounterTrain-1,1,51,fileCounterTrain-1));

save('Train3DSplit1.mat','Train3DSplit1','-v7.3');
save('Train3DSplit1Labels.mat','Train3DSplit1Labels','-v7.3');
save('Train3DSplit1LabelsMat.mat','Train3DSplit1LabelsMat');
disp(fileCounterTrain)

Test3DSplit1Labels = Test3DSplit1Labels';
Test3DSplit1LabelsMat = full(sparse(Test3DSplit1Labels+1,1:fileCounterTest-1,1,51,fileCounterTest-1));

save('Test3DSplit1.mat','Test3DSplit1','-v7.3');
save('Test3DSplit1Labels.mat','Test3DSplit1Labels','-v7.3');
save('Test3DSplit1LabelsMat.mat','Test3DSplit1LabelsMat');
disp(fileCounterTest)
