path = ('/mnt/kufs/scratch/szia13/dataset/datasetsplits/split1/train/'); %the path where the whole dataset is
subfolder = dir(path);

dimPad1 = 224; %the resize dimension
dimPad2 = 224;

fileCounterTest = 1;
fileCounterTrain = 1;
label = 0;

load splits.mat; %this file defines the splits
foldnum = 1; %split number

for x = 3:size(subfolder,1)
    subfolderpath = strcat(path,subfolder(x).name);
    subfolderpath = strcat(subfolderpath,'/');
    subsubfolder = dir(subfolderpath);
    
    
    for y = 3:size(subsubfolder,1)
        subsubfolderpath = strcat(subfolderpath,subsubfolder(y).name);
        instanceNum = strsplit(subsubfolder(y).name,'_');
        instanceNum = str2num(instanceNum{end});
        subsubfolderpath = strcat(subsubfolderpath,'/');
        fileList = dir(fullfile(subsubfolderpath,'*_crop.png'));
        flag = (splits(x-2,foldnum) == double(instanceNum));
		
        for z = 1:size(fileList,1)
            im = imread(strcat(subsubfolderpath,fileList(z).name));
		    ex1 = imresize(im,[dimPad1 dimPad2]);
            
            if(flag)
                 TestRGBSplit1(:,:,1:3,fileCounterTest) = ex1;
                 TestRGBSplit1Labels(fileCounterTest) = label;
                 fileCounterTest = fileCounterTest+1;
            else
                TrainRGBSplit1(:,:,1:3,fileCounterTrain) = ex1;
                TrainRGBSplit1Labels(fileCounterTrain) = label;
                fileCounterTrain = fileCounterTrain+1;
            end
        end
        
    end
    disp(subfolder(x).name);
    label = label+1;
    
end


TrainRGBSplit1Labels = TrainRGBSplit1Labels';
TrainRGBSplit1LabelsMat = full(sparse(TrainRGBSplit1Labels+1,1:fileCounterTrain-1,1,51,fileCounterTrain-1));

save('TrainRGBSplit1.mat','TrainRGBSplit1','-v7.3');
save('TrainRGBSplit1Labels.mat','TrainRGBSplit1Labels','-v7.3');
save('TrainRGBSplit1LabelsMat.mat','TrainRGBSplit1LabelsMat');
disp(fileCounterTrain)

TestRGBSplit1Labels = TestRGBSplit1Labels';
TestRGBSplit1LabelsMat = full(sparse(TestRGBSplit1Labels+1,1:fileCounterTest-1,1,51,fileCounterTest-1));

save('TestRGBSplit1.mat','TestRGBSplit1','-v7.3');
save('TestRGBSplit1Labels.mat','TestRGBSplit1Labels','-v7.3');
save('TestRGBSplit1LabelsMat.mat','TestRGBSplit1LabelsMat');
disp(fileCounterTest)
