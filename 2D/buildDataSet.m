path = ('/mnt/kufs/scratch/szia13/dataset/datasetsplits/split1/train/');
subfolder = dir(path);

dimPad1 = 224;
dimPad2 = 224;

fileCounterTest = 1;
fileCounterTrain = 1;
label = 0;
filterSize = 3;

load splits.mat;
foldnum = 2;

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
                 TestRGBSplit2(:,:,1:3,fileCounterTest) = ex1;
                 TestRGBSplit2Labels(fileCounterTest) = label;
                 fileCounterTest = fileCounterTest+1;
            else
                TrainRGBSplit2(:,:,1:3,fileCounterTrain) = ex1;	
                TrainRGBSplit2Labels(fileCounterTrain) = label;
                fileCounterTrain = fileCounterTrain+1;
            end
        end
        
    end
    disp(subfolder(x).name);
    label = label+1;
    
end


TrainRGBSplit2Labels = TrainRGBSplit2Labels';
TrainRGBSplit2LabelsMat = full(sparse(TrainRGBSplit2Labels+1,1:fileCounterTrain-1,1,51,fileCounterTrain-1));

save('TrainRGBSplit2.mat','TrainRGBSplit2','-v7.3');
save('TrainRGBSplit2Labels.mat','TrainRGBSplit2Labels','-v7.3');
save('TrainRGBSplit2LabelsMat.mat','TrainRGBSplit2LabelsMat');
disp(fileCounterTrain)

TestRGBSplit2Labels = TestRGBSplit2Labels';
TestRGBSplit2LabelsMat = full(sparse(TestRGBSplit2Labels+1,1:fileCounterTest-1,1,51,fileCounterTest-1));

save('TestRGBSplit2.mat','TestRGBSplit2','-v7.3');
save('TestRGBSplit2Labels.mat','TestRGBSplit2Labels','-v7.3');
save('TestRGBSplit2LabelsMat.mat','TestRGBSplit2LabelsMat');
disp(fileCounterTest)
