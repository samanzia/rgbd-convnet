% This files builds the dataset from the given input folders along with
% their labels by doing normalization and whitening.

path = ('/mnt/kufs/scratch/szia13/dataset/datasetsplits/split1/train/');
subfolder = dir(path);
fileCount = 34844;%41877 %34844;
dimPad1 = 148;
dimPad2 = 148;
filterSize = 9;
TrainRGBSplit1 = (zeros(dimPad1,dimPad2,3,fileCount));
fileCounter = 1;
label = 0;
TrainRGBSplit1Labels = zeros(fileCount,1);

for x = 3:size(subfolder,1)
    subfolderpath = strcat(path,subfolder(x).name);
    subfolderpath = strcat(subfolderpath,'/');
    subsubfolder = dir(subfolderpath);
    
    for y = 3:size(subsubfolder,1)
        subsubfolderpath = strcat(subfolderpath,subsubfolder(y).name);
        subsubfolderpath = strcat(subsubfolderpath,'/');
        fileList = dir(fullfile(subsubfolderpath,'*_crop.png'));
        for z = 1:size(fileList,1)
            im = imread(strcat(subsubfolderpath,fileList(z).name));
			ex1 = imresize(im,[dimPad1 dimPad2]);
            ex1 = preprocessImg(ex1,filterSize);
            TrainRGBSplit1(:,:,:,fileCounter) = ex1;
            TrainRGBSplit1Labels(fileCounter) = label;
            fileCounter = fileCounter+1;
        end
        
    end
    disp(subfolder(x).name);
    label = label+1;
    
end

x = randperm(fileCount);
TrainRGBSplit1Labels = TrainRGBSplit1Labels(x);
TrainRGBSplit1 = TrainRGBSplit1(:,:,:,x);

TrainRGBSplit1LabelsMat = full(sparse(TrainRGBSplit1Labels+1,1:fileCount,1,51,fileCount));

TrainRGBSplit1 = TrainRGBSplit1;
save('TrainRGBSplit1.mat','TrainRGBSplit1','-v7.3');
save('TrainRGBSplit1Labels.mat','TrainRGBSplit1Labels','-v7.3');
save('TrainRGBSplit1LabelsMat.mat','TrainRGBSplit1LabelsMat');

