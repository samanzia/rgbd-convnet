path = ('/mnt/kufs/scratch/szia13/dataset/datasetsplits/split1/train/');
subfolder = dir(path);
dimPad1 = 256;
dimPad2 = 256;

fileCounterTrain = 1;
label = 0;
filterSize = 3;

load splits.mat;
foldnum = 1;

for x = 3:size(subfolder,1)
    subfolderpath = strcat(path,subfolder(x).name);
    subfolderpath = strcat(subfolderpath,'/');
    subsubfolder = dir(subfolderpath);
    
    
    for y = 3:size(subsubfolder,1)
        subsubfolderpath = strcat(subfolderpath,subsubfolder(y).name);
        instanceNum = strsplit(subsubfolder(y).name,'_');
        %disp(instanceNum);
        instanceNum = str2num(instanceNum{end});
        subsubfolderpath = strcat(subsubfolderpath,'/');
        fileList = dir(fullfile(subsubfolderpath,'*_crop.png'));
        flag = (splits(x-2,foldnum) == double(instanceNum));
        %fileListMask = dir(fullfile(subsubfolderpath,'*_maskcrop.png'));
        for z = 1:size(fileList,1)
            
            randomVector1 = floor((32 - 1)*rand(15,1) + 1);
            randomVector2 = floor((32 - 1)*rand(15,1) + 1);
            im = imread(strcat(subsubfolderpath,fileList(z).name));
            ex1 = imresize(im,[dimPad1 dimPad2]);
            if(~flag)
                for i=1:15
                    a = randomVector1(i);
                    b = randomVector2(i);
                    
                    ex11 = ex1(a:a+224-1,b:b+224-1,:);
                    ex12 = flipdim(ex11,2);
                    
                    TrainRGBSplit1(:,:,1:3,fileCounterTrain) = ex11;
                    TrainRGBSplit1Labels(fileCounterTrain) = label;
                    fileCounterTrain = fileCounterTrain+1;
                    
                    TrainRGBSplit1(:,:,1:3,fileCounterTrain) = ex12;
                    TrainRGBSplit1Labels(fileCounterTrain) = label;
                    fileCounterTrain = fileCounterTrain+1;
                    
                end
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
disp(label)