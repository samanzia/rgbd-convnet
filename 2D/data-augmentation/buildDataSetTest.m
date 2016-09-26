path = ('/mnt/kufs/scratch/szia13/dataset/datasetsplits/split1/train/');
subfolder = dir(path);

dimPad1 = 256;
dimPad2 = 256;
fileCounterTest = 1;
label = 0;
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
            im = imread(strcat(subsubfolderpath,fileList(z).name));
            ex1 = imresize(im,[dimPad1 dimPad2]);
            ex2 = ex1(1:224,1:224,:);
            ex3 = ex1(1:224,33:end,:);
            ex4 = ex1(33:end,33:end,:);
            ex5 = ex1(33:end, 1:224,:);
            ex6 = ex1(17:240, 17:240, :);
            ex7 = flipdim(ex2,2);
            ex8 = flipdim(ex3,2);
            ex9 = flipdim(ex4,2);
            ex10 = flipdim(ex5,2);
            ex11 = flipdim(ex6,2);
            if(flag)
                TestRGBSplit1(:,:,1:3,fileCounterTest) = ex2;
                TestRGBSplit1Labels(fileCounterTest) = label;
                fileCounterTest = fileCounterTest+1;
                
                TestRGBSplit1(:,:,1:3,fileCounterTest) = ex3;
                TestRGBSplit1Labels(fileCounterTest) = label;
                fileCounterTest = fileCounterTest+1;
                
                TestRGBSplit1(:,:,1:3,fileCounterTest) = ex4;
                TestRGBSplit1Labels(fileCounterTest) = label;
                fileCounterTest = fileCounterTest+1;
                
                TestRGBSplit1(:,:,1:3,fileCounterTest) = ex5;
                TestRGBSplit1Labels(fileCounterTest) = label;
                fileCounterTest = fileCounterTest+1;
                
                TestRGBSplit1(:,:,1:3,fileCounterTest) = ex6;
                TestRGBSplit1Labels(fileCounterTest) = label;
                fileCounterTest = fileCounterTest+1;
                
                TestRGBSplit1(:,:,1:3,fileCounterTest) = ex7;
                TestRGBSplit1Labels(fileCounterTest) = label;
                fileCounterTest = fileCounterTest+1;
                
                TestRGBSplit1(:,:,1:3,fileCounterTest) = ex8;
                TestRGBSplit1Labels(fileCounterTest) = label;
                fileCounterTest = fileCounterTest+1;
                
                TestRGBSplit1(:,:,1:3,fileCounterTest) = ex9;
                TestRGBSplit1Labels(fileCounterTest) = label;
                fileCounterTest = fileCounterTest+1;
                
                TestRGBSplit1(:,:,1:3,fileCounterTest) = ex10;
                TestRGBSplit1Labels(fileCounterTest) = label;
                fileCounterTest = fileCounterTest+1;
                
                TestRGBSplit1(:,:,1:3,fileCounterTest) = ex10;
                TestRGBSplit1Labels(fileCounterTest) = label;
                fileCounterTest = fileCounterTest+1;
                
            end
            
        end
    end
    disp(subfolder(x).name);
    label = label+1;
end

TestRGBSplit1Labels = TestRGBSplit1Labels';
TestRGBSplit1LabelsMat = full(sparse(TestRGBSplit1Labels+1,1:fileCounterTest-1,1,51,fileCounterTest-1));

save('TestRGBSplit1.mat','TestRGBSplit1','-v7.3');
save('TestRGBSplit1Labels.mat','TestRGBSplit1Labels','-v7.3');
save('TestRGBSplit1LabelsMat.mat','TestRGBSplit1LabelsMat');
disp(fileCounterTest)
disp(label)
