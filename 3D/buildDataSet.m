%builds 3-D dataset using RGB and Depth files 
path = ('/mnt/kufs/scratch/szia13/dataset/datasetsplits/split1/train/');
subfolder = dir(path);
dim1 = 30; % specify dimensions here
dim2 = 30;
dim3 = 30;
TrainRGBSplit1 = (zeros(dim1,dim2,dim3,3,10));
TestRGBSplit1 = (zeros(dim1,dim2,dim3,3,10));
fileCounterTest = 1;
fileCounterTrain = 1;
label = 0;
load splits.mat; 
foldnum = 1; %specify split number here 



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
        fileListDepth = dir(fullfile(subsubfolderpath,'*_depthcrop.png'));
		fileListMask =  dir(fullfile(subsubfolderpath,'*_maskcrop.png'));
        flag = (splits(x-2,foldnum) == double(instanceNum));
		
        for z = 1:size(fileList,1)
            im = imread(strcat(subsubfolderpath,fileList(z).name));
            imDepth = imread(strcat(subsubfolderpath,fileListDepth(z).name));
			imMask = imread(strcat(subsubfolderpath,fileListMask(z).name));
			imDepth = double(imDepth);
            imDepth(imDepth == 0) = NaN;
            imDepth = inpaint_nans(imDepth,2);
            imDepth = uint16(imDepth);
            im = imresize(im,[dim1 dim2],'nearest');
            imDepth = imresize(imDepth,[dim1 dim2],'nearest');
			imMask = imresize(imMask,[dim1 dim2],'nearest');
			%imDepth = imDepth.*uint16(imMask);
            imDepthz = imDepth;
            maxDepth = max(max(imDepth));
            imDepth(imDepth == 0) = 10000;
            minDepth = min(min(imDepth));
			
		
				imFinal = uint8(zeros(dim1,dim2,maxDepth,3));
				
				for i = 1:size(imDepthz,1)
					for j = 1:size(imDepthz,2)
						if(imDepthz(i,j) > 0)
							imFinal(i,j,imDepthz(i,j),:) = im(i,j,:);
						end
					end
				end
				
				imFinal = imFinal(:,:,minDepth:maxDepth,:);
				ex1 = uint8(zeros(dim1,dim2,dim3,3));
				
				for b = 1:dim1
					temp = permute(imFinal(b,:,:,:),[2 3 4 1]);
					temp = imresize(temp,[dim1 dim3]);
					ex1(b,:,:,:) = temp;
				end
			
				mean1 = mean(mean(mean(mean(ex1))));
				std1 = std(std(std(std(double(ex1)))));
				ex1 = (ex1 - mean1)/std1;
				
				if(flag)
					 TestSplit1(:,:,:,:,fileCounterTest) = ex1;
					 TestSplit1Labels(fileCounterTest) = label;
					 fileCounterTest = fileCounterTest+1;
				else
					TrainSplit1(:,:,:,:,fileCounterTrain) = ex1;	
					TrainSplit1Labels(fileCounterTrain) = label;
					fileCounterTrain = fileCounterTrain+1;
				end
			
			
        end
        
    end
    disp(subfolder(x).name);
    label = label+1;
    
end

% saving 3-D dataset 
TrainSplit1Labels = TrainSplit1Labels';
TrainSplit1LabelsMat = full(sparse(TrainSplit1Labels+1,1:fileCounterTrain-1,1,51,fileCounterTrain-1));

save('TrainSplit1.mat','TrainSplit1','-v7.3');
save('TrainSplit1Labels.mat','TrainSplit1Labels','-v7.3');
save('TrainSplit1LabelsMat.mat','TrainSplit1LabelsMat');
disp(fileCounterTrain)

TestSplit1Labels = TestSplit1Labels';
TestSplit1LabelsMat = full(sparse(TestSplit1Labels+1,1:fileCounterTest-1,1,51,fileCounterTest-1));

save('TestSplit1.mat','TestSplit1','-v7.3');
save('TestSplit1Labels.mat','TestSplit1Labels','-v7.3');
save('TestSplit1LabelsMat.mat','TestSplit1LabelsMat');
disp(fileCounterTest)
