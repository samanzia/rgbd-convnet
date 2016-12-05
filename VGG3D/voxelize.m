path = ('Y:\split1full\train\'); %enter the path to your dataset here
subfolder = dir(path);
fileCounter =0;
label = 0;
for x = 3:size(subfolder,1)
    subfolderpath = strcat(path,subfolder(x).name);
    subfolderpath = strcat(subfolderpath,'\');
    subsubfolder = dir(subfolderpath);
    
    for y = 3:size(subsubfolder,1)
        subsubfolderpath = strcat(subfolderpath,subsubfolder(y).name);
        subsubfolderpath = strcat(subsubfolderpath,'\');
        fileList = dir(fullfile(subsubfolderpath,'*_depthcrop.png'));
		fileListMask = dir(fullfile(subsubfolderpath,'*_maskcrop.png'));
        fileListRGB = dir(fullfile(subsubfolderpath,'*_crop.png'));
        instance = strsplit(subsubfolder(y).name,'_');
        instance  = instance{end};
        disp(instance);
        for z = 1:size(fileList,1)
            fname = (strcat(subsubfolderpath,fileList(z).name));
            mname = (strcat(subsubfolderpath,fileListMask(z).name));
            cname = (strcat(subsubfolderpath,fileListRGB(z).name));
            imDepth = imread(fname);
            imMask = imread(mname);
            im = imread(cname);
            imDepth = double(imDepth);
            imDepth(imDepth == 0) = NaN;
            imDepth = inpaint_nans(imDepth,2);
            imDepth = uint16(imDepth);
            imDepth = imresize(imDepth,[224 224],'nearest');
            imMask = imresize(imMask,[224 224]);
            im = imresize(im,[224 224]);
            output = voxel3D(imDepth,imMask, im);
            imMask3D = zeros(size(imMask,1),size(imMask,2),3);
            imMask3D(:,:,1) = imMask;
            imMask3D(:,:,2) = imMask;
            imMask3D(:,:,3) = imMask;
            imX = im.*uint8(~imMask3D);

            output(:,:,6,1:3) = imX;
            
            finalIm = reshape(sum(output,3),[224 224 3]);
           
            fileFinal = strsplit(fileList(z).name,'_');
            fileFinal{end} = 'depth6unmaskedinterpol.mat';
            fileFinal = strjoin(fileFinal,'_');
            if(sum(finalIm(:) == im(:)) ~= 224*224*3)
                 disp('error');
                 disp(fileFinal);
            end
            
            save(fileFinal, 'output','label','instance');
            fileCounter = fileCounter+1;
        end
        
    end
    disp(subfolder(x).name);
    label = label+1;
    
end

disp(fileCounter)
