function [imgResult] = preprocessImg(img, neighborhoodSize)
% Function for preprocessing an image by doing mean and dev normalization
% followed by whitening the image

    % convert into blocks for local mean and dev normalization
    patches = [];
    for row = 1:size(img,3)
        patches = [patches; im2col(img(:,:,row),[neighborhoodSize neighborhoodSize])];
    end

    %normalization and whitening
    patches = double(patches);
    patches = patches';
    patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
    patches = bsxfun(@minus, patches, params.whiten.M) * params.whiten.P;
    patches = patches';
    %reshape the result to image again to input to the conv layer
    imgResult = zeros(size(img));
    for y = 1:size(img,3)
        sIndex = (y-1)*neighborhoodSize*neighborhoodSize + 1;
        eIndex = y*neighborhoodSize*neighborhoodSize;
        B = patches(sIndex:eIndex,:);
        Bsize = sqrt((size(B,1)*size(B,2)));
        A = col2im(B,[neighborhoodSize neighborhoodSize],[Bsize Bsize],'distinct');
        i = 1;j = 1;
        for col = 1:size(img,1)
            i = 1;
            for row = 1:size(img,1)
                imgResult(row,col,y) = A(i,j);
                if(i+neighborhoodSize > size(A,1))
                    i = i+1;
                else
                    i = i+neighborhoodSize;
                end
            end

            if(j+neighborhoodSize > size(A,2))
                j = j+1;
            else
                j = j+neighborhoodSize;
            end
        end
    end
end