function voxel = voxel3D(imDepth, depthMask, Img)
    depthImg = imDepth.*uint16(depthMask);
    voxel = uint8(zeros(224,224,5,3));
    for i = 1:size(Img,1)
        for j = 1:size(Img,2)
            vdepth = depthImg(i,j);
            if(vdepth > 0)
                if(vdepth <= 706)
                    depthLevel = 1;
                elseif(vdepth <= 756)
                    depthLevel = 2;
                elseif(vdepth <= 804)
                    depthLevel = 3;
                elseif(vdepth <= 864)
                    depthLevel = 4;
                else
                    depthLevel  = 5;
                end
                voxel(i,j,depthLevel,1:3) = Img(i,j,1:3);
            end
        end
    end
end