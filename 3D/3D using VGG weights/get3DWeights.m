%converts 2-D weights to 3-D
function w3D = get3DWeights(w)
%w is assumed to be a 4-D weight here, returns a 5-D weight
w3D = zeros(size(w,1),size(w,2), size(w,1), size(w,3), size(w,4));
% assumes a square filter which gets turned to a cubic filter
for x = 1:size(w,3)
    for y = 1:size(w,4)
        for z = 1:size(w,1)
            w3D(:,:,z,x,y) = w(:,:,x,y);
        end
    end
end
end