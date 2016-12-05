load Train3DSplit1;

rchannel = Train3DSplit1(:,:,:,1,:);
gchannel = Train3DSplit1(:,:,:,2,:);
bchannel = Train3DSplit1(:,:,:,3,:);

%r = 138.95952;
%g = 131.80444;
%5b = 128.681;
%disp(r);
%disp(g);
%disp(b);

%rchannel(rchannel~=0) = rchannel(rchannel~=0) -r;
%gchannel(gchannel~=0) = rchannel(gchannel~=0) -g;
%bchannel(bchannel~=0) = rchannel(bchannel~=0) -b;

%Train3DSplit1(:,:,:,1,:)=rchannel;
%Train3DSplit1(:,:,:,2,:)=gchannel;
%Train3DSplit1(:,:,:,3,:)=bchannel;


Train3DSplit1Reshaped = reshape(Train3DSplit1, 224 ,224, 18, size(Train3DSplit1,5));
save('Train3DSplit1Reshaped.mat','Train3DSplit1Reshaped','-v7.3');
%clear rchannel, gchannel, bchannel;
clear Train3DSplit1;
load Test3DSplit1;

%rchannel = Test3DSplit1(:,:,:,1,:);
%gchannel = Test3DSplit1(:,:,:,2,:);
%bchannel = Test3DSplit1(:,:,:,3,:);

%rchannel(rchannel~=0) = rchannel(rchannel~=0) -r;
%gchannel(gchannel~=0) = rchannel(gchannel~=0) -g;
%bchannel(bchannel~=0) = rchannel(bchannel~=0) -b;

%Test3DSplit1(:,:,:,1,:)=rchannel;
%Test3DSplit1(:,:,:,2,:)=gchannel;
%Test3DSplit1(:,:,:,3,:)=bchannel;


Test3DSplit1Reshaped = reshape(Test3DSplit1, 224 ,224, 18, size(Test3DSplit1,5));
save('Test3DSplit1Reshaped.mat','Test3DSplit1Reshaped','-v7.3');

%load Dev3DSplit1;

%Dev3DSplit1Reshaped = reshape(Dev3DSplit1, 224 ,224, 15, size(Dev3DSplit1,5));
%save('Dev3DSplit1Reshaped.mat','Dev3DSplit1Reshaped','-v7.3');

