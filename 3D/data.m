load TrainRGBSplit2Features1                                               
size(TrainRGBSplit2Features1)

x = reshape(TrainRGBSplit2Features1,[],size(TrainRGBSplit2Features1,4));
size(x)

load TrainRGBSplit1Features3D                 
size(TrainRGBSplit1Features3D)

%y = reshape(TrainRGBSplit1Features3D,[],size(TrainRGBSplit1Features3D,5));
y = TrainRGBSplit1Features3D;
z = [x;y];
size(z)

TrainRGBSplit1Features4 = z;
save('TrainRGBSplit1Features4.mat','TrainRGBSplit1Features4', '-v7.3');
 
 
 
load TestRGBSplit2Features1                                                
size(TestRGBSplit2Features1)

x = reshape(TestRGBSplit2Features1,[],size(TestRGBSplit2Features1,4));
size(x)

load TestRGBSplit1Features3D                
size(TestRGBSplit1Features3D)

%y = reshape(TestRGBSplit1Features3D,[],size(TestRGBSplit1Features3D,5));
y = TestRGBSplit1Features3D;
z = [x;y];
size(z)

TestRGBSplit1Features4 = z;
save('TestRGBSplit1Features4.mat','TestRGBSplit1Features4','-v7.3');

%load DevRGBSplit1Features                                                
%size(DevRGBSplit1Features)

%x = reshape(DevRGBSplit1Features,[],6982);
%size(x)

%load DevRGBSplit1Features1                 
%size(DevRGBSplit1Features1)

%y = reshape(DevRGBSplit1Features1,[],6982);
%z = [x;y];
%size(z)

%DevRGBSplit1Features2 = z;
%save('DevRGBSplit1Features2.mat','DevRGBSplit1Features2');
 
 