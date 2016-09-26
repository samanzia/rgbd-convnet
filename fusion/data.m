load TrainRGBSplit1Features2                                                
size(TrainRGBSplit1Features2)

x = reshape(TrainRGBSplit1Features2,[],size(TrainRGBSplit1Features2,4));
size(x)

load TrainRGBSplit1Features3                 
size(TrainRGBSplit1Features3)

y = reshape(TrainRGBSplit1Features3,[],size(TrainRGBSplit1Features3,4));
z = [x;y];
size(z)

TrainRGBSplit1Features4 = z;
save('TrainRGBSplit1Features4.mat','TrainRGBSplit1Features4', '-v7.3');
 
 
 
load TestRGBSplit1Features2                                                
size(TestRGBSplit1Features2)

x = reshape(TestRGBSplit1Features2,[],size(TestRGBSplit1Features2,4));
size(x)

load TestRGBSplit1Features3                
size(TestRGBSplit1Features3)

y = reshape(TestRGBSplit1Features3,[],size(TestRGBSplit1Features3,4));
z = [x;y];
size(z)

TestRGBSplit1Features4 = z;
save('TestRGBSplit1Features4.mat','TestRGBSplit1Features4');

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
 
 