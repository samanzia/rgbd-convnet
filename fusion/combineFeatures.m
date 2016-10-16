load TrainRGBSplit1Features1                                                
size(TrainRGBSplit1Features1)

x = reshape(TrainRGBSplit1Features1,[],size(TrainRGBSplit1Features1,4));
size(x)

load TrainRGBSplit1Features3D                 
size(TrainRGBSplit1Features3D)

y = TrainRGBSplit1Features3D; 
z = [x;y];
size(z)

TrainRGBSplit1Features4 = z;
save('TrainRGBSplit1Features4.mat','TrainRGBSplit1Features4', '-v7.3');
 
 
 
load TestRGBSplit1Features1                                                
size(TestRGBSplit1Features1)

x = reshape(TestRGBSplit1Features1,[],size(TestRGBSplit1Features1,4));
size(x)

load TestRGBSplit1Features3D                
size(TestRGBSplit1Features3D)

y = TestRGBSplit1Features3D;
z = [x;y];
size(z)

TestRGBSplit1Features4 = z;
save('TestRGBSplit1Features4.mat','TestRGBSplit1Features4');

 