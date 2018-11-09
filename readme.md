复现时的几点疑问：

1.雷达点云是截取了摄像头范围内的点，这样是否合理，比如换一种做法，直接截取一定范围内的点;

2.目前的实现方式是将点存在voxel（batch,channels,z,y,x）里，直接pointnet、conv3d，没有选voxel;

