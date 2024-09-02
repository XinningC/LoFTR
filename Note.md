记录一下改LoFTR的过程以及数据流程

使用MegaDepth数据集进行训练，由于D2Net的数据已经没有了，需要对Indices进行修改，参考了这个Issue: https://github.com/zju3dv/LoFTR/issues/276 保存在 indices_process.py

创建环境：使用mkvirtualenv loftr --python=/usr/local/bin/python3.9 pip install -r requirements.txt 需要改变numpy版本，pip install numpy==1.23.2

调试训练：
这torch_lightning让人有点摸不到北，试试别的法子
只要输入对齐就可以
/mnt/ssd/home/chuxinning/LoFTR/src/datasets/megadepth.py __getitem__(self, idx)

后续需要做的：修改数据集，创建与XFeat相同种类的synthetic数据格式，直接在RGB_IR上进行训练

问题：
MegaDepth输入的可见光图像是经过reshape到设定尺寸的，但是其对应的深度图在原文代码中并没有体现reshape这个流程，而是直接padding成了（2000，2000）的大小。
这对RGB_IR数据集倒是影响不大，因为深度可以默认设定为1