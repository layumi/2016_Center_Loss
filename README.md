# Center_Loss
Idea is from [A Discriminative Feature Learning Approach for Deep Face Recognition ECCV16](http://ydwen.github.io/papers/WenECCV16.pdf)

The author's code can be find in following links:

*[caffe](https://github.com/kpzhang93/caffe-face);

*[mxnet](https://github.com/pangyupo/mxnet_center_loss)


I rebuild the code based on Matconvnet.
![](https://github.com/layumi/2016_Center_Loss/blob/master/demo.jpg)

# How to train & test
1.You may compile matconvnet first by running `gpu_compile.m`  (you need to change some setting in it)

For more compile information, you can learn it from www.vlfeat.org/matconvnet/install/#compiling

2.run `show.m` for test result.

3.If you want to train it by yourself, I have include the mnist data in the `data/imdb.mat`. (I have substract the mean of the data.) 

4.Use `train_id_net_mnist.m` to have fun~

# Tricks
I train the network without center_loss first. 
And I use its result to initial the center location. 
You can learn more from `update_center.m`.
