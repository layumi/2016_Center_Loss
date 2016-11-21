function train_id_net_minist(varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load('./data/mnist/imdb.mat') ;
set = imdb.images.set;
imdb.images.set(set==3) = 2;
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = LeNets_plusplus();
net.conserveMemory = true;
net.meta.normalization.averageImage = imdb.images.data_mean;
net = update_center(net,imdb);
% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------
opts.train.averageImage = net.meta.normalization.averageImage;
opts.train.batchSize = 128;
opts.train.continue = true; 
opts.train.gpus = 1;
opts.train.prefetch = false ;
opts.train.expDir = './data/LeNets_plusplus_center' ;
opts.train.derOutputs = {'objective', 1,'objective_center', 0.1} ;
opts.train.learningRate = [0.001*ones(1,20)] ;
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

% Call training function in MatConvNet
[net,info] = cnn_train_dag(net, imdb, @getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch,opts)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,batch) ; 
labels = imdb.images.labels(:,batch) ;
%im = bsxfun(@minus,im,opts.averageImage);
im = reshape(im,28,28,1,[]);
inputs = {'data',gpuArray(im),'label',labels};
