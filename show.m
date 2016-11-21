load('./data/mnist/imdb.mat');
test_data = images.data(:,:,find(images.set==3));
test_label = images.labels(find(images.set==3));

% normal one
netStruct = load('./data/LeNets_plusplus/net-epoch-20.mat');
net1 = dagnn.DagNN.loadobj(netStruct.net);
net1.mode = 'test' ;
net1.move('gpu') ;
%center loss
netStruct = load('./data/LeNets_plusplus_center/net-epoch-20.mat');
net2 = dagnn.DagNN.loadobj(netStruct.net);
net2.mode = 'test' ;
net2.move('gpu') ;

ten_color = [1,0,0;0,1,0;0,0,1;...
    0.5,0,0;0,0.5,0;0,0,0.5;...
    0.5,0.5,0;0,0.5,0.5;0.5,0,0.5;...
    0.3,0.3,0.3];

figure(1);
clf;
%normal
subplot(1,2,1);
for i = 1:numel(test_label)
    data_i = test_data(:,:,i);
    net1.vars(net1.getVarIndex('fc2')).precious = true;
    net1.eval({'data',gpuArray(data_i)});
    xy = net1.vars(net1.getVarIndex('fc2')).value;
    x = xy(1); y = xy(2);
    label_i = test_label(i);
    plot(x,y,'Color',ten_color(label_i,:),'Marker','*');
    hold on;
end
title('normal');
%center
subplot(1,2,2);
for i = 1:numel(test_label)
    data_i = test_data(:,:,i);
    net2.vars(net2.getVarIndex('fc2')).precious = true;
    net2.eval({'data',gpuArray(data_i)});
    xy = net2.vars(net2.getVarIndex('fc2')).value;
    x = xy(1); y = xy(2);
    label_i = test_label(i);
    plot(x,y,'Color',ten_color(label_i,:),'Marker','*');
    hold on;
end
title('center');

