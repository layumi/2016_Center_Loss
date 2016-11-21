function net = update_center( net,imdb )
net_pre = load('./data/LeNets_plusplus/net-epoch-20.mat');
net_pre = dagnn.DagNN.loadobj(net_pre.net);
data = imdb.images.data(:,:,find(imdb.images.set==1));
label = imdb.images.labels(find(imdb.images.set==1));
net_pre.mode = 'test' ;
net_pre.move('gpu');
net_pre.conserveMemory = true;
center = zeros(2,10,'single');
count = zeros(1,10,'single');
for i = 1:1000 %numel(label)
    disp(i);
    data_i = data(:,:,i);
    net_pre.vars(net_pre.getVarIndex('fc2')).precious = true;
    net_pre.eval({'data',gpuArray(data_i)});
    xy = net_pre.vars(net_pre.getVarIndex('fc2')).value;
    center(:,label(i)) = center(:,label(i)) + gather(reshape(xy,[],1));
    count(label(i)) = count(label(i))+1;
end
count = repmat(count,2,1);
center = center./count;
net.params(net.getParamIndex('centers')).value = center./20;%
%x = net.params(17).value(1,:)./100;
%y = net.params(17).value(2,:)./100;
%plot(x,y,'r*');
end

