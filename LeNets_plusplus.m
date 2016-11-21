function net = LeNets_plusplus()
net = dagnn.DagNN();
ReLU = dagnn.ReLU('leak',0.1);
%stage1
Conv1_1 = dagnn.Conv('size',[5 5 1 32],'hasBias',true,'stride',[1,1],'pad',[2,2,2,2]);
net.addLayer('Conv1_1',Conv1_1,{'data'},{'conv1_1'},{'c1_1f','c1_1b'});
net.addLayer('relu1_1',ReLU,{'conv1_1'},{'conv1_1x'});
Conv1_2 = dagnn.Conv('size',[5 5 32 32],'hasBias',true,'stride',[1,1],'pad',[2,2,2,2]);
net.addLayer('Conv1_2',Conv1_2,{'conv1_1x'},{'conv1_2'},{'c1_2f','c1_2b'});
net.addLayer('relu1_2',ReLU,{'conv1_2'},{'conv1_2x'});
Pool1 = dagnn.Pooling('poolSize',[2 2],'stride',[2 2],'pad',[0,0,0,0]);
net.addLayer('pool1',Pool1,{'conv1_2x'},{'conv1_2xp'});
%stage2
Conv2_1 = dagnn.Conv('size',[5 5 32 64],'hasBias',true,'stride',[1,1],'pad',[2,2,2,2]);
net.addLayer('Conv2_1',Conv2_1,{'conv1_2xp'},{'conv2_1'},{'c2_1f','c2_1b'});
net.addLayer('relu2_1',ReLU,{'conv2_1'},{'conv2_1x'});
Conv2_2 = dagnn.Conv('size',[5 5 64 64],'hasBias',true,'stride',[1,1],'pad',[2,2,2,2]);
net.addLayer('Conv2_2',Conv2_2,{'conv2_1x'},{'conv2_2'},{'c2_2f','c2_2b'});
net.addLayer('relu2_2',ReLU,{'conv2_2'},{'conv2_2x'});
Pool2 = dagnn.Pooling('poolSize',[2 2],'stride',[2 2],'pad',[0,0,0,0]);
net.addLayer('pool2',Pool2,{'conv2_2x'},{'conv2_2xp'});
%stage3
Conv3_1 = dagnn.Conv('size',[5 5 64 128],'hasBias',true,'stride',[1,1],'pad',[2,2,2,2]);
net.addLayer('Conv3_1',Conv3_1,{'conv2_2xp'},{'conv3_1'},{'c3_1f','c3_1b'});
net.addLayer('relu3_1',ReLU,{'conv3_1'},{'conv3_1x'});
Conv3_2 = dagnn.Conv('size',[5 5 128 128],'hasBias',true,'stride',[1,1],'pad',[2,2,2,2]);
net.addLayer('Conv3_2',Conv3_2,{'conv3_1x'},{'conv3_2'},{'c3_2f','c3_2b'});
net.addLayer('relu3_2',ReLU,{'conv3_2'},{'conv3_2x'});
Pool3 = dagnn.Pooling('poolSize',[2 2],'stride',[2 2],'pad',[0,0,0,0]);
net.addLayer('pool3',Pool3,{'conv3_2x'},{'conv3_2xp'});

fc2Block = dagnn.Conv('size',[3 3 128 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc2',fc2Block,{'conv3_2xp'},{'fc2'},{'fc2f','fc2b'});

fcBlock = dagnn.Conv('size',[1 1 2 10],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('con_p',fcBlock,{'fc2'},{'prediction'},{'pf','pb'});

lossBlock = dagnn.Loss('loss', 'softmaxlog');
net.addLayer('softmaxloss',lossBlock,{'prediction','label'},'objective');
net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label'}, 'top1err') ;
net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
    'opts', {'topK',5}), ...
    {'prediction','label'}, 'top5err') ;
net.initParams();
net.params(net.getParamIndex('fc2f')).value = 0.1*net.params(net.getParamIndex('fc2f')).value;
net.addLayer('center_loss',Center_Loss(),{'fc2','label'},'objective_center',{'centers'});
rng(1);
net.params(net.getParamIndex('centers')).value = rand(2,10,'single');
%x = net.params(17).value(1,:);
%y = net.params(17).value(2,:);
%plot(x,y,'r*');
end

