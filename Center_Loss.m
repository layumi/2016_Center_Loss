classdef Center_Loss < dagnn.Loss    
    methods
        function outputs = forward(obj, inputs, params)
            % feature 2*128
            % center 2*10
            % label 128
            batchsize = size(inputs{1},4);
            feature = reshape(inputs{1},[],batchsize);
            center = params{1};
            label = inputs{2};
            gt_feature = [];
            for i = 1:batchsize
                gt = label(i);  
                gt_feature = cat(2,gt_feature,center(:,gt));  % gt_feature 2*128
            end
            ff = (feature - gt_feature).^2;
            outputs{1} = sum(ff(:));
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            batchsize = size(inputs{1},4);
            dim = size(inputs{1},3);
            feature = reshape(inputs{1},[],batchsize);
            center = params{1};
            label = inputs{2};
            gt_feature = [];
            gt_sum = gpuArray(zeros(dim,10,'single'));  % 10 indicate class
            count = zeros(1,10,'single'); %
            for i = 1:batchsize
                gt = label(i);  
                gt_feature = cat(2,gt_feature,center(:,gt));  % gt_feature 2*128
                gt_sum(:,gt) =  gt_sum(:,gt) + (center(:,gt)-feature(:,i));
                count(gt) = count(gt) + 1;
            end
            derInputs{1} = derOutputs{1}*(inputs{1}-reshape(gt_feature,1,1,[],batchsize));
            derInputs{2} = [];
            no_zero = count>0;
            count = repmat(count,2,1);
            gt_sum(no_zero) = gt_sum(no_zero)/count(no_zero);
            derParams{1} = derOutputs{1}*gt_sum;
        end
        
        function obj = Center_Loss(varargin)
            obj.load(varargin) ;
        end
    end
end
