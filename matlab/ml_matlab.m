readdata =      false; % usually false, only create data once

% read data from with read_data script from input_data
if readdata
  clear
  read_data
end

datacheck =     false; % condition used in make_data.m
makedata =      false; % usually false, only create data once

% % don't delete these
% keepvars = {'inputs','targets'};

% % delete these
% delvars = setdiff(who,keepvars);
% clear(delvars{:},'delvars');

ml =            true;
trainSwitch =   true; % true = train, false = trainNetwork
printdata =     true;
benchmark =     true;
analyzedata =   false; % not important

% create inputs and targets of length ntr, remove rows = 0
if makedata
  make_data
  
  inputs=transpose(inputs);
  targets=transpose(targets);
else
  % load('/Users/christoph/Documents/coding/CERN_input_data/ml_inputs_targets_ex1_20180305.mat')

  % inputs=transpose(inputs);
  % targets=transpose(targets);
end

for i=54:1000
  % itr1=1+(i-1)*10000
  % itr2=10000+(i-1)*10000
  itr1=1
  itr2=10000*i

  % matlab train (neural network)
  if ml
    for trainMethod=1:1

      targets_train = 0;
      inputs_train = 0;

      targets_train=targets(1:3,itr1:itr2);
      inputs_train=inputs(:,itr1:itr2);

      if trainSwitch
  % MATLAB Train

        if trainMethod == 1
            neurons = 5
  %          neurons = [48,24]
            net=feedforwardnet(neurons)
        end

        % net = Newly trained network
        % tr = Training record (epoch and perf)
  %      [net,tr]=train(net,inputs,targets_train,'useParallel','yes','useGPU','yes'); % train network
        [net,tr]=train(net,inputs_train,targets_train); % train network

      else
  % MATLAB TrainNetwork: CONVOLUTIONAL NETWORK
  % doesn't work with MATLAB

        options = trainingOptions('sgdm',...
                                  'MiniBatchSize',64,...
                                  'MaxEpochs',20,...
                                  'Plots','training-progress');

        layers = [sequenceInputLayer(72)
                  fullyConnectedLayer(1)
                  regressionLayer()];

    %    [trainedNet,traininfo] = trainNetwork(___)
        trainedNet=trainNetwork(inputs_train,targets_train,layers,options); % train network

      end

      netout=net(inputs); % compute net output of trained net to given input

      if size(netout,1)<5
        newmat = zeros(5,size(netout,2));
        newmat(1:size(netout,1),:)=netout;
        netout=newmat;
      end

    end

    % print histos & save outputs
    if printdata
      try
        results_print
      catch
        disp('results_print failed')
      end
    end

    % compares MAD, Matlab & Tensorflow results
    if benchmark
      try
        benchmark_check
      catch
        disp('benchmark_check failed')
      end
    end

    % analyzed ml output
    if analyzedata
      try
        analyze_ml_output
      catch
        disp('analyze_ml_output failed')
      end
    end

  end
end