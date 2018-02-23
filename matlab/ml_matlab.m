readdata =      false;

% read data from with read_data script from input_data
if readdata
  clear
  read_data
end

makedata =      false;
ml =            true;
printdata =     true;
benchmark =     true;
analyzedata =   true;

% create inputs and targets of length ntr, remove rows = 0
if makedata
  make_data
else
  load('ml_input/ml_inputs_targets_20180127.mat')
   inputs=transpose(inputs);
   targets=transpose(targets);
   inp=inputs;
   targ=targets;
end

for i=1:60
  inputs=0;
  targets=0;
%  itr1=1+(i-1)*60000;
%  itr2=60000+(i-1)*60000;
  itr1=1;
  itr2=10000*i;
%  itr1=220000;
%  itr2=220000+10000*i;
  inputs=inp(:,itr1:itr2);
  targets=targ(:,itr1:itr2);

  % matlab train (neural network)
  for trainMethod=1:1
    if ml
  %    load data_root/matlab_inputs_tagets

      ntr=size(inputs,2);

      trainSwitch = false;
      targets_train = 0;
      inputs_train = 0;

  % switch to test different methods
    if trainMethod == 1
        neurons = 5
%        neurons = [48,24]
        targets_train=targets(1:3,:);
        inputs_train=inputs;
        net=feedforwardnet(neurons)
      end

      if trainMethod == 2
        neurons = 5
        targets_train=targets(1:3,:);
        inputs_train=inputs;
        net=feedforwardnet(neurons)
      end

      if trainMethod == 3
        neurons = 5
        targets_train=targets(1:3,:);
        inputs_train=inputs;
        net=feedforwardnet(neurons)
      end

      if trainMethod == 4
        neurons = 5
        targets_train=targets(1:3,:);
        inputs_train=inputs;
        net=feedforwardnet(neurons)
      end

      if trainMethod == 5
        neurons = 5
        targets_train=targets(1:3,:);
        inputs_train=inputs;
        net=feedforwardnet(neurons)
      end

      if trainMethod == 6
        neurons = 5
        targets_train=targets(1:3,:);
        inputs_train=inputs;
        net=feedforwardnet(neurons)
      end

  %    if trainMethod == 2
  %      neurons = 5
  %      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
  %      targets_train = targets;
  %      inputs_train=inputs;
  %      net=feedforwardnet(neurons)
  %    end
  %
  %    if trainMethod == 3
  %      neurons = [48,24]
  %      targets_train=targets(1:3,1:100000);
  %      inputs_train=inputs(:,1:100000);
  %      net=feedforwardnet(neurons)
  %    end
  %
  %    if trainMethod == 4
  %      neurons = [48,24]
  %      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
  %      targets_train = targets;
  %      inputs_train=inputs;
  %      net=feedforwardnet(neurons)
  %    end
  %
  %    if trainMethod == 5
  %      neurons = 5
  %      targets_train=targets(1:3,:);
  %      inputs_train=inputs;
  %      net=feedforwardnet(neurons)
  %    end
  %
  %  if trainMethod == 6
  %      neurons = 5
  %      targets_train=targets(1:3,:);
  %      inputs_train=inputs;
  %      net=feedforwardnet(neurons)
  %    end

    if trainMethod == 7
        neurons = 5
        targets_train=targets(1:3,:);
        inputs_train=inputs;
        net=feedforwardnet(neurons)
      end

    if trainMethod == 8
        neurons = 12
        targets_train=targets(1:3,:);
        inputs_train=inputs;
        net=feedforwardnet(neurons)
      end

    if trainMethod == 9
        neurons = 24
        targets_train=targets(1:3,:);
        inputs_train=inputs;
        net=feedforwardnet(neurons)
      end

      size(inputs)
      size(inputs_train)
      size(targets)
      size(targets_train)
      trainMethod

  % MATLAB Train
      if trainSwitch == false
    %    regression network with a single hidden layer with 12 neurons
    %    neurons = [12]
    %    net=feedforwardnet(neurons);

        % net = Newly trained network
        % tr = Training record (epoch and perf)
  %      [net,tr]=train(net,inputs,targets_train,'useParallel','yes','useGPU','yes'); % train network
        [net,tr]=train(net,inputs_train,targets_train); % train network
      end

  % MATLAB TrainNetwork: CONVOLUTIONAL NETWORK
      if trainSwitch == true

        options = trainingOptions('sgdm','MiniBatchSize',64);

        layers = [sequenceInputLayer(size(inputs,1))
                  softmaxLayer
                  regressionLayer];

    %    [trainedNet,traininfo] = trainNetwork(___)
        trainedNet=trainNetwork(inputs,targets,layers,options); % train network

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
      results_print
    end

    % compares MAD, Matlab & Tensorflow results
    if benchmark
      benchmark_check
    end

    % analyzed ml output
    if analyzedata
      analyze_ml_output
    end

  end
end