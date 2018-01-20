readdata =      true;

% read data from with read_data script from input_data
if readdata
  clear
  read_data
end

makedata =      true;
datacheck =     true;
ml =            true;
printdata =     true;
benchmark =     true;
analyzedata =   true;

% create inputs and targets of length ntr, remove rows = 0
if makedata
  make_data
end

% matlab train (neural network)
for trainMethod=6:6
  if ml
    load data_root/matlab_inputs_tagets

    inputs=transpose(inputs);
    targets=transpose(targets);

    ntr=size(inputs,2);

    trainSwitch = false;
    targets_train = 0;

% switch to test different methods
    if trainMethod == 1
%      neurons = [48,24]
      neurons = [5]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'trainbr');
    end

    if trainMethod == 2
%      neurons = [48,24]
      neurons = [12]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'trainbr');
    end

    if trainMethod == 3
%      neurons = [48,24]
      neurons = [24]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'trainbr');;
    end

   if trainMethod == 4
%      neurons = [48,24]
      neurons = [48]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'trainbr');
    end

    if trainMethod == 5
      neurons = [24,12]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'trainbr');
    end

    if trainMethod == 6
      neurons = [48,24]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'trainbr');
    end

    size(inputs)
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
      [net,tr]=train(net,inputs,targets_train,'useParallel','yes','useGPU','yes'); % train network
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