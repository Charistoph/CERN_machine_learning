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
for trainMethod=20:23
  if ml
    load data_root/matlab_inputs_tagets

    inputs=transpose(inputs);
    targets=transpose(targets);

    ntr=size(inputs,2);

    trainSwitch = false;
    targets_train = 0;

% switch to test different methods
     if trainMethod == 1
      neurons = [48,24]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=feedforwardnet(neurons);
    end

    if trainMethod == 2
      neurons = [48,24]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=feedforwardnet(neurons);
    end

    if trainMethod == 3
      neurons = [48,24]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=feedforwardnet(neurons);
    end

   if trainMethod == 4
      neurons = [48,24]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'trainlm');
    end

    if trainMethod == 5
      neurons = [48,24]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'traingd');    
    end

    if trainMethod == 6
      neurons = [48,24]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'traingdm');
    end

    if trainMethod == 7
      neurons = [48,24]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'traingdx');
    end

   if trainMethod == 8
      neurons = [5]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'trainlm');
    end

    if trainMethod == 9
      neurons = [5]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'traingd');    
    end

    if trainMethod == 10
      neurons = [5]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'traingdm');
    end

    if trainMethod == 11
      neurons = [5]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'traingdx');
    end

   if trainMethod == 12
      neurons = [12]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'trainlm');
    end

    if trainMethod == 13
      neurons = [12]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'traingd');    
    end

    if trainMethod == 14
      neurons = [12]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'traingdm');
    end

    if trainMethod == 15
      neurons = [12]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'traingdx');
    end

   if trainMethod == 16
      neurons = [24]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'trainlm');
    end

    if trainMethod == 17
      neurons = [24]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'traingd');    
    end

    if trainMethod == 18
      neurons = [24]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'traingdm');
    end

    if trainMethod == 19
      neurons = [24]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'traingdx');
    end

   if trainMethod == 20
      neurons = [48]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'trainlm');
    end

    if trainMethod == 21
      neurons = [48]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'traingd');    
    end

    if trainMethod == 22
      neurons = [48]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'traingdm');
    end

    if trainMethod == 23
      neurons = [48]
      targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;
      targets_train = targets;
      net=fitnet(neurons,'traingdx');
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