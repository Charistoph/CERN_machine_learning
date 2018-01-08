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

% create inputs and targets of length ntr, remove rows = 0
if makedata
  make_data
end

% neurons_list = [5,12,24,48,72];

% for i=1:size(neurons_list,2)
%  neurons = neurons_list(i)

% matlab train (neural network)
if ml
  load data_root/matlab_inputs_tagets

  inputs=transpose(inputs);
  targets=transpose(targets);

  size(inputs)
  size(targets)

  ntr=size(inputs,2);

%  targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;

%  regression network with a single hidden layer with 12 neurons
  neurons = [12,5]
  net=feedforwardnet(neurons);

  options = trainingOptions('sgdm','MiniBatchSize',64)

  % net = Newly trained network
  % tr = Training record (epoch and perf)
  [net,tr]=train(net,inputs,targets); % train network
%  [net,tr]=trainNetwork(traininputs,traintargets,net,options); % train network
  netout=net(inputs); % compute net output of trained net to given input

end

% print histos & save outputs
if printdata
  results_print
end

% compares MAD, Matlab & Tensorflow results
if benchmark
  benchmark_check
end
