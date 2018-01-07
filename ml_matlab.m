
readdata = true;

% read data from with read_data script from input_data
if readdata
  clear
  read_data
end

makedata =  true;
ml =        true;
print =     true;
benchmark = true;

% create inputs and targets of length ntr, remove rows = 0
if makedata
  inputs=zeros(ntr,72);
  targets=zeros(ntr,5);

  for itr=1:ntr
    if track(itr).ncomp==12
      for i=1:track(itr).ncomp
        for j=1:size(track(1).comp(i).par,1)
        % inputs
          inputs(itr,i*6-5+j)=track(itr).comp(i).par(j);
        end
        inputs(itr,i*6-5)=track(itr).comp(i).weight;
      end
    end
    % targets
    for k=1:size(track(1).tp,1)
      targets(itr,k)=track(itr).tp(k);
    end
  end

  % How many rows are = 0
  count=0
  pos=[];
  for i=1:size(inputs,1)
    if inputs(i,3)==0
      count=count+1;
      i;
      pos=[pos,i];
    end
  end

  % Remove rows = 0
  count2=0
  for i=1:size(inputs,1)
    inputs(i-count2,:)=inputs(i,:);
    targets(i-count2,:)=targets(i,:);
    for j=1:size(pos,2)
      if pos(j)==i
        count2=count2+1;
      end
    end
  end
  % Cut away empty rows
  inputs=inputs(1:ntr-count2,:);
  targets=targets(1:ntr-count2,:);

  size(inputs)
  size(targets)
  save data_root/matlab_inputs_tagets inputs targets ntr
end

% matlab train (neural network)
if ml
  load data_root/matlab_inputs_tagets

  inputs=transpose(inputs);
  targets=transpose(targets);

  size(inputs)
  size(targets)

  ntr=size(inputs,2);
%  ntr=1
%  ntr=17443

%  targets(4:5,:)=rand(size(targets(4:5,:)))*10^-10;

%  net=feedforwardnet(12); % regression network with a single hidden layer with 12 neurons
%  net=feedforwardnet([24,12]);
  neurons = 24
  net=feedforwardnet(neurons);

  options = trainingOptions('sgdm','MiniBatchSize',64)

  % net = Newly trained network
  % tr = Training record (epoch and perf)
  [net,tr]=train(net,inputs,targets); % train network
%  [net,tr]=trainNetwork(traininputs,traintargets,net,options); % train network
  netout=net(inputs); % compute net output of trained net to given input

end

% print histos & save outputs
if print
  results=zeros(size(netout,1),2);
  res=targets-netout;
  for i=1:size(netout,1)
    figure(i),clf,hist(res(i,:),50);
    results(i,:)=[mean(res(i,:)),std(res(i,:))];
  end

  dt = datestr(now,'yyyy.mm.dd-HH:MM:SS')
  path = strcat(pwd,'/ml_output_matlab/',dt)
  resultspath = strcat(path,'/results.csv')
  benchmark_resultspath = strcat(path,'/benchmark_results.csv')

  mkdir(path)

  saveas(figure(1),[path '/figure_' num2str(1) '.fig']);
  saveas(figure(2),[path '/figure_' num2str(2) '.fig']);
  saveas(figure(3),[path '/figure_' num2str(3) '.fig']);
  saveas(figure(4),[path '/figure_' num2str(4) '.fig']);
  saveas(figure(5),[path '/figure_' num2str(5) '.fig']);

  csvwrite('ml_output_matlab/results.csv',results)
  csvwrite(resultspath,results)
end

if benchmark
  benchmark_check;
  csvwrite(benchmark_resultspath,benchmark_result);

  ml_train_log  = strcat(dt, {':     '});
  for i=1:5
    ml_train_log = strcat(ml_train_log, num2str(matlab_worse_than_baseline(i)), {'    '});
  end
  ml_train_log  = strcat(ml_train_log, {'  - neurons: '}, num2str(neurons))

  dlmwrite('ml_output_matlab/ml_train_log.csv',ml_train_log,'delimiter','','-append');
end
