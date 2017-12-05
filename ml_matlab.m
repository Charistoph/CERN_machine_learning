readdata = false;

if readdata
  clear
  read_data
end

makedata = false;
ml =       true;
print =    false;

if makedata
%  ntr=1
%  ntr=17443

  ntr
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

  size(inputs)
  size(targets)
  save data_root/matlab_inputs_tagets inputs targets ntr
end

if ml
  load data_root/matlab_inputs_tagets

  inputs=transpose(inputs);
  targets=transpose(targets);

  size(inputs)
  size(targets)

  net=feedforwardnet(12); % regression network with a single hidden layer with 12 neurons
%  net=fitnet(72,36,24,15,5);

  traininputs=inputs(:,1:ntr*19/20);
  traintargets=targets(:,1:ntr*19/20);
  testinputs=inputs(:,ntr*19/20+1:end);
  testtargets=targets(:,ntr*19/20+1:end);

%  options = trainingOptions('sgdm','MiniBatchSize',64)

  [net,tr]=train(net,traininputs,traintargets); % train network
%  [net,tr]=trainNetwork(traininputs,traintargets,net,options); % train network
  netout=net(testinputs); % compute net output

end

if print
  targets=zeros(size(netout));
  res=testtargets-netout;
  for i=1:size(netout,1)
    figure(i),clf,hist(res(i,:),50)
  end

  [mean(res1) std(res1)]
  [mean(res2) std(res2)]
end
