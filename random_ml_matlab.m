len_1 = 17000;
len_2 = 70;
inputs = rand(len_1,len_2);
for i=1:len_1
  for j=1:len_2/14
    a = 0;
    for k=1:14
      a = a + inputs(i,(j-1)*14+k);
    end
    targets(i,j) = a/14;
  end
end

inputs=transpose(inputs);
targets=transpose(targets);

neurons = [12,5]
net=feedforwardnet(neurons);

options = trainingOptions('sgdm','MiniBatchSize',64)

[net,tr]=train(net,inputs,targets); % train network

netout=net(inputs); % compute net output of trained net to given input


results=zeros(size(netout,1),2);
res=targets-netout;
for i=1:size(netout,1)
  figure(i),clf,hist(res(i,:),50);
  results(i,:)=[mean(res(i,:)),std(res(i,:))];
end
