simulate=false;
N=100000;
if simulate
    inputs=zeros(N,5);
    target=zeros(N,2);
    for idata=1:N;
        a=unifrnd(0.8,1.2);
        b=unifrnd(3.8,4.2);
        x=linspace(0,b/a,5);
        inputs(idata,1:5)=-a*x.^2+b*x;
        inputs(idata,6:10)=x;
        target(idata,1)=b/2/a;
        target(idata,2)=-a*target(idata,1)^2+b*target(idata,1);
    end
    inputs=inputs';
    target=target';
    save forCB inputs target
else
    load forCB
    net=fitnet(12); % regressionm network with a single hidden layer with 12 neurons
    traininputs=inputs(:,1:N/2);
    traintarget=target(:,1:N/2);
    testinputs=inputs(:,N/2+1:end);
    testtarget=target(:,N/2+1:end);
    [net,tr]=train(net,traininputs,traintarget); % train network
    netout=net(testinputs); % compute net output
    res1=testtarget(1,:)-netout(1,:);
    res2=testtarget(2,:)-netout(2,:);
    figure(1),clf,hist(res1,50)
    figure(2),clf,hist(res2,50)
    [mean(res1) std(res1)]
    [mean(res2) std(res2)]
end
