clear
load('forCB.mat')
ntr=length(inputs);
%ntr=1;
counter = 0;
dt = datestr(now,'yyyy.mm.dd_HH_MM_SS');

targets = target

meaninp=mean(inputs)
inputs=inputs-meaninp;
stdinp=std(inputs)

for int1=1:length(inputs(:,1))
    for int2=2:length(inputs(1,:))-1
        inputs(int1,int2)=inputs(int1,int2)/stdinp(int2);
    end
end

mean(inputs)
std(inputs)

meaninp=mean(targets)
targets=targets-meaninp;
stdinp=std(targets)

for int1=1:length(targets(:,1))
    for int2=1:length(targets(1,:))
        targets(int1,int2)=targets(int1,int2)/stdinp(int2);
    end
end

mean(targets)
std(targets)

% saving
saveFunction(inputs, dt, 'inputs')
saveFunction(targets, dt, 'targets')

%===============================================================================
% main

% save to csv file
function savefunction_1=saveFunction(val, dt, name)
    path = strcat('ml_input_', dt);
    status = mkdir(path);
    path = strcat('ml_input_', dt, '/', name, '.csv');
    csvwrite(path,val);
end
