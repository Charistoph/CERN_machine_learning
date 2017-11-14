clear
load('tracks.mat')
ntr=length(track);
%ntr=1;
counter = 0;
dt = datestr(now,'yyyy.mm.dd_HH_MM_SS');

%===============================================================================
% main

for itr=1:ntr
    if track(itr).ncomp == 12
        itr;
        counter = counter + 1;
        allparas(1,counter*13-12)=track(counter).ncomp;
        allparas(2:6,counter*13-12)=track(counter).mixt.par;
        for int=1:track(counter).ncomp
            allparas(1,counter*13-12+int)=track(counter).comp(int).weight;
            allparas(2:6,counter*13-12+int)=track(counter).comp(int).par;
        end
    end
end

means = mean(allparas,2);
stds = transpose(std(transpose(allparas)));
allparas(2:6,:)=allparas(2:6,:)-means(2:6);

for int2=1:length(allparas(1,:))
    for int3=2:length(allparas(:,1))
        allparas(int3,int2)=allparas(int3,int2)/stds(int3);
    end
end

allparas;

% formating
c1 = 0;
for int2=1:length(allparas(1,:))
    if mod(int2,13) == 1
        c1 = c1 + 1;
        labels(1:5,c1) = allparas(2:6,int2);
    end
end

c1 = 0;
for int2=1:length(allparas(1,:))
    if mod(int2,13) ~= 1
        c1 = c1 + 1;
        inputs(1:6,c1) = allparas(1:6,int2);
    end
end

% counts
string('tracks not used = ')
itr-counter
ntr
allparaslength = length(allparas(1,:))/13
length(labels(1,:))
length(inputs(1,:))/12

% saving
saveFunction(transpose(inputs), dt, 'inputs')
saveFunction(transpose(labels), dt, 'labels')

function savefunction_1=saveFunction(val, dt, name)
    path = strcat('ml_input_', dt);
    status = mkdir(path);
    path = strcat('ml_input_', dt, '/', name, '.csv');
    csvwrite(path,val);
end
