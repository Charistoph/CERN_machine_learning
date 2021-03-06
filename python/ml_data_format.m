ntr=length(track);
%ntr=1;
counter = 0;
dt = datestr(now,'yyyy.mm.dd_HH.MM.SS');

%===============================================================================
% main

%%%% ---- Marker 1 ---- %%%%
% loop over all tracks
for itr=1:ntr
    % eliminates all tracks which don`t have 12 components
    if track(itr).ncomp == 12
        itr;
        counter = counter + 1;
        % allparas(1,1) = number of components
%        allparas(1,counter*13-12)=track(counter).ncomp;
        % simparas(1:5,:) = targets (simulated 5 parameters)
        simparas(1:5,counter)=track(counter).tp;
        for int=1:track(counter).ncomp
            % recoparas(1,2:13) = component weights
            recoparas(1,counter*12-12+int)=track(counter).comp(int).weight;
            % recoparas(2:6,2:13) = component 5 parameters
            recoparas(2:6,counter*12-12+int)=track(counter).comp(int).par;
        end
    end
end

% Sim formatting to Targets
%simparas(1:5,1:5);

%meansim=mean(simparas,2);
%simparas=simparas-meansim;
%stdsim=transpose(std(transpose(simparas)));

%for int2=1:length(simparas(1,:))
%    for int3=1:length(simparas(:,1))-2
%        simparas(int3,int2)=simparas(int3,int2)/stdsim(int3);
%    end
%end

%%%% ---- Marker 2 ---- %%%%
%simparas(1:5,1:5);
targets = transpose(simparas);

% Reco formatting to Inputs
recoparas(1:5,1:5);

meanreco=mean(recoparas,2);
recoparas(2:6,:)=recoparas(2:6,:)-meanreco(2:6);
stdreco=transpose(std(transpose(recoparas)));

for int2=1:length(recoparas(1,:))
    for int3=2:length(recoparas(:,1))
        recoparas(int3,int2)=recoparas(int3,int2)/stdreco(int3);
    end
end

recoparas(1:5,1:5);
inputs = transpose(recoparas);

%%%% ---- Marker 3 ---- %%%%
% length counts
ntr
itr-counter
length(targets(:,1))
length(inputs(:,1))/12

%%%% ---- Marker 4 ---- %%%%
% mean test
mean(simparas,2)
a=mean(recoparas,2);
a(2:6)

%%%% ---- Marker 5 ---- %%%%
% std test
a=transpose(std(transpose(simparas)));
a(1:5)
b=transpose(std(transpose(recoparas)));
b(2:6)

%%%% ---- Marker 6 ---- %%%%
% saving
saveFunction(inputs, dt, 'inputs')
saveFunction(targets, dt, 'targets')

%===============================================================================
% main

function savefunction_1=saveFunction(val, dt, name)
    path = strcat('ml_input/', dt);
    status = mkdir(path);
    path = strcat('ml_input/', dt, '/', name, '.csv');
    csvwrite(path,val);
end
