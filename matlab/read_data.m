% THIS FILE CAN RUN ALONE
% CALLED ALSO WHILE RUNNING *** ml_matlab.m ***

clear
%temp=load('input_data/output.csv');
% mydir  = pwd;
% idcs   = strfind(mydir,'/');
% newdir = mydir(1:idcs(end)-1); 

% output_20171204
% output_20180127
% output_ex1_20180305

% filepath = '/input_data/output_ex1_20180305.csv';
% newpath = strcat(newdir,filepath);
% newpath = '/Users/christoph/Documents/coding/CERN_input_data/output_ex1_180126_081601_20180305.csv';
newpath = '/Users/christoph/Documents/coding/CERN_input_data/output_ex2_180129_165048_20180307.csv';
temp=load(newpath);

%%%% ---- Marker 1 ---- %%%%
l=length(temp)
ntr=0;
iline=1;
while iline<l
    ntr=ntr+1;
    track(ntr).nevt=temp(iline);
    iline=iline+1;
    track(ntr).ncomp=temp(iline);
    iline=iline+1;
    track(ntr).tp=temp(iline:iline+4);
    iline=iline+7;
    track(ntr).mixt.par=temp(iline:iline+4);
    iline=iline+5;
    track(ntr).mixt.covm=temp(iline:iline+14);
    iline=iline+15;
    for icomp=1:track(ntr).ncomp
        iline=iline+1;
        track(ntr).comp(icomp).weight=temp(iline);
        iline=iline+1;
        track(ntr).comp(icomp).par=temp(iline:iline+4);
        iline=iline+5;
        track(ntr).comp(icomp).covm=temp(iline:iline+14);
        iline=iline+15;
    end
end

%track(1).tp
%track(1).comp(1).par
%track(1).mixt.par

%for itr=1:ntr
%  for i=4:5
%    track(itr).tp(i)=track(itr).tp(i)-track(itr).mixt.par(i);
%    % components
%    if track(itr).ncomp == 12
%      for j=1:12
%        track(itr).comp(j).par(i)=track(itr).comp(j).par(i)-track(itr).mixt.par(i);
%      end
%    end
%
%    track(itr).mixt.par(i)=0;
%  end
%end

%track(1).tp
%track(1).comp(1).par
%track(1).mixt.par

% file to big to save
% save tracks track
