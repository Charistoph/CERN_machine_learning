% THIS FILE CAN RUN ALONE
% CALLED ALSO WHILE RUNNING *** ml_matlab.m ***

clear
%temp=load('input_data/output.csv');
temp=load('input_data/output_20171204.csv');
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

for itr=1:ntr
  for i=4:5
    track(itr).tp(i)=track(itr).tp(i)-track(itr).mixt.par(i);
    % components
    if track(itr).ncomp == 12
      for j=1:12
        track(itr).comp(j).par(i)=track(itr).comp(j).par(i)-track(itr).mixt.par(i);
      end
    end

    track(itr).mixt.par(i)=0;
  end
end

%track(1).tp
%track(1).comp(1).par
%track(1).mixt.par

save tracks track
