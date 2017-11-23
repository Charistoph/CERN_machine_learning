clear
temp=load('input_data/output.csv');
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
save tracks track
