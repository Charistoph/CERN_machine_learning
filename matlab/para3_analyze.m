mixtlen=10000;

mixt3(1:mixtlen)=0;
targ3(1:mixtlen)=0;

for i=1:mixtlen
    mixt3(i)=track(i).mixt.par(3);
    targ3(i)=track(i).tp(3);
end

scatter(mixt3,targ3)
hist(mixt3,100)
hist(targ3,100)