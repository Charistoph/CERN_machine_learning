one = false;
two = true;

if one
    x = -5:0.1:5;

    for idx = 1:numel(x)
        if x(idx)<0
            y1(idx) = 0;

        else
            y1(idx) = 1;
        end 
    end

    for idx = 1:numel(x)
        y2(idx) = 1/(1+exp(-x(idx)));
    end

    figure
    subplot(1,3,1)       % add first plot in 2 x 1 grid
    plot(x,y1,'red')
    axis([-5 5 -1 1])
    title('1. Step Function')

    subplot(1,3,2)       % add second plot in 2 x 1 grid
    plot(x,y2,'red')       % plot using + markers
    axis([-5 5 -1 1])
    title('2. Sigmoid')

    subplot(1,3,3)       % add second plot in 2 x 1 grid
    plot(x,tanh(x),'red')       % plot using + markers
    axis([-5 5 -1 1])
    title('3. Hyperbolic Tangent')
end

if two
    x = -5:0.1:100;

   for idx = 1:numel(x)
        % y(idx) = exp(-x(idx)/20)+sin(x(idx)*2)*x(idx)/3000+x(idx)/2000;
        y(idx) = exp(-x(idx)/20)+(rand(1)-1)/15+x(idx)/1500+0.035;
    end 

    figure
    subplot(1,2,1)       % add first plot in 2 x 1 grid
    plot(x,exp(-x/20),'red')
    axis([0 100 0 1])
    title('1. Desired Loss')

    subplot(1,2,2)       % add second plot in 2 x 1 grid
    plot(x,y,'red')       % plot using + markers
    axis([0 100 0 1])
    title('2. Overfitted Loss')
end