one = true;
two = false;

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

    for idx = 1:numel(x)
        if x(idx)<0
            y3(idx) = 0;

        else
            y3(idx) = x(idx);
        end 
    end

    figure
    subplot(2,2,1)
    plot(x,y1,'red')
    axis([-5 5 -1.5 1.5])
    title('1. Step Function')

    subplot(2,2,2)
    plot(x,y2,'red')
    axis([-5 5 -1.5 1.5])
    title('2. Sigmoid')

    subplot(2,2,3)
    plot(x,tanh(x),'red')
    axis([-5 5 -1.5 1.5])
    title('3. Hyperbolic Tangent')

    subplot(2,2,4)
    plot(x,y3,'red')
    axis([-5 5 -1.5 1.5])
    title('4. RELU')
end

if two
    x = -5:0.1:100;

   for idx = 1:numel(x)
        % y(idx) = exp(-x(idx)/20)+sin(x(idx)*2)*x(idx)/3000+x(idx)/2000;
        y(idx) = exp(-x(idx)/20)+(rand(1)-1)/15+x(idx)/1500+0.035;
    end 

    figure
    subplot(1,2,1)
    plot(x,exp(-x/20),'red')
    axis([0 100 0 1])
    title('1. Desired Loss')

    subplot(1,2,2)
    plot(x,y,'red')
    axis([0 100 0 1])
    title('2. Overfitted Loss')
end