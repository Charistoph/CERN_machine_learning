% fplot(@(x) (1+exp(-x))^(-1),[-5 5],'b', 'LineWidth', 1)
% grid on
% hold on
% ylabel('sigmoid');
% fplot(@(x) tanh(x),[-5 5],'g-.', 'LineWidth', 1)
% hold on
% ylabel('tanh');
% fplot(@(x) y(x),[-5 5],'r--', 'LineWidth', 1)
% hold on
% ylabel('step function');

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
title('Step Function')

subplot(1,3,2)       % add second plot in 2 x 1 grid
plot(x,tanh(x),'red--')       % plot using + markers
axis([-5 5 -1 1])
title('TanH')

subplot(1,3,3)       % add second plot in 2 x 1 grid
plot(x,y2,'red-.')       % plot using + markers
axis([-5 5 -1 1])
title('Sigmoid')