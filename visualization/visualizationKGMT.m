clc
clear all

% Opening Samples Data:
wslFilePath = '\\wsl.localhost\Ubuntu-20.04\home\nic\dev\research\cudaSBMP\build\samples.csv';
samples = readmatrix(wslFilePath);

% Opening Paren Relations Data:
wslFilePath = '\\wsl.localhost\Ubuntu-20.04\home\nic\dev\research\cudaSBMP\build\parentRelations.csv';
parentRelations = readmatrix(wslFilePath);

% Initialize Variables:
numDisc = 10;
sampleSize = 7;
stateSize = 4;
controlSize = 3;
L = 1;  % Car length

% Create a figure for dynamic plotting
figure;
hold on;
grid on;
title('Dynamic Car Path');
xlabel('X Position');
ylabel('Y Position');
plot(samples(1,1),samples(1,2), 'ko', 'MarkerFaceColor', 'k');

for i = 2:size(parentRelations)
    if parentRelations(i) == -1
        break;
    end
    segmentX = [];
    segmentY = [];
    x0 = samples((parentRelations(i) + 1), 1:stateSize);
    u = samples(i, stateSize+1:sampleSize-1);
    dt = samples(i, sampleSize);
    stepSize = dt / numDisc;
    for step = 1:numDisc
        x1 = dynamicsPropagator(x0, u, stepSize, L);
        segmentX = [segmentX, x1(1)];
        segmentY = [segmentY, x1(2)];
        x0 = x1;
    end
    plot(segmentX, segmentY, '-.', 'Color', 'k');
    %plot(x1(1), x1(2), 'ko', 'MarkerFaceColor', 'k');
    drawnow;
    pause(.01);
end

% Car dynamics function
function x1 = dynamicsPropagator(x0, u, dt, length_car)
    % x: state vector [x; y; psi; v]
    % u: control input vector [a; steeringAngle]
    % dt: time step
    % length_car: car length
    
    a = u(1);  % acceleration
    steeringAngle = u(2);  % steering angle
    
    x_new = x0(1) + x0(4) * cos(x0(3)) * dt;
    y_new = x0(2) + x0(4) * sin(x0(3)) * dt;
    psi_new = x0(3) + x0(4) / length_car * tan(steeringAngle) * dt;
    v_new = x0(4) + a * dt;
    x1 = [x_new; y_new; psi_new; v_new];
end