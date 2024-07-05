close all
clc
clear all

% Parameters
numFiles = 1;
width = 10.0;
height = 10.0;
N = 16;
n = 16;
R1_width = width / N;
R1_height = height / N;
R2_width = R1_width / n;
R2_height = R1_height / n;
numDisc = 10;
sampleSize = 7;
stateSize = 4;
controlSize = 3;
L = 1;  % Car length
xGoal = [10,10];
alphaValue = 0.1;

for i = 1:numFiles
    % Construct file paths
    sampleFilePath ='\\wsl.localhost\Ubuntu-20.04\home\nic\dev\research\cudaSBMP\build\samples.csv';
    parentFilePath = '\\wsl.localhost\Ubuntu-20.04\home\nic\dev\research\cudaSBMP\build\parentRelations.csv';

    % Read data from files
    samples = readmatrix(sampleFilePath);
    parentRelations = readmatrix(parentFilePath);

    % Create a new figure for each file
    figure;
    hold on;
    axis equal;
    title(sprintf('Dynamic Car Path - Iteration %d', i));
    xlabel('X Position');
    ylabel('Y Position');
    plot(samples(1,1),samples(1,2), 'ko', 'MarkerFaceColor', 'k');

    % Plot R1 cells with transparency
    for j = 0:N
        x = j * R1_width;
        patch([x x x x], [0 height height 0], 'k', 'EdgeColor', 'k', 'FaceColor', 'none', 'FaceAlpha', alphaValue, 'EdgeAlpha', alphaValue);
    end
    for j = 0:N
        y = j * R1_height;
        patch([0 width width 0], [y y y y], 'k', 'EdgeColor', 'k', 'FaceColor', 'none', 'FaceAlpha', alphaValue, 'EdgeAlpha', alphaValue);
    end

    % Plot R2 cells within each R1 cell with transparency
    for j = 0:N*n
        x = j * R2_width;
        patch([x x x x], [0 height height 0], 'k', 'EdgeColor', 'k', 'FaceColor', 'none', 'FaceAlpha', alphaValue, 'EdgeAlpha', alphaValue);
    end
    for j = 0:N*n
        y = j * R2_height;
        patch([0 width width 0], [y y y y], 'k', 'EdgeColor', 'k', 'FaceColor', 'none', 'FaceAlpha', alphaValue, 'EdgeAlpha', alphaValue);
    end
    plot(xGoal(1), xGoal(2), 'ko', 'MarkerFaceColor', 'g')

    % Plot paths
    for j = 2:size(parentRelations, 1)
        if parentRelations(j) == -1
            break;
        end
        x0 = samples((parentRelations(j) + 1), 1:stateSize);
        segmentX = [x0(1)];
        segmentY = [x0(2)];
        u = samples(j, stateSize+1:sampleSize-1);
        dt = samples(j, sampleSize);
        stepSize = dt / numDisc;
        for step = 1:numDisc
            x1 = dynamicsPropagator(x0, u, stepSize, L);
            segmentX = [segmentX, x1(1)];
            segmentY = [segmentY, x1(2)];
            x0 = x1;
        end
        plot(segmentX, segmentY, '-.', 'Color', 'k');
    end
    drawnow;
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
