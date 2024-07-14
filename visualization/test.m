clc
clear all

% Parameters
numFiles = 1;
width = 20.0;
height = 20.0;
N = 16;
n = 8;
R1_width = width / N;
R1_height = height / N;
R2_width = R1_width / n;
R2_height = R1_height / n;
numDisc = 10;
sampleSize = 7;
stateSize = 4;
controlSize = 3;
agentLength = 1;  % Car length
xGoal = [10, 10];
alphaValue = 0.3;

% File paths
sampleFilePath = '/home/nicolas/dev/research/KGMT/build/samples.csv';
parentFilePath = '/home/nicolas/dev/research/KGMT/build/parentRelations.csv';
obstacleFilePath = '/home/nicolas/dev/research/KGMT/include/config/obstacles/obstacles.csv';


% Read obstacle data
obstacles = readmatrix(obstacleFilePath);

for i = 1:numFiles

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
    % for j = 0:N*n
    %     x = j * R2_width;
    %     patch([x x x x], [0 height height 0], 'k', 'EdgeColor', 'k', 'FaceColor', 'none', 'FaceAlpha', alphaValue, 'EdgeAlpha', alphaValue);
    % end
    % for j = 0:N*n
    %     y = j * R2_height;
    %     patch([0 width width 0], [y y y y], 'k', 'EdgeColor', 'k', 'FaceColor', 'none', 'FaceAlpha', alphaValue, 'EdgeAlpha', alphaValue);
    % end

    % Plot goal position
    plot(xGoal(1), xGoal(2), 'ko', 'MarkerFaceColor', 'g')

    for j = 1:size(obstacles, 1)
        x0 = obstacles(j, 1);
        y0 = obstacles(j, 2);
        x1 = obstacles(j, 3);
        y1 = obstacles(j, 4);
        patch([x0 x1 x1 x0], [y0 y0 y1 y1], 'r', 'EdgeColor', 'r', 'FaceColor', 'r', 'FaceAlpha', 1.0);
    end

    % Plot paths
    for j = 2:size(parentRelations, 1)
        if parentRelations(j) == -1
            break;
        end
        x0 = samples((parentRelations(j) + 1), 1:stateSize);
        segmentX = [x0(1)];
        segmentY = [x0(2)];
        u = samples(j, stateSize+1:sampleSize-1);
        duration = samples(j, sampleSize);
        dt = duration / numDisc;
        x = x0(1);
        y = x0(2);
        theta = x0(3);
        v = x0(4);
        for k = 1:(numDisc)
            cos_theta = cos(theta);
            sin_theta = sin(theta);
            tan_steering = tan(u(2));
            x = x + v * cos_theta * dt;
            y = y + v * sin_theta * dt;
            theta = theta + (v / agentLength) * tan_steering * dt;
            v = v + u(1) * dt;
            segmentX = [segmentX, x];
            segmentY = [segmentY, y];
            % disp(['Iteration: ', num2str(k)]);
            % disp(['cos_theta: ', num2str(cos_theta)]);
            % disp(['sin_theta: ', num2str(sin_theta)]);
            % disp(['tan_steering: ', num2str(tan_steering)]);
            % disp(['x: ', num2str(x)]);
            % disp(['y: ', num2str(y)]);
            % disp(['theta: ', num2str(theta)]);
            % disp(['v: ', num2str(v)]);
            % disp(' ');
        end

        plot(segmentX, segmentY, '-.', 'Color', 'k', LineWidth=.01);
        plot(samples(j, 1), samples(j, 2), 'bo', 'MarkerFaceColor', 'b', MarkerSize=2);
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
