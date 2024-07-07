close all
clc
clear all

% Parameters
numFiles = 10;
width = 20.0;
height = 20.0;
N = 16;
n = 8;
R1_width = width / N;
R1_height = height / N;
R2_width = R1_width / n;
R2_height = R1_height / n;
sampleSize = 7;
stateSize = 4;
alphaValue = 0.1;
xGoal = [10, 10];

% Obstacle file path
obstacleFilePath = '\\wsl.localhost\Ubuntu-20.04\home\nic\dev\research\cudaSBMP\configurations\obstacles\obstacles.csv';

% Read obstacle data
obstacles = readmatrix(obstacleFilePath);

for i = 1:numFiles
    % Construct file pathsgit 
    sampleFilePath = "\\wsl.localhost\Ubuntu-20.04\home\nic\dev\research\cudaSBMP\build\Data\Samples\samples" + i + ".csv";
    parentFilePath = "\\wsl.localhost\Ubuntu-20.04\home\nic\dev\research\cudaSBMP\build\Data\Parents\parents" + i + ".csv";

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

    % Plot obstacles
    for j = 1:size(obstacles, 1)
        x0 = obstacles(j, 1);
        y0 = obstacles(j, 2);
        x1 = obstacles(j, 3);
        y1 = obstacles(j, 4);
        patch([x0 x1 x1 x0], [y0 y0 y1 y1], 'r', 'EdgeColor', 'r', 'FaceColor', 'r', 'FaceAlpha', 1.0);
    end

    % Plot goal position
    plot(xGoal(1), xGoal(2), 'ko', 'MarkerFaceColor', 'g');

    % Plot samples
    plot(samples(:, 1), samples(:, 2), 'bo', 'MarkerFaceColor', 'b', MarkerSize=.4);


    drawnow;
end
