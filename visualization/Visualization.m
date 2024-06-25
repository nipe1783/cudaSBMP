clc;
clear all;

% Opening Control Data:
wslFilePath = '\\wsl.localhost\Ubuntu-20.04\home\nic\dev\research\cudaSBMP\build\controls.csv';
controls = readmatrix(wslFilePath);  % Read CSV as matrix of doubles

% Opening Samples Data:
wslFilePath = '\\wsl.localhost\Ubuntu-20.04\home\nic\dev\research\cudaSBMP\build\samples.csv';
samples = readmatrix(wslFilePath);  % Read CSV as matrix of doubles

% Insert a row of zeros at the top of the samples matrix
samples = [zeros(1, size(samples, 2)); samples];

% Initialize Variables:
sample_dim = 7;
state_dim = 4;
input_dim = 3;
threadsPerBlock = 32;
blocksPerGrid = 32;
rowsTree = 10;
granularity = 20;
x_initial = [0; 0; 0; 0];
L = 1;  % Car length

% Number of time steps
num_steps = rowsTree * granularity;
% Initialize array to store results
x_values = zeros(num_steps, state_dim * (threadsPerBlock * blocksPerGrid));

% Create a figure for dynamic plotting
figure;
hold on;
grid on;
title('Dynamic Car Path');
xlabel('X Position');
ylabel('Y Position');
plot(x_initial(1), x_initial(2), 'ko', 'MarkerFaceColor', 'k');

% Generate unique RGB colors for each row using HSV space
colors = hsv(rowsTree);

for row = 1:rowsTree
    color = colors(row, :);
    for block = 1:blocksPerGrid
        for thread = 1:threadsPerBlock
            segment_x = [];
            segment_y = [];
            ctrlRow = row + 1;
            ctrlCol = ((block-1)*threadsPerBlock + thread) * sample_dim;
            dt = samples(ctrlRow, ctrlCol);
            u = samples(ctrlRow, (ctrlCol - input_dim + 1):(ctrlCol - 1));
            col = ((block-1)*threadsPerBlock) * sample_dim + 1;
            x0 = samples(row,col:(col+state_dim-1));
            stepSize = dt / granularity;
            for step = 1:granularity
                x1 = dynamics_function(x0, u, stepSize, L);
                segment_x = [segment_x, x1(1)];
                segment_y = [segment_y, x1(2)];
                x0 = x1;
            end
            plot(segment_x, segment_y, '-.', 'Color', color);
            plot(x1(1), x1(2), 'ko', 'MarkerFaceColor', color);
        end
    end
    drawnow;
    pause(.01);
end

% Car dynamics function
function x_prime = dynamics_function(x, u, dt, length_car)
    % x: state vector [x; y; psi; v]
    % u: control input vector [a; steeringAngle]
    % dt: time step
    % length_car: car length
    
    a = u(1);  % acceleration
    steeringAngle = u(2);  % steering angle
    
    x_new = x(1) + x(4) * cos(x(3)) * dt;
    y_new = x(2) + x(4) * sin(x(3)) * dt;
    psi_new = x(3) + x(4) / length_car * tan(steeringAngle) * dt;
    v_new = x(4) + a * dt;
    
    x_prime = [x_new; y_new; psi_new; v_new];
end
