%% Project Part 2 - Peter_Vilau_Tivadar
% Load the data
clc
clear
close all
load("iddata-08.mat");

% Input and output data
yid = id.y;
uid = id.u;
yval = val.y;
uval = val.u;

N = length(yid); % Length of yid and yval

% Choose your desired parameters
disp('Choose parameters: ');
na = input('Choose na (output lags): ');
nb = input('Choose nb (input lags): ');
m = input('Choose m (degree of polynomial): ');

% Creating regressor and regressor matrix with 
% polynomial expansion for identification
d_id = construct_regressor(yid, uid, na, nb);
matrix_id = generate_polynomial_terms(d_id, m); 

% Creating regressor and regressor matrix with 
% polynomial expansion for validation
d_val = construct_regressor(yval, uval, na, nb);
matrix_val = generate_polynomial_terms(d_val, m); 

% Calculating unknown parameters
theta = matrix_id \ yid;

%% One-Step-Ahead (OSA) Prediction Identification

yhat_id = matrix_id * theta;

% Calculate and display OSA MSE
mse_osa = mean((yid - yhat_id).^2);
disp(['OSA MSE (Identification): ', num2str(mse_osa)]);

yhat_iddata = iddata(yhat_id, id.u, id.Ts); 
figure
compare(id, yhat_iddata);
title('Compare function on Prediction (Identification)');
legend('Identification', 'na=1, nb=5, m=8');

%% One-Step-Ahead (OSA) Prediction Validation

yhat_val = matrix_val * theta;

% Calculate and display OSA MSE
mse_osa = mean((yval - yhat_val).^2);
disp(['OSA MSE (Validation): ', num2str(mse_osa)]);


yhat_iddata_val = iddata(yhat_val, val.u, val.Ts); 
figure
compare(val, yhat_iddata_val);
title('Compare function on Prediction (Validation)');
legend('Validation', 'na=1, nb=5, m=8');

%% Simulation Identification

ysim_id = zeros(N, 1);
ysim_id(1:max(na, nb)) = yid(1:max(na, nb));

for i = max(na, nb) + 1:N
    ypast = -ysim_id(i - (1:na)); % Negative lagged outputs
    upast = uid(i - (1:nb));  % Lagged inputs
    phi_row = [ypast', upast'];
    
    phi_sim = generate_polynomial_terms(phi_row, m);
    ysim_id(i) = phi_sim * theta;
end

% Calculate and display simulation MSE
mse_sim = mean((yid - ysim_id).^2);
disp(['Simulation MSE (Identification): ', num2str(mse_sim)]);

ysim_id_iddata = iddata(ysim_id, id.u, id.Ts); 
figure
compare(id, ysim_id_iddata);
title('Compare function on Simulation (Identification)');
legend('Identification', 'na=1, nb=5, m=8');

%% Simulation Validation

ysim = zeros(N, 1);
ysim(1:max(na, nb)) = yval(1:max(na, nb));

for i = max(na, nb) + 1:N
    ypast = -ysim(i - (1:na)); % Negative lagged outputs
    upast = uval(i - (1:nb));  % Lagged inputs
    phi_row = [ypast', upast'];

    phi_sim = generate_polynomial_terms(phi_row, m); % Numeric array
    ysim(i) = phi_sim * theta;
end

% Calculate and display simulation MSE
mse_sim = mean((yval - ysim).^2);
disp(['Simulation MSE (Validation): ', num2str(mse_sim)]);

ysim_iddata = iddata(ysim, val.u, val.Ts); 
figure
compare(val, ysim_iddata);
legend('Validation', 'na=1, nb=5, m=8');
title('Compare function on Simulation (Validation)');

%% General Functions

% This function generate_polynomial_terms  creates a matrix phi that contains polynomial 
% terms up to degree m based on the input data d. 
    % Input:
    %   d - Input matrix (n x num_vars), where each column represents a variable
    %       and each row represents a time step.
    %   m - Maximum degree of the polynomial terms to generate.
    %
    % Output:
    %   phi - Matrix containing polynomial terms. Each column represents
    %         a unique polynomial term, and each row corresponds to a time step.
function phi = generate_polynomial_terms(d, m)
    % Number of time steps (rows)
    n = size(d, 1); 
    % Number of input/output variables (na + nb)(columns)
    num_vars = size(d, 2);  
    
    % Generates all combinations of powers for num_vars variables up to degree m.
    combinations = generate_power_combinations(num_vars, m);
    % Initialize phi matrix
    phi = ones(n, size(combinations, 1)); 

    % Generate polynomial terms for each combination of powers
    for i = 1:size(combinations, 1)
        term = ones(n, 1);
        for j = 1:num_vars
            % Elementwise multiplication for each power combination
            term = term .* d(:, j).^combinations(i, j); 
        end
        % Store the generated term
        phi(:, i) = term;  
    end
end


% This function generate_power_combinations enerates all combinations of powers for the polynomial terms
    % Input:
    %   num_vars - The number of variables involved (e.g., lagged inputs/outputs).
    %   degree - The maximum total degree of the polynomial terms.
    %
    % Output:
    %   power_combinations - A matrix where each row represents a combination
    %                        of powers for the variables. The number of rows
    %                        corresponds to the number of polynomial terms.

function power_combinations = generate_power_combinations(num_vars, degree)
    % Initialize an empty matrix to store power combinations
    power_combinations = [];
    % Loop over total degrees from 0 up to the specified maximum degree
    for total_degree = 0:degree
        % Generate all combinations of powers for the given number of variables
        % that sum up to 'total_degree' using the function "generate_combinations"
        combinations = generate_combinations(num_vars, total_degree);
        power_combinations = [power_combinations; combinations];
    end
end

% This function "generate_combinations" generate all combinations
% of powers for N variables that sum to total_degree using recursion
    % Input:
    %   N - Number of variables
    %   total_degree - The total degree to which the powers should sum
    %
    % Output:
    %   combinations - A matrix where each row represents a unique combination of
    %                  powers for N variables, such that the sum of each row equals
    %                  total_degree.
function combinations = generate_combinations(N, total_degree)
    % Generate power combinations for N variables
    if N == 1
        combinations = total_degree;
    else
        combinations = [];
        for k = 0:total_degree
            % Recursive call to generate combinations for the remaining (N - 1) variables
            % with a reduced total degree (total_degree - k)
            sub_combinations = generate_combinations(N - 1, total_degree - k);
            % Create a new matrix where each row starts with the current power (k)
            % for the first variable, followed by the sub_combinations for the remaining variables.
            combinations = [combinations; [k * ones(size(sub_combinations, 1), 1), sub_combinations]];
        end
    end
end


% This function "construct_regressor" constructs a regression matrix (phi)
% for time series data based on the given input and output variables,
% where phi is used in autoregressive and moving average modeling.
% Input:
%   y   - The output or dependent variable time series (length N)
%   u   - The input or independent variable time series (length N)
%   na  - The number of past observations of y (AR part)
%   nb  - The number of past observations of u (MA part)
%
% Output:
%   phi - A matrix where each row contains the lagged values of y and u
%         for the corresponding time point, used in model estimation.
function phi = construct_regressor(y, u, na, nb)
    % Number of data points
    N = length(y);
    % Initialize the regressor matrix phi with zeros, size N x (na + nb)
    phi = zeros(N, na + nb);
    for i = 1:N 
        % Construction: the first na columns represent lagged values of y
        for j = 1:na 
            if i - j > 0 
                phi(i, j) = -y(i - j); 
            else 
                phi(i, j) = 0; 
            end
        end
        % Construction: the next nb columns represent lagged values of u
        for j = 1:nb 
            if i - j  > 0 
                phi(i, j+na) = u(i - j); 
            else 
                phi(i, j+na) = 0;
            end 
        end 
    end 
end

%% Plots for presentation

% compare(val, ysim1_iddata, ysim2_iddata, ysim3_iddata);
% title('Compare function on Simulation')
% legend('Validation', 'na=3, nb=1, m=2', 'na=4, nb=1, m=2', 'na=6, nb=1, m=5' )

% compare(val, ysim4_iddata, ysim5_iddata, ysim6_iddata);
% title('Compare function on Simulation')
% legend('Validation', 'na=2, nb=2, m=2', 'na=1, nb=1, m=1', 'na=3, nb=3, m=3' )

% compare(val, ysim7_iddata, ysim8_iddata, ysim9_iddata);
% title('Compare function on Simulation')
% legend('Validation', 'na=4, nb=5, m=1', 'na=2, nb=6, m=1', 'na=8, nb=5, m=1' )
