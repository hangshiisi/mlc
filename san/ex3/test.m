
data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = lrCostFunction(initial_theta, X, y, 0.01);



%>> cost
%cost =  0.69315
%>> grad
%grad =
%
%   -0.10000
%  -12.00922
%  -11.26284
  
%===================================================
%===================================================
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

