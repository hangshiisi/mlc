function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

for i = 1:m
  J = J + (-y(i) * log(sigmoid(X(i,:)* theta )) - (1.0 - y(i)) * 
            log(1.0-sigmoid(X(i,:) *theta))); 
endfor 

J = J / m; 



%>> X(10,:)
%ans =
%
%    1.0000   84.4328   43.5334
%
%>> initial_theta
%initial_theta =
%
%   0
%   0
%   0
%>> kk2 = (sigmoid(initial_theta * X(10))) .* transpose(X(10, :));
%>> kk2
%kk2 =
%
%    0.50000
%   42.21641
%   21.76670
   
   
for i = 1:m     
    %for j = 1: length(theta)
    %    grad(j) = grad(j) + (sigmoid(theta*X(i)) - y(i)) * X(i)
    %endfor
    %grad = grad + (sigmoid(theta*X(i)) - y(i)) * [ X(i,:) ];
    %kk1 = (sigmoid(X(i) * theta) - y(i)) *  transpose(X(i,:));
    grad = grad + (sigmoid(X(i,:) * theta) - y(i)) *  transpose(X(i,:));

endfor 

    
grad = grad / m; 






% =============================================================

end
