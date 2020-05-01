function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

g = sigmoid(X*theta);
n = length(theta);

%cost fuction
J = (-log(g)' * (y == 1)  - log(1 - g)' * (y == 0)) / m;  %glab

%regularization term
sum_theta_sqr=0;

for j =1:n,
  if j ==1,
    sum_theta_sqr=0;
  else,
    theta_sqr =(theta(j))^2;
    sum_theta_sqr = sum_theta_sqr + theta_sqr;
  end;
end;


reg = (lambda / (2*m)) * sum_theta_sqr;
J = J + reg;

%gradient descent
grad = (g - y)' * X / m;  %glab

for j = 2:n,
  grad(j) = grad(j) + lambda / (m) * theta(j);
end;

% =============================================================

end
