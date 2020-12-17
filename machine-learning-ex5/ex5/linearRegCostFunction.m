function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta);
% You need to return the following variables correctly 
J = 0;
##grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%X=[ones(m,1) X];
%for i=1:m
H=X*theta;
%endfor
J=(1/(2*m))*sum((H-y).^2)+(lambda/(2*m))*(theta(2:end,:)'*theta(2:end,:));
 G=(H-y)'*X;
for j=1:n
  if (j==1)
      grad(1,j)=(1/m)*G(1,j);
  else 
    grad(1,j)=(1/m)*G(1,j)+(lambda/m)*theta(j,1);
  endif
 
endfor










% =========================================================================

grad = grad(:);

end
