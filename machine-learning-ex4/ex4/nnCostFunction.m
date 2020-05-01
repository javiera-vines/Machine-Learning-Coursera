function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Part 1:
% 1.a Expand the 'y' output values into a matrix of single values.
y_matrix = eye(num_labels)(y,:);

% 1.b Feed Forward prop
%a1: 5000x401 a2: 5000x26 a3: 5000x10
a1= [ones(m,1) X];
z2= a1*Theta1';
a2= sigmoid(z2);
a2= [ones(m,1) a2];

z3= a2*Theta2';
a3= sigmoid(z3); %h(x)
%now 'y' and 'h' are both matrices of size (m x K)

% 1.c Cost Function
%if 'y' and 'h' are vectors is easy to set J as a result of (1 x m) * (m x 1)
J= -( y_matrix'*log(a3) + (1-y_matrix)'*log(1-a3))/m; %here we get A'*B matrix
J= trace(J); %to sum the diagonal of the matrix and obtain J as a scalar

%1.d Regularized Cost Fuction
sum_sq_Theta1=0;
sum_sq_Theta2=0;
%input_layer_size  = 400;  % 20x20 Input Images of Digits
%hidden_layer_size = 25;   % 25 hidden units
%num_labels = 10;   


%sum_sq_Theta1
for j=1:hidden_layer_size,
  for k=2:input_layer_size+1, %we don't use the first column because is bias value of theta
    sum_sq_Theta1=sum_sq_Theta1+Theta1(j,k)^2;
  end;                           
end;
%sum_sq_Theta2
for j=1:num_labels,
  for k=2:hidden_layer_size+1, %we don't use the first column because is bias value of theta
    sum_sq_Theta2=sum_sq_Theta2+Theta2(j,k)^2;
  end;                           
end;
%regularized term
reg= (lambda/(2*m))*(sum_sq_Theta1+sum_sq_Theta2);
%Cost Function Regularized
J= J+reg;

%Part2:
%Backpropagation to compute the gradients Theta1_grad and Theta2_grad
d3=a3-y_matrix; %(5000x10)-(5000x10) &&same a3 size  d3: 5000x10
d2=(d3*Theta2(:,2:end)) .* sigmoidGradient(z2); %%same a2 size %d2: 5000x25 %%  ignore bias!!


%Theta1, Delta1 and Theta1_grad: 25x401
%Theta2, Delta2 and Theta2_grad: 10x26
Delta2=d3'*(a2); %10x26
Delta1=d2'*(a1); %25x401

%Delta2_grd= (1/m)*(Delta2+lambda*Theta2);
%Delta1_grd= (1/m)*(Delta1);

Theta1_grad = Theta1_grad + (1/m) * Delta1;
Theta2_grad = Theta2_grad + (1/m) * Delta2;


% REGULARIZATION OF THE GRADIENT %% from j>=1
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*(Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*(Theta2(:,2:end));



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
