function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
s_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

err_val = 0;
initial_err = 100;

for C_value = C_list,
  for s_value = s_list,
    % your code goes here to train using C_val and sigma_val
    %and compute the validation set errors 'err_val'
    model= svmTrain(X, y, C_value, @(x1, x2) gaussianKernel(x1, x2, s_value)); 
    predictions = svmPredict(model, Xval);
    err_val = mean(double(predictions ~= yval));
    
    if err_val < initial_err,
      initial_err = err_val;
      C = C_value;
      sigma = s_value;
      %fprintf('New Min found!! C, sigma = %f, %f with error = %f', C, sigma, err_val);
    end
  end
end



% =========================================================================

end
