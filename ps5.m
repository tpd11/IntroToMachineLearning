clc
clear
load('hw4_data3.mat');
X_test = [ones(25,1), X_test];
X_train = [ones(125,1), X_train];
sigma = [.01 .05 .2 1.5 3.2 5];
y_predict = zeros(size(X_test,1),size(sigma,2));
for i = 1:size(sigma,2)
    y_predict(:,i) = weightedKNN(X_train, y_train, X_test, sigma(i));
end

for j = 1:size(sigma,2) %number of sigmas
    count = 0;
    for i = 1:size(X_test) %numbver of test samples
        if(y_predict(i,j) == y_test(i))
            count = count + 1;
        end
    end
    accuracy(j) = count/size(y_test,1);
end