function [y_predict] = weightedKNN(X_train, y_train, X_test, sigma)
%WEIGHTEDKNN This function calculates a predicted y value given some test
%data and training data, along with a value for sigma.

y_predict = zeros(size(X_test,1),1); %column vector

for i = 1:size(X_test,1)    %For every test point
    c_sum = zeros(1,max(y_train));
    
    dist = pdist2(X_train, X_test(i,:),  'euclidean');
    w = exp(-(dist.^2)/(sigma^2));
    for k = 1:size(X_train,1)    %Run # of samples
        for j = 1:max(y_train)  %Run # of classes iterations
            if(y_train(k) == j)
                c_sum(j) = c_sum(j) + w(k); %Sums all the weights for each training sample fpr the appropriate class
                
            end
        end
    end
    [~,y_predict(i)] = max(c_sum);
   
end
return
end
