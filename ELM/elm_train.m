

function [TrAccuracy,TsAccuracy]=elm_train(X,Y,L,M,number_neurons,ELM_Type)
% elm_train: this function allows to train a single hidden layer
% feedforward network  for regression with Moore-Penrose pseudoinverse of matrix.
% Inputs:- number_neurons: number of neurons in the hidden layer
%        - X:  N instances by Q atrebutes matrix of  training inputs;
%        - Y:  N raws and S atrebutes matrix of training targets
%        - L:  testing inputs
%        - M:  testing targets
%        - ELM_Type: 0 for regression 1 for cllassification
% outputs:- prefomance: RMSE of regression

    %%%%    Author:        TAREK BERGHOUT
    %%%%    BATNA 2 TECHNOLOGICAL UNIVERSITY, ALGERIA
    %%%%    EMAIL:          berghouttarek@gmail.com
    %%%%    WEB PROFILE:    https://www.mathworks.com/matlabcentral/profile/authors/6602421-berghout-tarek


%%%% 1st step: generate a random input weights
input_weights=rand(number_neurons,size(X,2))*2-1;
%%%% 2nd step: calculate the hidden layer
H=radbas(input_weights*X');
%%%% 3rd step: calculate the output weights beta
B=pinv(H') * Y ; %Moore-Penrose pseudoinverse of matrix
%%%% calculate the actual output of traning ans testing 
output=(H' * B)' ;
output_ts=(radbas(input_weights*L')'*B)';
%%%% calculate the prefomance of training and testing
if ELM_Type==0
TrAccuracy=sqrt(mse(Y'-output));% RMSE for regression
TsAccuracy=sqrt(mse(M'-output_ts));% RMSE for regression
else
     MissClassificationRate_Training=0;
     MissClassificationRate_Testing=0;
     T=Y';
     TV.T=M';
    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(output(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrAccuracy=1-MissClassificationRate_Training/size(T,2);% classification rate   
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(output_ts(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end   
    TsAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);% classification rate   
 
end
end