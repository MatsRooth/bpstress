function [LD_ratio_correct,SVM_ratio_correct,SVM_RBF_ratio_correct] = tid_classify3(data_train,data_test,features)

% August, 2018: adjust it to work with MATLAB 2018a and fitcsvm.

% For instance:
% web1 and web2 datasets
% data1 = thanIdid1_mean_impute();
% data2 = thanIdid2_mean_impute();
% F = {'duration_V2', 'f1f2Time50_V2', 'maxf0_ratio','duration_C3'}
% tid_classify2(data1,data2,F)
    
% data has this structure
%            C: 310
%          col: {1x309 cell}
%          dat: {394x309 cell}
%            N: 394
%        index: [1x394 double]
%    soundname: {1x394 cell} 
%        focus: {1x394 cell}   's' or 'ns'


% Defaults for testing and publish.
if nargin < 1
    data_train = thanIdid1_mean_impute();
    data_test = thanIdid2_mean_impute();
    % Features Experimenter-selected B, see line 12 of Table 3.
    features = {'duration_V2', 'f1f2Time50_V2', 'maxf0_ratio','duration_C3'}
    % features = {'duration_V2', 'f1f2Time50_V2','duration_C3'}
end

% Data matrix for features F.
% data.dat, which is converted from R, has strings
% instead of numbers to defer conversion/rounding issues.
% Indices corresponding to F are the second value.
function [D,I] = data_matrix(data,F)
    % Column indices corresponding to featues F.
    I = find(ismember(data.col,F));
    % Matrix of values for the designated columns, converted to double.
    D = str2double(data.dat(:,I));
end

    function r = ratio_correct(y_predict,y_true)
        % Boolean vector with 1 where the classes agree and zero where
        % they disagree.
        boolean_correctness = strcmp(y_predict,y_true);
        % Ratio of number on non-zero entries to length of vector.
        r = nnz(boolean_correctness) / max(size(boolean_correctness));
    end
        
        % Training data. 
        x = data_matrix(data_train,features);
        % Training classification values.
        y = data_train.focus';
        
        % Fit LDA model.
        ldacls = fitcdiscr(x,y);
        % Fit linear SVM. It was worse without standardization.
        svmcls = fitcsvm(x,y,'Standardize',true);
        % Fit radial SVM.
        % See https://www.mathworks.com/help/stats/fitcsvm.html for
        % parameters. It was much worse without standardization.
        svm_rbf_cls = fitcsvm(x,y,'KernelFunction','rbf','Standardize',true);
        
        % Test data
        x_te = data_matrix(data_test,features);
        % Correct classes, 's' or 'ns'.
        y_te = data_test.focus';
        
        % Labels for data_test that are predicted by LD model.
        y_ld = predict(ldacls,x_te);
        % Ratio correct for LD model
        LD_ratio_correct = ratio_correct(y_ld,y_te)
        
        % Classes for data_test that are predicted by linear SVM.
        y_svm = predict(svmcls,x_te);
        SVM_ratio_correct = ratio_correct(y_svm,y_te)
        
        % Classes for data_test that are predicted by rbf SVM.
        y_svm_rbf = predict(svm_rbf_cls,x_te);
        SVM_RBF_ratio_correct = ratio_correct(y_svm_rbf,y_te)

end
 
 