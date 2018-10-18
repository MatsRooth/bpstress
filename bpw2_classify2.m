function R = bpw2_classify2(matfile)
% Use a single weight feature
       
       
% Initialize the result.
R = {};
% The initial part of this is like bpw2_stat1.
if nargin < 1
    %matfile = '/local/matlab/Kaldi-alignments-matlab/data-bpn/tab4-sample.mat'; % Made with token_data_bpw2.
    matfile = '/local/matlab/Kaldi-alignments-matlab/data-bpn/tab4.mat'; % All the data, 15388 bisyllables
end

% Load sets L to a structure. It has to be initialized first.
L = 0;
load(matfile);

% Scale for combining the two weights.
acoustic_scale = 0.083333;
% Then combine by this formula, see
% /projects/speech/sys/kaldi-master/egs/bp_ldcWestPoint/bpw2/exp/u1/decode_word_1/tab-min.awk
% weight = weight1 +  acoustic_scale * weight2;

% Duration in frames
D = cellfun(@sum,L.phonedur)';

% Combined weights
W1 = cellfun(@(x,y) x + acoustic_scale * y,L.weight1,L.weight2,'UniformOutput',false)';

% Combined weights scaled down by duration.
% This produces weights in the range 7.0 to 9.5.
W2 = cellfun(@(x,y) x ./ y,W1,num2cell(D),'UniformOutput',false);

% Logical indices of ultimate-stressed bisyllables,
% and penultimate-stressed bisyllables.
U21 = L.syl == 2 & L.cstress == 1;
U22 = L.syl == 2 & L.cstress == 2;

% Logical indices of all bisyllables
% There are 15388 in the complete dataset.
U2 = L.syl == 2;

% Indices that are 1 in U2, for mapping back to L.
I2 = find(U2);

% Corresponding matrices of weights 
U21w = cell2mat(W2(U21));
U22w = cell2mat(W2(U22));
U2w = cell2mat(W2(U2));
% U2w has dimensions 402 x 2, with entries in the range 7 to 9.

% Signed distance to equal-weight diagonal

V2w = (U2w(:,2) - U2w(:,1)) ./ sqrt(2);


%%%%%%%% Duration %%%%%%%%

% Matrices of vowel duration
U21d = cell2mat(L.voweldur(U21)');
U22d = cell2mat(L.voweldur(U22)');
U2d = cell2mat(L.voweldur(U2)');
% U2d has dimensions 402 x 2, with entries (in centiseconds) in the range
% 3 to circa 20

% Feature matrix. Row indices are items, columns are ..
X = [V2w,U2d];


% The class vector is a boolean vector, with 1 indicating final stress (21)
% and 0 indicating initial stress (22). There are two classes.

Y = U21(U2);

% Reduce the number of data points
%X = X(1:1000,:);
%Y = Y(1:1000);

% Find the loss from comparing weights directly
% This compares the prediction from the weights to Y.
% There is one error
% [X(1:30,:),Y(1:30),(X(1:30,2) > X(1:30,1))]
% Indices where they disagree.
% Y(1:30) ~= (X(1:30,2) > X(1:30,1))
% Ratio
% R.wLoss = nnz(Y ~= (X(:,2) >= X(:,1))) / size(Y,1)

% Fit svm using 3 columns
R.svm = fitcsvm(X,Y,'Standardize',true,'KernelScale','auto','KernelFunction','linear');
 
% Crossvalidate
R.csvm = crossval(R.svm);
 

% Loss
R.L = kfoldLoss(R.csvm);
% linear 0.914
% rbf 

% Save the data
R.X = X;
R.Y = Y;

disp 1
% Results for 3 columns in X
% linear  
% Results for w_linear, d_rbf, all_linear
% Need also all_rbf
%     0.1080 0.2224 0.0916

% Predicted labels for weight only
%R.Lw = predict(R.svm1w_linear,X);

% Mislabeled items
%R.M1 = ~(R.Lw == Y);
% Indices in L of mislabeled items
%R.M1l = I2(R.M1);

% IDs of mislabeled items
%R.M1id = L.wid(M1l);

 
% Parse a line into a key and a vector of int.
function [key,a] = parse_alignment(line)
    key = sscanf(line,'%s',1);
    [~,klen] =  size(key);
    [~,llen] = size(line);
    line = line((klen+1):llen);
    a = sscanf(line,'%d')';
end

% Parse a line from the table.
% The input line looks like this.
% f58br08b11k1-s087-2	abacaxi	abacaxi_U411	4	1	1	4.45933 4.46457 4.43014 4.40614	5115.16 5122.39 5166.43 5153.47	362_364_3
% uid                   wf1     wf2             syl cit dec [w1] [w2]
%   bns04_st1921_trn 1 12 ; 6 7 ; 143 3 ; 50 8 ; 60 3 ; 143 4 ; 146 13
function [uid,word_form1,word_form2,syl_count,citation_stress,decode_stress,weight1,weight2] = parse_line(line)
    part = strsplit(line,'\t');
    uid = part{1};
    word_form1 = part{2};
    word_form2 = part{3};
    syl_count = str2num(part{4});
    citation_stress = str2num(part{5});
    decode_stress = str2num(part{6});
    weight1 = str2num(part{7});
    weight2 = str2num(part{8});
end

% Result of 'OptimizeHyperparameters','all'
% Best estimated feasible point (according to models):
%    BoxConstraint    KernelScale    KernelFunction    PolynomialOrder    Standardize
%    _____________    ___________    ______________    _______________    ___________
%
%       6.9424            NaN          polynomial             2              true    
%
%Estimated objective function value = 0.082355
%Estimated function evaluation time = 0.48633

% 0.0846 weight
% 0.3035 duration
% 0.0871 both--it's a bit worse

end

