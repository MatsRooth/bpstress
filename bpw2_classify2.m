function R = bpw2_classify2(matfile)
% Use a single weight feature
       
% Initialize the result.
R = {};
% The initial part of this is like bpw2_stat1.
if nargin < 1
    %matfile = '/local/matlab/Kaldi-alignments-matlab/data-bpn/tab4-sample.mat'; % Made with token_data_bpw2.
    matfile = '/local/matlab/Kaldi-alignments-matlab/data-bpn/tab4.mat'; % All the data, 15388 bisyllables
end

% This defines L as a structure.
load(matfile,'L');

% L has this shape.
%         wid: [40007×1 string]
%        word: [40007×1 string]
%       wordu: [40007×1 string]
%         syl: [40007×1 double]
%     cstress: [40007×1 double]
%     astress: [40007×1 double]
%     weight1: {1×40007 cell}
%     weight2: {1×40007 cell}
%       align: {1×40007 cell}
%    voweldur: {1×40007 cell}
%    phonedur: {1×40007 cell}
%       spell: {1×40007 cell}

% Scale for combining the two weights.
acoustic_scale = 0.083333;
% Then combine by this formula, see
% /projects/speech/sys/kaldi-master/egs/bp_ldcWestPoint/bpw2/exp/u1/decode_word_1/tab-min.awk
% weight = weight1 +  acoustic_scale * weight2;

% Duration of the word in frames.  This is used for scaling weights.
D = cellfun(@sum,L.phonedur)';

% Logical indices of bisyllables, and of
% ultimate-stressed bisyllables.
% There are 15388 bisyllables. Call this number N.
U = L.syl == 2;
U21 = L.syl == 2 & L.cstress == 1;

% Indices that are 1 in U, for mapping back to L.
I = find(U);

% Small indices are the row (item) indices of X and Y.
% Large indices are item indices in L.
% Where k is a small index, the corresponding large index is I(k).
% As an illustration, this looks up the word form for item 2.
% L.word(I(2)) => "demais".

% Combine acoustic and FSM weights. The result has size Nx4.
W1 = cellfun(@(x,y) x + acoustic_scale * y,L.weight1,L.weight2,'UniformOutput',false)';

% Combined weights scaled down by duration.
% This produces weights in the range 7.0 to 9.5.
W2 = cellfun(@(x,y) x ./ y,W1,num2cell(D),'UniformOutput',false);

% The above are cell arrays using large indices, and include
% weight-possibilites for non-bisyllables.
% Redefine them to be Nx2 matrices, with small row indices,
% giving two weight possibilies for each bi-syllables.

W1 = cell2mat(W1(U));
W2 = cell2mat(W2(U));

% Corresponding matrices of weights 
% U21w = cell2mat(W2(U21));
% U22w = cell2mat(W2(U22));
% U2w = cell2mat(W2(U));  THIS is now W2.
% U2w has dimensions 402 x 2, with entries in the range 7 to 9.

% Signed distance to equal-weight diagonal based on duration-scaled
% weights W2. These range from circa 5.2 to 12.4, but are mostly
% in the range 7 to 9.  This is a single number for each item.

Xw = (W2(:,2) - W2(:,1)) ./ sqrt(2);

% Signed difference between unscaled weights realizing
% penultimately and ultimately stressed word forms.
% It is the signed difference in point cross entropy
% for the two readings.
Xwe = W1(:,2) - W1(:,1);

% Xw and Xwe are set up so that positive indicates ultimate stress,
% and negative indicates penultimate stress.

%%%%%%%% Duration %%%%%%%%

% Matrices of vowel duration
% U21d = cell2mat(L.voweldur(U21)');
% U22d = cell2mat(L.voweldur(U22)');
Xd = cell2mat(L.voweldur(U)');
% U2d has dimensions N x 2, with entries (in centiseconds).
% giving the durations of the two vowels.
% The range is 3 to circa 20.

% Feature matrix. Row indices are items, columns are ..
X = [Xw,Xd];

% The same with the alternative version of Xw.
Xe = [Xwe,Xd];


% The class vector is a boolean vector, with 1 indicating final stress (21)
% and 0 indicating initial stress (22). There are two classes.

Y = U21(U);




% Find the loss from comparing weights directly.
% This should agree with Kaldi classification by alignment.
R.wLoss = nnz((Xwe > 0) == Y) / length(Y);
disp(R.wLoss);
% 0.8965

% Fit svm using 3 columns
% R.svm = fitcsvm(X,Y,'Standardize',true,'KernelScale','auto','KernelFunction','linear');
 
% Crossvalidate
% R.csvm = crossval(R.svm);
 

% Loss
% R.svmLoss = kfoldLoss(R.csvm);
% linear 0.914
% linear 0.0912
% 1 - R.svmLoss
% 0.9088
% The improvement is more that 1 point.

% Do the same with rbf
% R.rsvm = fitcsvm(X,Y,'Standardize',true,'KernelScale','auto','KernelFunction','rbf');
% R.rcsvm = crossval(R.rsvm);
% R.rsvmLoss = kfoldLoss(R.rcsvm);
% disp(1 - R.rsvmLoss);
% 0.9044 worse!

% Do the same using Xwe.
% R.ersvm = fitcsvm(Xe,Y,'Standardize',true,'KernelScale','auto','KernelFunction','rbf');
% R.ercsvm = crossval(R.ersvm);
% R.ersvmLoss = kfoldLoss(R.ercsvm);
% disp(1 - R.ersvmLoss);
% 0.9012

% Linear
% R.esvm = fitcsvm(Xe,Y,'Standardize',true,'KernelScale','auto','KernelFunction','linear');
% R.ecsvm = crossval(R.esvm);
% R.esvmLoss = kfoldLoss(R.ecsvm);
%disp(1 - R.esvmLoss);

% 0.9169  It's an absolute improvement of 0.0204, not trivial, and an error
% reduction of 19.71%, not trivial.

% 0.9169 - 0.8965 = 0.0204
% (0.9169 - 0.8965) / (1 - 0.8965) = 0.1971
% Just weights Xw or Xwe. This is just one number!

R.elsvm2 = fitcsvm(Xw,Y,'Standardize',true,'KernelScale','auto','KernelFunction','linear');
R.elcsvm2 = crossval(R.elsvm2);
R.elsvm2Loss = kfoldLoss(R.elcsvm2);
disp(1 - R.elsvm2Loss);
% 0.6367 How can it be so bad?

R.ersvm2 = fitcsvm(Xw,Y,'Standardize',true,'KernelScale','auto','KernelFunction','rbf');
R.ercsvm2 = crossval(R.ersvm2);
R.ersvm2Loss = kfoldLoss(R.ercsvm2);
disp(1 - R.ersvm2Loss);
% 0.8622

R.elsvm3 = fitcsvm(Xwe,Y,'Standardize',true,'KernelScale','auto','KernelFunction','linear');
R.elcsvm3 = crossval(R.elsvm3);
R.elsvm3Loss = kfoldLoss(R.elcsvm3);
disp(1 - R.elsvm3Loss);
% 0.6988 Again bad. Why can't the SVM find the point?  Maybe something
% breaks down with just one dimension.

R.ersvm3 = fitcsvm(Xwe,Y,'Standardize',true,'KernelScale','auto','KernelFunction','rbf');
R.ercsvm3 = crossval(R.ersvm3);
R.ersvm3Loss = kfoldLoss(R.ercsvm3);
disp(1 - R.ersvm3Loss);
% 0.8668

% Save the data
R.X = X;
R.Y = Y;

disp 1;

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

