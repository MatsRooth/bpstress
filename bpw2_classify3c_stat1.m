function R = bpw2_classify3c_stat1(matfile,modelfile)

if nargin < 1
    %matfile = '/local/matlab/Kaldi-alignments-matlab/data-bpn/tab4-sample.mat'; % Made with token_data_bpw2.
    datafile = '/local/matlab/Kaldi-alignments-matlab/data-bpn/tab4.mat'; % All the data, 15388 bisyllables
    modelfile = '/local/matlab/bpstress/bpw2_classify3c.mat';
end

% Load sets L to a structure. It has to be initialized first.
% Do we need this?  Every thing we need should be stored in R.
%L = 0;
%load(datafile);

% Similarly for the model structure R
R = 0;
load(modelfile);


% Recover the data and dimension
X = R.X;
Y = R.Y;
dim = R.dim;

% Compute contingency table of counts
% (3,3) matrix
fprintf("Contingency table\n");
fprintf("Row index: lexical stress\n");
fprintf("Column index: predicted stress\n");

% Initialize the table
Cn = zeros(3,3);

% Tabulate the contingency table.
for k = 1:8
   % Predicted labels in testfold k
   lab = R.l{3,k};
   % Lexical labels for the same tokens
   lex = Y(testfold(k));
   % Actual stress positions
   for i = 1:3
       % Boolean indices of guys in lex with stress i
       % I = find(lex == i);
       I = (lex == i);
       % Predicted stress positions
       for j = 1:3
          % Boolean indices of guys with predicted stress j 
          J = (lab == j);
          % Increment a cell of the contingency table.
          Cn(i,j) = Cn(i,j) + nnz(I & J);
       end
   end
end

disp(Cn);

disp(1);

% R.l is a cell(3,8) array of labels for the 

% Scale for combining the two weights.
% acoustic_scale = 0.083333;
% Then combine by this formula, see
% /projects/speech/sys/kaldi-master/egs/bp_ldcWestPoint/bpw2/exp/u1/decode_word_1/tab-min.awk
% weight = weight1 +  acoustic_scale * weight2;

% Duration in frames
%D = cellfun(@sum,L.phonedur)';

% Combined weights
%W1 = cellfun(@(x,y) x + acoustic_scale * y,L.weight1,L.weight2,'UniformOutput',false)';

% Combined weights scaled down by duration.
% This produces weights in the range 7.0 to 9.5.
%W2 = cellfun(@(x,y) x ./ y,W1,num2cell(D),'UniformOutput',false);

% This part is like bpw2_stat3.m
% Logical indices of ultimate-stressed triplus-syllables
% and penultimate-stressed triplus, and
% ante-penultimate tripus
%U31 = L.syl > 2 & L.cstress == 1;
%U32 = L.syl > 2 & L.cstress == 2;
%U33 = L.syl > 2 & L.cstress == 3;

% Logical indices of all tokens with three or more syllables
%U3 = L.syl > 2;

% Indices that are 1 in U3, for mapping back to L.
%I3 = find(U3);

% Corresponding matrices of weights, with varying number of readings.
% Cell3mat can't be applied.
%U31wv = W2(U31);  % 1584 3
%U32wv = W2(U32);  % 7331 3
%U33wv = W2(U33);  %  336 3
%U3wv = W2(U3);

% Select three columns and map to matrix
% Each token is characterized by its weights in three readings.
%U31w = cell2mat(cellfun(@(x) [x(1),x(2),x(3)], U31wv,'UniformOutput',false));
%U32w = cell2mat(cellfun(@(x) [x(1),x(2),x(3)], U32wv,'UniformOutput',false));
%U33w = cell2mat(cellfun(@(x) [x(1),x(2),x(3)], U33wv,'UniformOutput',false));
%U3w = cell2mat(cellfun(@(x) [x(1),x(2),x(3)], U3wv,'UniformOutput',false));

%%%%%%%% Duration %%%%%%%%
% Vowel durations. L.voweldur is not of uniform length,
% and the vowels need to count from the end. This is
% adjusted by the anonymous function.
% We assume L.voweldur has vowel lengths in time order.
%U31d = cell2mat(cellfun(@(x) [x(length(x)),x(length(x)-1),x(length(x)-2)], L.voweldur(U31)','UniformOutput',false));
%U32d = cell2mat(cellfun(@(x) [x(length(x)),x(length(x)-1),x(length(x)-2)], L.voweldur(U32)','UniformOutput',false));
%U33d = cell2mat(cellfun(@(x) [x(length(x)),x(length(x)-1),x(length(x)-2)], L.voweldur(U33)','UniformOutput',false));
%U3d = cell2mat(cellfun(@(x) [x(length(x)),x(length(x)-1),x(length(x)-2)], L.voweldur(U3)','UniformOutput',false));



% Recover the data and dimension
X = R.X;
Y = R.Y;
dim = R.dim;

% 9821 items
dim = length(X(:,1));

% Functions for finding the folds in the noble eight-fold way.
% The functions return logical vectors, giving the indices of
% the kth train and test folds. The training data for fold k is
% X(trainfold(k),:)
    function I = trainfold(k)
        I =  ~((mod(0:dim,8) + 1) == k);
        I = I(1:dim);
    end
    function I = testfold(k)
        I = (mod(0:dim,8) + 1) == k;
        I = I(1:dim);
    end

disp(dim);
 

% Cell array of models
% Columns are 8 folds
% Rows are
%   1 just weight
%   2 just duration
%   3 both
 
% Predicted labels for test folds
%R.label_w = cell(1,8);
%R.label_d = cell(1,8);
%R.label_wd = cell(1,8);
% R.l = cell(3,8);

%for k = 1:8
%   disp(k);
%   R.l{1,k} = predict(R.mw{k},X(testfold(k),1:3));
%   R.l{2,k} = predict(R.md{k},X(testfold(k),4:6));
%   R.l{3,k} = predict(R.mwd{k},X(testfold(k),:));
%end
% First 20 labels in 1st fold: R.l{3,1}(1:20)'

% Save R
% This resulted in savename.mat. I moved it to bpw2_classify3c.mat.
%save savename R;
% save(savename,'R');

disp(1);  

%%% ===> Next, compute the 3x3 contingency table.
 
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

 

end

