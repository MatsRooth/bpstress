function bpw2_stat3(matfile)
% Words with three or more syllables
if nargin < 1
    % matfile = '/local/matlab/Kaldi-alignments-matlab/data-bpn/tab4-sample.mat'; % Made with token_data_bpw2.
    matfile = '/local/matlab/Kaldi-alignments-matlab/data-bpn/tab4.mat'; % All the data.
end

% Load sets L to a structure. It has to be initialized first.
L = 0;
load(matfile);

% Number of data points to graph
dcount = 1000;

% Scale for combining the two weights.
acoustic_scale = 0.083333;
% Then combine by this formulat, see
% /projects/speech/sys/kaldi-master/egs/bp_ldcWestPoint/bpw2/exp/u1/decode_word_1/tab-min.awk
% weight = weight1 +  acoustic_scale * weight2;

% Duration in frames
D = cellfun(@sum,L.phonedur)';

% Combined weights
% We are interested only in readings 1-3, but there are more for longer
% words
W1 = cellfun(@(x,y) x + acoustic_scale * y,L.weight1,L.weight2,'UniformOutput',false)';

% Combined weights scaled down by duration.
% This produces weights around 8.
W2 = cellfun(@(x,y) x ./ y,W1,num2cell(D),'UniformOutput',false);

% Logical indices of ultimate-stressed triplus-syllables
% and penultimate-stressed triplus, and
% ante-penultimate tripus
U31 = L.syl > 2 & L.cstress == 1;
U32 = L.syl > 2 & L.cstress == 2;
U33 = L.syl > 2 & L.cstress == 3;

% Logical indices of all tokens with three or more syllables
U3 = L.syl > 2;

% Corresponding matrices of weights, with varying number of readings.
% Cell3mat can't be applied.
U31wv = W2(U31);  % 1584 3
U32wv = W2(U32);  % 7331 3
U33wv = W2(U33);  %  336 3

% Select three columns and map to matrix
% Each token is characterized by its weights in three readings.
U31w = cell2mat(cellfun(@(x) [x(1),x(2),x(3)], U31wv,'UniformOutput',false));
U32w = cell2mat(cellfun(@(x) [x(1),x(2),x(3)], U32wv,'UniformOutput',false));
U33w = cell2mat(cellfun(@(x) [x(1),x(2),x(3)], U33wv,'UniformOutput',false));

% Sy = L.syl;
% Sy3 = Sy(1:100);


% 3x3 contingency table
Ct = [nnz(U33w(:,3) < U33w(:,1) & U33w(:,3) < U33w(:,2)),nnz(U33w(:,2) < U33w(:,1) & U33w(:,2) < U33w(:,3)),nnz(U33w(:,1) < U33w(:,2) & U33w(:,1) < U33w(:,3));
      nnz(U32w(:,3) < U32w(:,1) & U32w(:,3) < U32w(:,2)),nnz(U32w(:,2) < U32w(:,1) & U32w(:,2) < U32w(:,3)),nnz(U32w(:,1) < U32w(:,2) & U32w(:,1) < U32w(:,3));
      nnz(U31w(:,3) < U31w(:,1) & U31w(:,3) < U31w(:,2)),nnz(U31w(:,2) < U31w(:,1) & U31w(:,2) < U31w(:,3)),nnz(U31w(:,1) < U31w(:,2) & U31w(:,1) < U31w(:,3))];

% Put ultimate stress in first row/column instead of last.  

Ct = rot90(Ct,2); 
disp('Contingency table');
disp(Ct);

% Majority class
Maj =  nnz(U32) / nnz(U3);
disp('Majority class (penultimate stress)');
disp(Maj);

% Classifcation from weights
% Correct guys are on the diagonal.
Wrat = (Ct(1,1) + Ct(2,2) + Ct(3,3)) / sum(sum(Ct));
disp('Classification from weights');
disp(Wrat);

disp('Classification from weights mapping 3 to 2');
disp((Ct(1,1) + Ct(2,2) + Ct(2,3)) / sum(sum(Ct)))

% Balanced error matrix
Ctb =  [Ct(1,:) / sum(Ct(1,:)); Ct(2,:) / sum(Ct(2,:)); Ct(3,:) / sum(Ct(3,:))];
disp('Row-normalized contingency table');
disp(Ctb);

% Balanced error rate
Ber = (Ctb(1,1) + Ctb(2,2) + Ctb(3,3)) / 3;
disp('Balanced correctness rate');
disp(Ber);

disp(1);
%%%%%%%%%%
% Reduce weights to one number and histogram them
% Signed distances to diagonal
%W21 = (U21w(1:dcount,2) - U21w(1:dcount,1)) ./ sqrt(2);

%W22 = (U22w(1:dcount,2) - U22w(1:dcount,1)) ./ sqrt(2);

% Rotatable 3D scatter plots of weights for three stress classes
figure();
scatter3(U33w(1:200,1),U33w(1:200,2),U33w(1:200,3),10,[0 0.5 0]);
hold;
scatter3(U32w(1:200,1),U32w(1:200,2),U32w(1:200,3),10,'b');
% Don't say hold again.
scatter3(U31w(1:200,1),U31w(1:200,2),U31w(1:200,3),10,'r');
rotate3d;

disp(1);

%%%%%%%% Duration %%%%%%%%
% Vowel durations. L.voweldur is not of uniform length,
% and the vowels need to count from the end. This is
% adjusted by the anonymous function.
% We assume L.voweldur has vowel lengths in time order.
U31d = cell2mat(cellfun(@(x) [x(length(x)),x(length(x)-1),x(length(x)-2)], L.voweldur(U31)','UniformOutput',false));
U32d = cell2mat(cellfun(@(x) [x(length(x)),x(length(x)-1),x(length(x)-2)], L.voweldur(U32)','UniformOutput',false));
U33d = cell2mat(cellfun(@(x) [x(length(x)),x(length(x)-1),x(length(x)-2)], L.voweldur(U33)','UniformOutput',false));

disp(1);

%
dcount = 200;

% Rotatable 3D scatter plots of vowel durations for three stress classes
figure();
scatter3(U33d(1:dcount,1)+(0.4 * rand(1,dcount))',U33d(1:dcount,2)+(0.4 * rand(1,dcount))',U33d(1:dcount,3)+(0.4 * rand(1,dcount))',10,[0 0.5 0]);
hold;
scatter3(U32d(1:dcount,1)+(0.4 * rand(1,dcount))',U32d(1:dcount,2)+(0.4 * rand(1,dcount))',U32d(1:dcount,3)+(0.4 * rand(1,dcount))',10,'b');
% Don't say hold again.
scatter3(U31d(1:dcount,1)+(0.4 * rand(1,dcount))',U31d(1:dcount,2)+(0.4 * rand(1,dcount))',U31d(1:dcount,3)+(0.4 * rand(1,dcount))',10,'r');
rotate3d;

disp(1);

figure();
scatter(U21d(1:dcount,1) + (0.9 * rand(1,dcount))',U21d(1:dcount,2) + (0.9 * rand(1,dcount))','blue');
axis([0 30 0 30]);
legend('lexical 21');
xlabel('initial vowel duration centiseconds (plus 0.9 noise)');
ylabel('final vowel duration centiseconds (plus 0.9 noise)');

figure();
scatter(U22d(1:dcount,1) + (0.9 * rand(1,dcount))',U22d(1:dcount,2) + (0.9 * rand(1,dcount))','red');
axis([0 30 0 30]);
legend('lexical 22');
xlabel('initial vowel duration centiseconds (plus 0.9 noise)');
ylabel('final vowel duration centiseconds (plus 0.9 noise)'); 

disp 1;

function box_ratio(xmin,ymax)
        
end


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

