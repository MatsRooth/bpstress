function bpw2_stat2(matfile)

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
W1 = cellfun(@(x,y) x + acoustic_scale * y,L.weight1,L.weight2,'UniformOutput',false)';

% Combined weights scaled down by duration.
% This produces weights around 8.
W2 = cellfun(@(x,y) x ./ y,W1,num2cell(D),'UniformOutput',false);

% Logical indices of ultimate-stressed bisyllables,
% and penultimate-stressed bisyllables.
U21 = L.syl == 2 & L.cstress == 1;
U22 = L.syl == 2 & L.cstress == 2;

% Corresponding matrices of weights 
U21w = cell2mat(W2(U21));
U22w = cell2mat(W2(U22));

 


 

%%%%%%%%%%
% Reduce weights to one number and histogram them
% Signed distances to diagonal
W21 = (U21w(1:dcount,2) - U21w(1:dcount,1)) ./ sqrt(2);

W22 = (U22w(1:dcount,2) - U22w(1:dcount,1)) ./ sqrt(2);

 
%%%%%%%% Duration %%%%%%%%
% Matrices of vowel duration
U21d = cell2mat(L.voweldur(U21)');
U22d = cell2mat(L.voweldur(U22)');

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

