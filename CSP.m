function CSPMatrix = CSP(eeg, nClass, nChannel)
%CSP �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
if nClass ~= 2
    disp('ERROR! CSP can only be used for two classes');
    return;
end

nSamples = length(eeg);

%����Э�������2*1
covMatrices = cell(nClass, 1);

%����trialCov���� 32*32*100
trialCov = zeros(nChannel, nChannel, nSamples);

for n=1:nSamples
   X = eeg{n}.X;
   XX = X * X';
   trialCov(:, :, n) = XX ./ trace(XX);
end

for n=1:nSamples
    labels(n)=eeg{n}.y;
end

clabels=unique(labels);

Acount = 0;
Bcount = 0;
for n=1:nSamples
    y = eeg{n}.y;
    if y == clabels(1)
        Acount = Acount + 1;
        classACov(:, :, Acount) = trialCov(:, :, n);
    end
    if y == clabels(2)
        Bcount = Bcount + 1;
        classBCov(:, :, Bcount) = trialCov(:, :, n);
    end   
end

classACov = mean(classACov, 3)/length(classACov);
classBCov = mean(classBCov, 3)/length(classBCov);

M = inv(classBCov) * classACov;
[U D] = eig(M);
eigenvalues = diag(D);
[eigenvalues, egIndex] = sort(eigenvalues, 'descend');
U = U(:,egIndex);
CSPMatrix = U';
end

