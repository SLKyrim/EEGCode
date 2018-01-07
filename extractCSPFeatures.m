function features = extractCSPFeatures(eeg, CSPMatrix, nFilterPairs)
%EXTRACTCSPFEATURE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
nTrials = length(eeg);
features = zeros(nTrials, 2*nFilterPairs+1);
Filter = CSPMatrix([1:nFilterPairs (end-nFilterPairs+1):end],:);

for t=1:nTrials     
    projectedTrial = Filter * eeg{t}.X;    
    variances = var(projectedTrial,0,2);    
    for f=1:length(variances)
        features(t,f) = log(variances(f));
    end
    features(t,end) = eeg{t}.y;    
end
end

