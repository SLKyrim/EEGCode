% ��ԭʼ��̬���ݽ��е�ͨ�˲�

rawGait = load('E:\EEGExoskeleton\Dataset\Ma\20180829\RawMotion.mat');

fs = 121; % ������׽ϵͳ����Ƶ��121Hz
Wn = 1; % ��ֹƵ��1Hz
[b,a] = butter(4, 2*Wn/fs);

filteredMotion = cell(1,length(rawGait.rawMotion));
for cell_no = 1:length(rawGait.rawMotion)
    rawRightKnee = rawGait.rawMotion{1,cell_no}(:,1); % ��ϥԭʼ����
    rawLeftKnee = rawGait.rawMotion{1,cell_no}(:,2); % ��ϥԭʼ����
    
    % ��ԭʼ��̬���ݽ��е�ͨ�˲�
    rightKnee = filtfilt(b,a,rawRightKnee);
    leftKnee = filtfilt(b,a,rawLeftKnee);
    
    filteredMotion{1,cell_no} = [rightKnee leftKnee];
end

save('E:\EEGExoskeleton\Dataset\Ma\20180829\filteredMotion.mat','filteredMotion');