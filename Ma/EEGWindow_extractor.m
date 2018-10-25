% ��4�������ڲ�̬ģʽ�л�λ��������ȡ������ǩ����EEG��

eeg = load('E:\EEGExoskeleton\Dataset\Ma\20180829\cutEEG.mat');
gaitSwitch_index = load('E:\EEGExoskeleton\Dataset\Ma\20180829\gaitSwitchIndex.mat');
gait = load('E:\EEGExoskeleton\Dataset\Ma\20180829\filteredMotion.mat');

eeg = eeg.cutEEG;
gaitSwitch_index = gaitSwitch_index.gaitSwitchIndex;
gait = gait.filteredMotion;

fs_eeg = 512; % EEG sampling rate (Hz)
fs_gait = 121; % gait sampling rate (Hz)
eeg_winWidth = 384; % the width of eeg window (384 sample points = 750 ms)
gait_winWidth = fs_gait / fs_eeg * eeg_winWidth; % the width of eeg window in gait data

output = {}; % �������ǩ��EEG����������Ϊ�����ļ�
count = 1; % output�������ݵļ�����
for i = 1:length(gait)
    % �в�̬�л���ͼ���������ֱ�Ϊ����ֹ���������ߣ��������߽�����¥�ݣ�����¥�ݽ��������ߣ��������߽���ֹ
    yep_index = gaitSwitch_index{i,1}; 
    % û�в�̬�л���ͼ����������Ϊ���л���ͼ�ĵ���м�㣨��Ϊ�м��һ���ڸò�̬�����м�Σ�
    % �ֱ�Ϊ���������߽�����¥�ݼ��е㣨�������߶Σ�������¥�ݽ��������߼��е㣨����¥�ݶΣ���
    % �������߽���ֹ���е㣨�������߶Σ�����ֹ����̬���һ������е㣨��ֹ�Σ�
    nop_index = [(yep_index(1)+yep_index(2))/2, (yep_index(2)+yep_index(3))/2, (yep_index(3)+yep_index(4))/2, (yep_index(4)+length(gait{1,i}))/2];
    nop_index = round(nop_index); % ����ȡ��
    
    % ����̬����ת��ΪEEG����
    eeg_yep_index = yep_index * fs_eeg / fs_gait;
    eeg_yep_index = round(eeg_yep_index);
    eeg_nop_index = nop_index * fs_eeg / fs_gait;
    eeg_nop_index = round(eeg_nop_index);
    
    for j = 1:length(eeg_yep_index)
        % �л�����ǰȡ����Ϊ���л���ͼ������ǩΪ1
        yep_eegWin = eeg{1,i}(:,eeg_yep_index(j)-eeg_winWidth+1:eeg_yep_index(j));
        output{count,1} = yep_eegWin;
        output{count,2} = 1;
        % �м��������ȡ����Ϊ���л���ͼ������ǩΪ-1
        nop_eegWin = eeg{1,i}(:,eeg_nop_index(j)-eeg_winWidth/2+1:eeg_nop_index(j)+eeg_winWidth/2);
        output{count+1,1} = yep_eegWin;
        output{count+1,2} = -1;
        % ���¼�����
        count = count + 2;
    end
end

save('E:\EEGExoskeleton\Dataset\Ma\20180829\labeledEEG.mat','output');

% %% ��ȡ��������ʾ��ͼ
% test = gait{1,1}(:,2);
% index = gaitSwitch_index{1,1};
% figure % ���л���̬��ͼ�Ĵ���ʾ��ͼ
% hold on 
% plot(1:length(test), test)
% plot(index, test(index), 'k*')
% for i = 1:length(index)
%     rectangle('Position',[index(i) - gait_winWidth, test(index(i)), gait_winWidth, 40], 'EdgeColor','r')
% end
% 
% no_index = [(index(1)+index(2))/2, (index(2)+index(3))/2, (index(3)+index(4))/2, (index(4)+length(test))/2];
% no_index = round(no_index); % ���л���ͼ������
% % figure % ���л���̬��ͼ�Ĵ���ʾ��ͼ
% % hold on
% % plot(1:length(test), test)
% plot(no_index, 50, 'k*')
% for i = 1:length(no_index)
%     rectangle('Position',[no_index(i) - gait_winWidth/2, 30, gait_winWidth, 40], 'EdgeColor','g')
% end
