% ��һ��

% ��ĳһ���Զ��󣨶�Ӧ��Ҫ�������Զ���ı��id_subject����
% EEG��bdf��ʽ��ԭʼ���ݺ�ת��txt��ʽ�Ĳ�̬���ݷֱ�洢Ϊmat��ʽ�ļ�

% �������Զ�����е������������num_sample
% �������Զ������������ϰ�ʹ�õ���(���Ȼ�����)����motion_flag

% ������ɵ�mat�ļ�ΪԪ�飬ÿһ����Ա��Ӧ���Զ���һ����������� 

num_sample = 15; % �����ļ���
motion_flag = 11; % ������Ӧ������-8����ϥ-11������-17����ϥ20
id_subject = 3; % ���Զ���ID��

rawEEG = cell(1,num_sample);
rawMotion = cell(1,num_sample);

for n = 1:num_sample
    if n < 10
        motion_filename = ['E:\EEGExoskeleton\Dataset\Subject_0' num2str(id_subject) '\txt_Gait\' num2str(0) num2str(n) 'Char00_biped.txt']; % ��ȡ��̬�ı��ļ����ļ���
        eeg_filename = ['E:\EEGExoskeleton\Dataset\Subject_0' num2str(id_subject) '\raw_EEG\' num2str(0) num2str(n) '.bdf']; % ��ȡEEG�ļ����ļ���
    else
        motion_filename = ['E:\EEGExoskeleton\Dataset\Subject_0' num2str(id_subject) '\txt_Gait\' num2str(n) 'Char00_biped.txt'];
        eeg_filename = ['E:\EEGExoskeleton\Dataset\Subject_0' num2str(id_subject) '\raw_EEG\' num2str(n) '.bdf'];
    end
    
    rawEEG{1,n} = eeg_read_bdf(eeg_filename,'all','n'); 
    temp = load(motion_filename);
    rawMotion{1,n} = temp(:,motion_flag);
end

save E:\EEGExoskeleton\EEGProcessor2\rawEEG_03 rawEEG;
save E:\EEGExoskeleton\EEGProcessor2\rawMotion_03 rawMotion;