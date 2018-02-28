 % ��һ��

% ��ĳһ���Զ��󣨶�Ӧ��Ҫ�������Զ���ı��id_subject����
% EEG��bdf��ʽ��ԭʼ���ݺ�ת��txt��ʽ�Ĳ�̬���ݷֱ�洢Ϊmat��ʽ�ļ�

% �������Զ�����е������������num_sample
% ֻѡȡ����ϥ�����ݣ�֮���ж������������ϰ�����Python�ж�
% ������Ӧ������-8����ϥ-11������-17����ϥ20

% ������ɵ�mat�ļ�ΪԪ�飬ÿһ����Ա��Ӧ���Զ���һ����������� 

id_subject = 1; % ���Զ���ID��
%motion_flag = 11; 

if id_subject < 10
    path = ['E:\EEGExoskeleton\Dataset\Subject_0' num2str(id_subject) '\raw_EEG\*.bdf'];
else
    path = ['E:\EEGExoskeleton\Dataset\Subject_' num2str(id_subject) '\raw_EEG\*.bdf'];
end
num_sample = length(dir(path)); % �����ļ���


rawEEG = cell(1,num_sample);
rawMotion = cell(1,num_sample);

if id_subject < 10
    for n = 1:num_sample
        if n < 10
            motion_filename = ['E:\EEGExoskeleton\Dataset\Subject_0' num2str(id_subject) '\txt_Gait\0' num2str(n) 'Char00_biped.txt']; % ��ȡ��̬�ı��ļ����ļ���
            eeg_filename = ['E:\EEGExoskeleton\Dataset\Subject_0' num2str(id_subject) '\raw_EEG\0' num2str(n) '.bdf']; % ��ȡEEG�ļ����ļ���
        else
            motion_filename = ['E:\EEGExoskeleton\Dataset\Subject_0' num2str(id_subject) '\txt_Gait\' num2str(n) 'Char00_biped.txt'];
            eeg_filename = ['E:\EEGExoskeleton\Dataset\Subject_0' num2str(id_subject) '\raw_EEG\' num2str(n) '.bdf'];
        end
        rawEEG{1,n} = eeg_read_bdf(eeg_filename,'all','n');
        temp = load(motion_filename);
        rawMotion{1,n} = [temp(:,11) temp(:,20)];% ��������ֻ��������ϥ������
    end
else
    for n = 1:num_sample
        if n < 10
            motion_filename = ['E:\EEGExoskeleton\Dataset\Subject_' num2str(id_subject) '\txt_Gait\0' num2str(n) 'Char00_biped.txt']; % ��ȡ��̬�ı��ļ����ļ���
            eeg_filename = ['E:\EEGExoskeleton\Dataset\Subject_' num2str(id_subject) '\raw_EEG\0' num2str(n) '.bdf']; % ��ȡEEG�ļ����ļ���
        else
            motion_filename = ['E:\EEGExoskeleton\Dataset\Subject_' num2str(id_subject) '\txt_Gait\' num2str(n) 'Char00_biped.txt'];
            eeg_filename = ['E:\EEGExoskeleton\Dataset\Subject_' num2str(id_subject) '\raw_EEG\' num2str(n) '.bdf'];
        end
        rawEEG{1,n} = eeg_read_bdf(eeg_filename,'all','n');
        temp = load(motion_filename);
        rawMotion{1,n} = [temp(:,11) temp(:,20)];% ��������ֻ��������ϥ������
    end
end

if id_subject < 10
    save_eeg_filename = ['E:\EEGExoskeleton\EEGProcessor\Subject_0' num2str(id_subject) '_Data\Subject_0' num2str(id_subject) '_RawEEG.mat'];
    save_motion_filename = ['E:\EEGExoskeleton\EEGProcessor\Subject_0' num2str(id_subject) '_Data\Subject_0' num2str(id_subject) '_RawMotion.mat'];
    save(save_eeg_filename,'rawEEG');
    save(save_motion_filename,'rawMotion');
else
    save_eeg_filename = ['E:\EEGExoskeleton\EEGProcessor\Subject_' num2str(id_subject) '_Data\Subject_' num2str(id_subject) '_RawEEG.mat'];
    save_motion_filename = ['E:\EEGExoskeleton\EEGProcessor\Subject_' num2str(id_subject) '_Data\Subject_' num2str(id_subject) '_RawMotion.mat'];
    save(save_eeg_filename,'rawEEG');
    save(save_motion_filename,'rawMotion');
end
