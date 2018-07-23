%% ��1��

%�����޸��ļ���ʽ
%bvh->txt
%�Ѹ��ļ���������ӦҪ�޸ĵ��ļ������ļ����Ｔ��
%bvh�ǲ�̬�����ļ�

clear, clc;
files=dir('*.bvh');
files_count=length(files);
for i=1:files_count
    oldfilename=files(i).name;
    len=length(oldfilename);
    newfilename=[oldfilename(1:len-4), '.txt'];
    eval(['!rename' 32 oldfilename 32 newfilename]);
    i=i
end


% ɾ��ת����txt��bvh�ļ����ļ�ͷ��ֻ������ֵ

files=dir('.\*.txt');
line = 0;
count = 0; % ������
for i=1:length(files)
    filename=files(i).name;
    temp = textread(filename,'','headerlines',312);
    save(filename,'temp','-ascii');
    count = count + 1
    line = 0;
end


%% ��2��

% ��ĳһ���Զ��󣨶�Ӧ��Ҫ�������Զ���ı��id_subject����
% EEG��bdf��ʽ��ԭʼ���ݺ�ת��txt��ʽ�Ĳ�̬���ݷֱ�洢Ϊmat��ʽ�ļ�

% �������Զ�����е������������num_sample
% ֻѡȡ����ϥ�����ݣ�֮���ж������������ϰ�����Python�ж�
% ������Ӧ������-8����ϥ-11������-17����ϥ20

% ������ɵ�mat�ļ�ΪԪ�飬ÿһ����Ա��Ӧ���Զ���һ����������� 

eeg_files = dir('*.bdf');
gait_files = dir('*.txt');
num_sample = length(eeg_files); % �����ļ���

rawEEG = cell(1,num_sample);
rawMotion = cell(1,num_sample);

count = 0; % ������
for n = 1:num_sample
    rawEEG{1,n} = eeg_read_bdf(eeg_files(n).name,'all','n');
    temp = load(gait_files(n).name);
    rawMotion{1,n} = [temp(:,11) temp(:,20)];% ��������ֻ��������ϥ������
    count = count + 1
end

save('RawEEG.mat','rawEEG');
save('RawMotion.mat','rawMotion');