%��1��

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
    path_file=[files(i).folder, '\', filename];     
    fid = fopen(path_file, 'r');
    start = 1;
    while start
        tline=fgetl(fid);
        line = line + 1; % ͳ���ļ�ͷ������
        if strcmpi(tline,char('Frame Time: 0.00800000'))
            start = 0;
            temp = textread(filename,'','headerlines',line);
            save(filename,'temp','-ascii');
            count = count + 1
            line = 0;
        end
        
        if line > 1000
            start = 0; % ������ѭ��
        end
    end
end
