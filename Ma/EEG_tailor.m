% ��3������EEG�źż������벽̬�ź�ͬ��

rawEEG = load('E:\EEGExoskeleton\Dataset\Ma\20180829\RawEEG.mat');
rawEEG = rawEEG.rawEEG;
cutEEG = cell(1,length(rawEEG));

for i = 1:length(rawEEG)
    flag_channel = rawEEG{1,i}(end,:); % ԭʼEEG�����еĴ��ͨ������
    
%     %% ���ƴ��ͨ�����ݹ۲����������Ƿ�����ȷ
%     figure
%     plot(1:length(flag_channel),flag_channel)
    
    %% ͨ�������������δ��λ��
    temp = flag_channel(1);
    for j = 2:length(flag_channel)
        if flag_channel(j) <= temp
            temp = flag_channel(j);
            continue;
        else
            temp = flag_channel(j);
            break;
        end
    end
    
    firstFlag = j; % ��һ�δ��λ��
    
    for j = (firstFlag+1):length(flag_channel)
        if flag_channel(j) <= temp
            temp = flag_channel(j);
            continue;
        else
            break;
        end
    end
    
    SecondFlag = j; % �ڶ��δ��λ��
    
    %% ��ȡ�������λ�ü��EEG����
    cutEEG{1,i} = rawEEG{1,i}(1:32,firstFlag:SecondFlag);
end

save('E:\EEGExoskeleton\Dataset\Ma\20180829\cutEEG.mat','cutEEG');