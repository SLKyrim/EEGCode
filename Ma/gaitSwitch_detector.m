% ���ڵ�ͨ�˲���Ĳ�̬����Ѱ�Ҳ�̬ģʽ�л���λ��������

filteredGait = load('E:\EEGExoskeleton\Dataset\Ma\20180829\filteredMotion.mat');

gaitSwitchIndex = cell(length(filteredGait.filteredMotion),2);
for cell_no = 1:length(filteredGait.filteredMotion)
    %% ��ȡ��ϥ��ֵ��͹�ֵ������
    rightKnee = filteredGait.filteredMotion{1,cell_no}(:,1); % ��ͨ�˲�������ϥ��̬����
       
    rightIndMax=find(diff(sign(diff(rightKnee)))<0)+1;   % �����ϥ�ֲ����ֵ��λ��
    rightIndMin=find(diff(sign(diff(rightKnee)))>0)+1;   % �����ϥ�ֲ���Сֵ��λ��  

    % ȥ��С��30�ȵķ�ֵ������
    index = [];
    for i = 1:length(rightIndMax)
       if rightKnee(rightIndMax(i)) < 30
          % ��ȡС��30�ȵķ�ֵ������
          index = horzcat(index,i);
       end
    end
    rightIndMax(index) = []; % ��С��30�ȵķ�ֵ��ȥ��
    
    % ��ȡ������ֵ��ǰ�Ĺ�ֵ������
    index = [];
    for i = 1:length(rightIndMax)
       for j = 1:length(rightIndMin)
           if rightIndMin(j) > rightIndMax(i)
               index = horzcat(index,rightIndMin(j-1));
               break;
           else
               continue;
           end
       end
    end
    index = horzcat(index,rightIndMin(j)); % ��Ϊ��������Ҫ��Ĺ�ֵ��ĺ�һ����ֵ���������������л�����ֹ�Ĺ�ֵ��
    rightIndMin = index;
    
    % ֻ���²�̬ģʽ�л����Ĺ�ֵ��
    index = [1]; % ��Ϊһ����ֵ���Ǿ�ֹ�л����������ߵ�λ��
    for i = 1:length(rightIndMax)
        if i+1 < length(rightIndMax)
            % ����������������ά��
            if rightKnee(rightIndMax(i+1)) > 70 && rightKnee(rightIndMax(i)) < 60
                % ����һ��ϥ�ؽڽǶȴ���70����ǰϥ�ؽڽǶ�С��60ʱ��˵�����ڽ������������߲�̬�л�������¥�ݲ�̬
                index = horzcat(index,i+1);
            elseif rightKnee(rightIndMax(i+1)) < 60 && rightKnee(rightIndMax(i)) > 70
                % ����һ��ϥ�ؽڽǶ�С��60����ǰϥ�ؽڽǶȴ���70ʱ��˵�����ڽ���������¥�ݲ�̬�л����������߲�̬
                index = horzcat(index,i+1);
            end
        else
            break;
        end
    end
    index = horzcat(index, length(rightIndMin)); % ��Ϊ���һ����ֵ�������������л�����ֹ��λ��
    rightIndMin = rightIndMin(index);

    %% ��ȡ��ϥ��ֵ��͹�ֵ������
    leftKnee = filteredGait.filteredMotion{1,cell_no}(:,2);  % ��ͨ�˲�������ϥ��̬����
    
    leftIndMax=find(diff(sign(diff(leftKnee)))<0)+1;   % �����ϥ�ֲ����ֵ��λ��
    leftIndMin=find(diff(sign(diff(leftKnee)))>0)+1;   % �����ϥ�ֲ���Сֵ��λ��  
    
    % ȥ��С��30�ȵķ�ֵ������
    index = [];
    for i = 1:length(leftIndMax)
       if leftKnee(leftIndMax(i)) < 30
          % ��ȡС��30�ȵķ�ֵ������
          index = horzcat(index,i);
       end
    end
    leftIndMax(index) = []; % ��С��30�ȵķ�ֵ��ȥ��
    
    % ��ȡ������ֵ��ǰ�Ĺ�ֵ������
    index = [];
    for i = 1:length(leftIndMax)
       for j = 1:length(leftIndMin)
           if leftIndMin(j) > leftIndMax(i)
               index = horzcat(index,leftIndMin(j-1));
               break;
           else
               continue;
           end
       end
    end
    index = horzcat(index,leftIndMin(j)); % ��Ϊ��������Ҫ��Ĺ�ֵ��ĺ�һ����ֵ���������������л�����ֹ�Ĺ�ֵ��
    leftIndMin = index;
    
    % ֻ���²�̬ģʽ�л����Ĺ�ֵ��
    index = [1]; % ��Ϊһ����ֵ���Ǿ�ֹ�л����������ߵ�λ��
    for i = 1:length(leftIndMax)
        if i+1 < length(leftIndMax)
            % ����������������ά��
            if leftKnee(leftIndMax(i+1)) > 70 && leftKnee(leftIndMax(i)) < 60
                % ����һ��ϥ�ؽڽǶȴ���70����ǰϥ�ؽڽǶ�С��60ʱ��˵�����ڽ������������߲�̬�л�������¥�ݲ�̬
                index = horzcat(index,i+1);
            elseif leftKnee(leftIndMax(i+1)) < 60 && leftKnee(leftIndMax(i)) > 70
                % ����һ��ϥ�ؽڽǶ�С��60����ǰϥ�ؽڽǶȴ���70ʱ��˵�����ڽ���������¥�ݲ�̬�л����������߲�̬
                index = horzcat(index,i+1);
            end
        else
            break;
        end
    end
    index = horzcat(index, length(leftIndMin)); % ��Ϊ���һ����ֵ�������������л�����ֹ��λ��
    leftIndMin = leftIndMin(index);
    
    % ��ͼ�����ֵ������Ƿ���ȷ
    figure
    hold on
    plot(rightKnee)
    plot(rightIndMax,rightKnee(rightIndMax),'k*')
    plot(rightIndMin,rightKnee(rightIndMin),'r^') 
   
    figure
    hold on
    plot(leftKnee)
    plot(leftIndMax,leftKnee(leftIndMax),'k*')
    plot(leftIndMin,leftKnee(leftIndMin),'r^') 

    %% ���沽̬�л�����λ��
    if rightIndMin(1) < leftIndMin(1)
        % ��������ȿ���¥�ݣ��򱣴����ȵ�������Ϊָʾ��̬�л�������
        gaitSwitchIndex{cell_no,1} = rightIndMin;
        gaitSwitchIndex{cell_no,2} = 1; % label 1 ָʾΪ����
    else
        % ��������ȿ���¥�ݣ��򱣴����ȵ�������Ϊָʾ��̬�л�������
        gaitSwitchIndex{cell_no,1} = leftIndMin;
        gaitSwitchIndex{cell_no,2} = 2; % label 2 ָʾΪ����
    end
end

save('E:\EEGExoskeleton\Dataset\Ma\20180829\gaitSwitchIndex.mat','gaitSwitchIndex');


