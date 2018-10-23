filteredGait = load('E:\EEGExoskeleton\Dataset\Ma\20180829\filteredMotion.mat');

%% ��ȡ��ϥ��ֵ��͹�ֵ������
for cell_no = 1:length(filteredGait.filteredMotion)
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
    rightIndMin = index;
    
    % ��ͼ�����ֵ������Ƿ���ȷ
    figure
    hold on
    plot(rightKnee)
    plot(rightIndMax,rightKnee(rightIndMax),'k*')
    plot(rightIndMin,rightKnee(rightIndMin),'r^') 
end

%% ��ȡ��ϥ��ֵ��͹�ֵ������
for cell_no = 1:length(filteredGait.filteredMotion)
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
    leftIndMin = index;
    
    % ��ͼ�����ֵ������Ƿ���ȷ
    figure
    hold on
    plot(leftKnee)
    plot(leftIndMax,leftKnee(leftIndMax),'k*')
    plot(leftIndMin,leftKnee(leftIndMin),'r^') 
end


