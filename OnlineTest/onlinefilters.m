% �������Ķ����˲�
function re = onlinefilters(out_store)

BACK = 20; % ���ݵ���
THRED = 18; % ���������ֵ % �ڻ��ݵ���������ͼ��������������ֵ���������ͼ����
thres = 5; % ������Ϊ��Խ��ͼ��1���ĸ�����������ֵthresʱ��ȫ�����-1
thres_inver = 15; % �����˲���ֵ����������Խ��ͼ��Ķ�-1�β���1
                
count_filter = 0;
out_length = length(out_store);
output_1 = [];

%% һ���˲�
% һ���˲���α������ǰȡBACK�����ı�ǩ��
% BACK�����б�ǩ����������ֵTHRED�������Խ����
for i = BACK : out_length
   for j  = i - BACK + 1 : i
      if out_store(j) == 1
          count_filter = count_filter + 1;
      else
          continue
      end
   end
   
   if count_filter >= THRED
       output_1 = [output_1 1];
       count_filter = 0;
   else
       output_1 = [output_1 -1];
       count_filter = 0;
   end
end

%% �����˲�
% �����˲���������Ϊ�޿�Խ��ͼ��-1���ĸ�����������ֵthres_interʱ��ȫ�����1
output_2 = output_1;
count_filter = 0;

for i = 1 : length(output_1)
    if output_2(i) == -1
        count_filter = count_filter + 1;
    else
        if count_filter < thres_inver
            for j = 1 : count_filter
                output_2(i-j) = 1;
            end
            count_filter = 0;
        else
            count_filter = 0;
            continue
        end
    end
end
output_2(end) = -1;

%% �����˲�
% ������Ϊ��Խ��ͼ��1���ĸ�����������ֵthresʱ��ȫ�����-1

count_filter = 0;
for i = 1 : length(output_2)
    if output_2(i) == 1
        if i == length(output_2) - 1
            for j = 1 : count_filter
                output_2(i-j) = -1;
            end
        else
            count_filter = count_filter + 1;
        end
    else
        if count_filter < thres
            for j = 1 : count_filter
                output_2(i-j) = -1;
            end
            count_filter = 0;
        else
            count_filter = 0;
        end
    end
end

re = output_2;

end