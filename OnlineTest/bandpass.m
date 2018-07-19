% ��data��up��lowΪ���½�ֹƵ�ʽ��д�ͨ�˲�
function re = bandpass(data,up,low)
warning off

fs = 512; % EEG�ɼ�Ƶ��
row = size(data,1); % EEG����ͨ����
col = size(data,2); % EEG����

Wn = [2*up/fs 2*low/fs];
[b,a] = butter(4,Wn,'bandpass');
data_filtered = zeros(row, col);

for i = 1 : row
    % data_filtered(i,:) = filter(b,a, data(i,:));
    data_filtered(i,:) = filtfilt(b,a, data(i,:));
end

re = data_filtered;

end