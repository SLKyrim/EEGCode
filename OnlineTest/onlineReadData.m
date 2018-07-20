function onlineReadData()

close all
clear all
clc
warning off

% ����ʵ���������
win_len = 384; % ��������λ��������
csp = load('csp.mat');
csp = csp.csp; % ��ȡָ�����Զ����CSPͶӰ����
interval = 28; % ����������ÿ����26��������һ��ȡ������ % 384��Ϊ750ms��26����Ϊ50ms

% ������ݳ�ʼ��
global run; run = true; % �Ƿ����ѭ��
data_history = []; % ����EEG��ʷ��Ϣ��ʱ��Խ����������Խ���Ժ���Ҫ������Զ����ʷ�Ĺ���
count = 0; % ������ļ�����
out_store = []; % ��¼���ָ��


% TCPIP ��������
% configure % the folowing 4 values should match with your setings in Actiview and your network settings
% Decimationѡ��1/4��;
port = 778;                % the port that is configured in Actiview , delault = 8888
port2 = 4484;
ipadress = 'localhost';    % the ip adress of the pc that is running Actiview
Channels = 32;             % set to the same value as in Actiview "Channels sent by TCP"
Samples = 4;               % set to the same value as in Actiview "TCP samples/channel" % Samples = fs/Channels/4
words = Channels*Samples;
data_current = zeros(Samples, Channels); % ���β�����ȡ��EEG����

% open tcp connection % ���Ե�ɼ���������ͨ��Socket
tcpipClient = tcpip(ipadress, port, 'NetworkRole', 'client');
set(tcpipClient,'InputBufferSize',words*9); % input buffersize is 3 times the tcp block size % 1 word = 3 bytes
set(tcpipClient,'Timeout',5);
% open tcp connection % ���������λ����ͨ��Socket
% tcpipClient2 = tcpip('172.20.10.132', port2, 'NetworkRole', 'client');
% set(tcpipClient2,'InputBufferSize',words*9); % input buffersize is 3 times the tcp block size % 1 word = 3 bytes
% set(tcpipClient2,'Timeout',5);
try
    fopen(tcpipClient);
%     fopen(tcpipClient2);
catch
    disp('Actiview is unreachable please check if Actiview is running on the specified ip address and port number');
    run = false;
end

while run
    
    % ��ȡÿ��tcpip���͵���ֵ
    [rawData,temp,msg] = fread(tcpipClient,[3 words],'uint8');
    if temp ~= 3*words
        run = false;
        disp(msg);
        disp('Is Actiview running with the same settings as this example?');
        break
    end
       
    % reorder bytes from tcp stream into 32bit unsigned words
    % normaldata = rawData(3,:)*(256^3) + rawData(2,:)*(256^2) + rawData(1,:)*256 + 0;
    % 2018-4-27-��ȻTCP�������ݱ�Labview�������ݶ�256�����Ͱ���ʽ����256�õ���ʽ (���ǻ�õ��������������)
    normaldata = rawData(3,:)*(256^3) + rawData(2,:)*(256^2) + rawData(1,:)*256 + 0;
    % reorder the channels into a array [samples channels]
    i = 0 : Channels : words-1; % words-1 because the vector starts at 0
    for d = 1 : Channels
        data_current(1:Samples,d) = typecast(uint32(normaldata(i+d)),'int32');   %create a data struct where each channel has a seperate collum     
    end

    data_current = data_current; % ���벻��256���õ�����������ͬ��
    data_history = [data_history;data_current];
    count = count + 4; % EEG 512Hz�Ĳ���Ƶ�ʵĻ�ÿ��ѭ�������4����
    
    if count > win_len && mod(count,interval) == 0
        data = data_history';
        data = data(:,count-win_len+1:count);
        save('data.mat', 'data');
        pyObj = py.onlineClassifier.OnlineClassifier(); 

        out_store = [out_store str2double(char(pyObj.outputCmd()))]; % Pythonԭʼ�������ݴ����ԣ���תstring��ת����ȥ������
        out_length = length(out_store);
        
        if out_length > 20
            output_cmd = onlinefilters(out_store) % ��out_store���ж����˲�
        end        
    end
end


data_history = (data_history(:,1:32))';

% save('data_current.mat', 'data_current');
save('data_history.mat', 'data_history');
save('output_cmd.mat', 'output_cmd');
% save('count.mat','count');
% save('feat.mat','feat');
save('out_store.mat','out_store');
% save('time.mat','time');
% save('count_win.mat','count_win');


%�ر�tcpip
fclose(tcpipClient);
delete(tcpipClient);
% fclose(tcpipClient2);
% delete(tcpipClient2);

end