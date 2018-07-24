function onlineReadData()

close all
clear all
clc
warning off

% ����ʵ���������
win_len = 384; % ��������λ��������;��������4��������
csp = load('csp.mat');
csp = csp.csp; % ��ȡָ�����Զ����CSPͶӰ����
interval = 28; % ����������ÿ����26��������һ��ȡ������ % 384��Ϊ750ms��26����Ϊ50ms

% ������ݳ�ʼ��
global run; run = true; % �Ƿ����ѭ��
data = []; % ���ʵʱEEG
count = 0; % ������ļ�����
out_store = []; % ��¼���ָ��
output_cmd = []; % �����˲������ָ��


% TCPIP ��������
% configure % the folowing 4 values should match with your setings in Actiview and your network settings
% Decimationѡ��1/4��;
port = 778;                % the port that is configured in Actiview , delault = 8888
port2 = 8080;
ipadress = 'localhost';    % the ip adress of the pc that is running Actiview
Channels = 32;             % set to the same value as in Actiview "Channels sent by TCP"
Samples = 4;               % set to the same value as in Actiview "TCP samples/channel" % Samples = fs/Channels/4
words = Channels*Samples;
data_current = zeros(Samples, Channels); % ���β�����ȡ��EEG����

while run
    tcpipClient = tcpip(ipadress, port, 'NetworkRole', 'client');
    set(tcpipClient,'InputBufferSize',words*9); % input buffersize is 3 times the tcp block size % 1 word = 3 bytes
    set(tcpipClient,'Timeout',5);
    fopen(tcpipClient);
    
    tcpipClient2 = tcpip('172.20.15.186', port2, 'NetworkRole', 'client');
    set(tcpipClient2,'InputBufferSize',words*9); % input buffersize is 3 times the tcp block size % 1 word = 3 bytes
    set(tcpipClient2,'Timeout',5); 
    fopen(tcpipClient2);
    
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

    data_current_t = data_current'; % ���벻��256���õ�����������ͬ��
    
    if length(data) ~= win_len
        % �տ�ʼ¼������
        data = [data data_current_t];
    else
        % ¼�������Ѵﵽ�趨����
        % ��ʵʱEEG������FIFO����ʹ�øô�����ԶΪwin_len
        % ��ǰ4������pop�������������µ�4������
        data = cat(2, data(:,5:end), data_current_t); 
        
        save('data.mat', 'data');
        pyObj = py.onlineClassifier.OnlineClassifier(); 
        
        if length(out_store) ~= 60
            % �����������δ�ﵽ����Ҫ��
            out_store = [out_store str2double(char(pyObj.outputCmd()))];
        else
            out_store = cat(2, out_store(2:end), str2double(char(pyObj.outputCmd())));
            
            output_cmd = onlinefilters(out_store) % ��out_store���ж����˲�
            
            if length(find(output_cmd(end) == 1)) == 1
                %������������20��ȫ��1ʱ���������1����
                fwrite(tcpipClient2,'1');
            else
                fwrite(tcpipClient2,'-1');
            end

        end
    end
        

    fclose(tcpipClient);
    delete(tcpipClient);
    fclose(tcpipClient2);
    delete(tcpipClient2);
    
end

% save('data_current_t.mat', 'data_current_t');
% save('data.mat', 'data');
% save('output_cmd.mat', 'output_cmd');
% save('count.mat','count');
% save('feat.mat','feat');
% save('out_store.mat','out_store');
% save('time.mat','time');
% save('count_win.mat','count_win');

%�ر�tcpip
fclose(tcpipClient);
delete(tcpipClient);
fclose(tcpipClient2);
delete(tcpipClient2);

end