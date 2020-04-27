clear variables
tic
mu=0.001
mu_dec=0.2
neuron_number=[1];
%1 hidden layer me 10 neurons gia 2 layers, 8a htane [10,5] me 10 neurons
%sto 1o kai 5 sto 2o px
RMSErr=[];
MAErr=[];
trainingdata={};
for sot=1:5
data = dlmread(['u' num2str(sot) '.base']);
datatest = dlmread(['u' num2str(sot) '.test']);
%alternatively
%data= dlmread(sprintf('u%d.base),sot))

[input, desired_output] = preprocess_data(data);
[inputtest, desired_outputtest] = preprocess_data(datatest);

 

layer_sizes = [neuron_number];
net = feedforwardnet(layer_sizes);
net = configure(net,input, desired_output);
net.layers{1}.transferFcn = 'tansig'
net.layers{end}.size = 1682;
net.layers{end}.dimensions = 1682;
net.layers{end}.transferFcn = 'purelin'
net.trainParam.mu=mu;
net.trainParam.mu_dec=mu_dec;
net.output.processFcns = {'mapminmax'};
net.input.processFcns = {'mapminmax'};
[net, tr] = train(net,input,desired_output,'useGPU','yes');
%[net,tr] = train(net,input,desired_output, 'showResources','yes'); 
%etsi 8a htane me CPU, htan para poly argo
trainingdata{end+1}=tr
outputtest=net(inputtest);
RMSErr(end+1) = calculateRMSE(outputtest, desired_outputtest)
MAErr(end+1) = calculateMAE(outputtest, desired_outputtest)

%view(net)

end
time_elapsed=toc 
TotalRMSErr=mean(RMSErr);
TotalMAErr=mean(MAErr);
save(['data' date() sprintf('neurons %d  mu %f ormh %f.mat', neuron_number, mu, mu_dec)])

function [input, desired_output] = preprocess_data(data)
data(:,4) = [];
old_user_id = 0;
user_data = {[]};
% Loop over all rows.
for i = 1:size(data,1)
    row = data(i,:);
    % Extract the user ID from the row.
    user_id = row(1);

    if (user_id == old_user_id) || (i == 1)
        % Still on the same user, accumulate his rows in user_data.
        user_data{end} = [user_data{end} ; row];
    else
        % New user, add a new cell in user_data and start accumulating the
        % user's data.
        user_data{end + 1} = row;
    end
    old_user_id = user_id;
end
% user_data now contains one user's data in each cell.

% Loop over all user data cells to convert them to the appropriate format.
for i = 1:length(user_data)
    A = user_data{i};
    user_data{i} = [[A(1,1) ; A(1,1)] A(:,2:3)'];
end



filled_user_data=zeros(length(user_data), 1682+1);
for k = 1:size(user_data,2)
avg_rating= mean(user_data{k}(2,2:end));
user_data{k}(2,2:end) =  user_data{k}(2,2:end) - mean(user_data{k}(2,2:end));
 %to neo diasthma timwn afou to max mean mporei na einai 5 kai to min mean mporei na einai 1 8a einai -4 ews 4 
    %edw na dhmiourgw kai na insertarw ta absent movies prepei na to xeiristw me kapoio for/if
    % sto absent movie na dinw rating to mean pou eswsa panw (giati alliws 8a
    % epairne to mean tou centered rating) mallon prepei na einai exw apo auth
    % thn for sthn for tou i
    
   
    filled_user_data(k,:) = zeros(1,1682+1);
    filled_user_data(k,1)= user_data{k}(1,1);
    for i = 2:size(user_data{k},2)
    movie_id = user_data{k}(1,i);
    movie_rating = user_data{k}(2,i);
    filled_user_data(k, 1+movie_id)=movie_rating;

end

end



% scaling se [-1,1] logw sigmoid
filled_user_data(:,2:end)=(filled_user_data(:,2:end))/4;


user_ids=filled_user_data(:,1);
input=zeros(943, length(user_ids));
for i=1:length(user_ids)
    input(user_ids(i),i)=1;
end


desired_output = filled_user_data(:,2:end)';
end





function RMSE = calculateRMSE(outputtest, desired_outputtest)
d = (outputtest-desired_outputtest).^2;
N=mean(d,1);
RMSE=mean(sqrt(N));

end



function MAE = calculateMAE(outputtest, desired_outputtest)
d = abs((outputtest-desired_outputtest));
N=mean(d,1);
MAE=mean(N);

end
