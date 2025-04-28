save_names = { '1.csv','2.csv','3.csv', '4.csv','5.csv'};
file_names = { 
        '1.mat'
        '2.mat'
        '3.mat'
        '4.mat'
        '5.mat'
       
              };

data_path = '';
save_path = '';

for f = 1:length(file_names)
    file = fullfile(data_path,  file_names{f});
    disp(file)
    
    mat_data = load(file);
    data_field = fieldnames(mat_data);
    data = mat_data.(data_field{1});
    
    % Set the CSV file path
    csvFileName = fullfile(save_path, save_names{f});
    
    % Check if the directory exists, if not, create it
    [folderPath, ~, ~] = fileparts(csvFileName);
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
        disp(['Directory created: ', folderPath]);
    end
    
    % Write the data to the CSV file
    writetable(data, csvFileName);
    
    disp(['Table has been written to ', csvFileName]);
end

