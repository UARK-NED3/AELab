function [q_tm, q] = importPDData(folder, qUnit)
    fileName = sprintf('%s\\%s.PD', folder, qUnit);
    file = fopen(fileName, 'rb');

    if file == -1
        error('Could not open file: %s', fileName);
    end

    % Determine number of events
    fseek(file, 0, 'eof');
    fileSize = ftell(file);
    numEvents = fileSize / (4 + 8); % each event = 12 bytes
    fseek(file, 0, 'bof');

    q = zeros(1, numEvents);
    q_tm = zeros(1, numEvents);

    for i = 1:numEvents
        q(i) = fread(file, 1, 'float32');   % read PD magnitude
        q_tm(i) = fread(file, 1, 'float64'); % read timestamp
    end

    fclose(file);
end