function lineTrigger = importTMData(folder, qUnit);

fileName = sprintf('%s\\%s.TM', folder, qUnit);

file = fopen(fileName, 'rb');

if file == -1
   msg = sprintf('file %s could not be opened', fileName);
   error(msg);
end

fseek(file, 0, 'bof');
lineTrigger = fread(file, inf, 'float64');

fclose(file);

