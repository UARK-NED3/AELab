function phase = importPHData(folder, qUnit);

fileName = sprintf('%s\\%s.PH', folder, qUnit);

file = fopen(fileName, 'rb');

if file == -1
   msg = sprintf('file %s could not be opened', fileName);
   error(msg);
end

fseek(file, 0, 'bof');
phase = fread(file, inf, 'float64');

fclose(file);

