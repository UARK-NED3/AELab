
% Test Code for MPD data export

folderPath = "C:\Users\templ\OneDrive\Desktop\MPD Export Stuff\20250512_11h13m42s";
fileID = "MPD 800 2.1.1-20250416_12h33m10s365ms";
[time_stamps, magnitudes] = importPDData(folderPath, fileID);

plot(time_stamps, magnitudes, '.')
xlabel('Time [s]')
ylabel('PD Magnitude [C or V]')
title('PD Events')

