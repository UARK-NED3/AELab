function [theHistogram, hc, img] = histogram(t, phase, pdevents, qMin, qMax, bipolar, logarithmic, numXPoints, numYPoints)

maxy = numYPoints;
if (bipolar)
    maxy = round(maxy * 0.5);
end

if (logarithmic)
   gain=maxy/log10(qMax/qMin);
   offset=-log10(qMin)*gain;
else
   gain=maxy/(qMax-qMin);
   offset= -qMin * gain;
end

theHistogram=zeros(numXPoints, numYPoints);
n=min(find(pdevents == 0)) - 1;
if (isempty(n))
    n = length(pdevents);
end
n=min(length(t), min(length(phase), n));
dt=t(n)-t(1);
pdevents = pdevents(1:n);
phase=phase(1:n);
if (logarithmic)
    work = [round(log10(abs(pdevents))*gain + offset) round(phase * (numXPoints - 1) + 1)];
    idx = find(work(:,1) >= 0);
    work = work(idx,:);
    pdevents=pdevents(idx);
elseif (bipolar)
    work = [round(pdevents*gain+offset) round(phase * (numXPoints - 1) + 1)];
else
    work = [round(abs(pdevents)*gain+offset) round(phase * (numXPoints - 1) + 1)];
end
if (logarithmic && bipolar)
    idx = find(pdevents < 0);
    work(idx, 1) = work(idx, 1) * -1;
end
if (bipolar)
    work(:,1) = round(work(:,1) + numYPoints *0.5);
end
work = work(intersect(find(work(:,1) > 0), find(work(:,1) <= numYPoints)), :);

theHistogram=accumarray(work, 1, [numYPoints numXPoints]);

hc=zeros(numYPoints, numXPoints);

mx=max(max(theHistogram));
mn=min(theHistogram(find(theHistogram > 0)));

numCol=length(colormap);

m=(((mx*0.85)/mn) ^ (numCol/(numCol-1))) * mn;
root = (m/mn) ^(1/numCol);
prev=mx + 1;
for i=1:numCol
    val=mn*root^(numCol - i);
    hc(intersect(find(theHistogram>=val), find(theHistogram<prev))) = numCol - i;
    prev=val;
end

theHistogram = theHistogram / dt;

img=flipud(hc);
