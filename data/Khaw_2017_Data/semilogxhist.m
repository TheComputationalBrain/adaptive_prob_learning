

function semilogxhist(val,M)

% semilogxhist - generate histogram with M bars and log-scale x axis
if nargin<2;
    M=min(30,sqrt(length(val)));
end

vmin=min(val); vmax=max(val);
edges=vmin*(vmax/vmin).^([0:M]/M);
count=histc(val,edges);

if size(count,2)==1
    count=count';
end

x=edges(sort([1:M 1:M]));
y=[0 count(sort([1:M-1 1:M-1])) 0];
% outline only: semilogx(x, y, '-');
plot(x, y, '-'); fill(x, y, 'k'); set(gca,'XScale','log');
set(gca, 'XTick', [1 3 10 32 100 320])
return 