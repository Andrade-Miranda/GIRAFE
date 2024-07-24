function [lrpvg,lrpvgColor,glpvg,glpvgColor, maxveloc, contours, edges, delta, signdelta,signdeltagl, To, sizepvg, D, V] = getdifpvg(seg, sizepvg, sequence)
% compute pvg (as from loscheller - lrpvg) and pvg (as from us.. - glpvg)
% maxveloc is the maximum speed profile
% seg : matrix of logical indices , size [M,N,X] same as sequence of
% high-speed images
% sizepvg : length of column vector 2M + 1~ sampling points used for the
% vocal contours
% default 256 for optimal visualization (glpvg will have half size)

if nargin==1
    sizepvg = 256;
end
warning('off','all');
% get open states To
To = getopenstates(seg);

% get glottal edges,  for every segment of the glottal area
[edges, To] = getedges(seg, To);

% get dorsal and versal endings and link em
[D, V] = linkendings(edges, To);

% connect dorsal and ventral endings with vocal fold edges
% renew edges
edges = connectboundaries(seg, edges, D, V);

% compute pvg
[delta, deltagl,signdelta,signdeltagl, edges] = computepvg_2(seg, edges, D, V, sizepvg);

% visualize pvg
[lrpvg,lrpvgColor,glpvg,glpvgColor, maxveloc] = drawall(delta, deltagl,signdelta,signdeltagl);

% get final representation
contours = getcontours(seg, edges);

warning('on','all');
end

function [lrpvg,lrpvgColor,glpvg,glpvgColor,maxveloc] = drawall(delta, deltagl,signdelta,signdeltagl)

% ordinary pvg
delta(delta>(mean(delta(:)) + 2*std(delta(:)))) = ...
    mean(delta(:)) + 2*std(delta(:));
delta = floor(delta);
lrpvg = (delta - min(delta(:)))./(max(delta(:)) - min(delta(:)));
lrpvgColor=pvgColorFc(lrpvg,signdelta);

% maximum speed profile
dorig = deltagl;
dbef = zeros(size(dorig));
dbef(:,2:end) = dorig(:,1:end-1);

dveloc = dorig - dbef;
dveloc = floor(dveloc);
dveloc(:,1) = 0;
posu = unique(dveloc(dveloc>0));
negu = unique(dveloc(dveloc<0));
d = dveloc;
d = d + max(abs(negu));
d(:,1) = 0;
d = (d - min(d(:)))./(max(d(:)) - min(d(:))); 
und = unique(d(:));
map = zeros(length(und),3);
red = linspace(0,1,length(posu));
map(2:length(posu)+1,1) = red(end:-1:1);
blue = linspace(0,1, length(negu));
map(length(posu)+1:end-1,3) =  blue(end:-1:1);

deltagl(deltagl>(mean(deltagl(:)) + 2*std(deltagl(:)))) = ...
    mean(deltagl(:)) + 2*std(deltagl(:)); % remove outliers from "intensity decision"
deltagl = floor(deltagl);
glpvg = (deltagl - min(deltagl(:)))./ (max(deltagl(:)) - min(deltagl(:)));
glpvgColor=pvgColorFc(glpvg,signdeltagl);

maxveloc = ind2rgb(dveloc, map);
% h = figure; % ('visible','off'); 
% imshow(glpvg); hold on;
% h2 = imagesc(ind2rgb(dveloc,map)); alpha(h2, .3);
% saveas(h, ['pvg_check_speed' sequence(1:end-4)],'jpeg')


% saveas(hf, ['pvg_' sequence],'jpeg')
end

function [delta, deltagl,signdelta,signdeltagl, edges] = computepvg_2(seg, edges, D, V, sizepvg)
% compute pvg image from delta & signdelta
% update edges
delta = zeros(sizepvg,size(seg,3));
signdelta = zeros(sizepvg, size(seg,3));
deltagl = zeros(sizepvg/2, size(seg,3));

for k = 1 : size(edges,2)
    
    % equidistant sampling
    left = edges(k).left;
    [leftunique, ileft, j] = unique(left(:,1));
    u = linspace(min(left(:,1)), max(left(:,1)), floor(sizepvg/2));
    y = interp1(leftunique, left(ileft,2), u, 'cubic','extrap'); % spline
    left = [u; y]';
    
    right = edges(k).right;
    [rightunique, iright, j] = unique(right(:,1));
    u = linspace(min(right(:,1)), max(right(:,1)), floor(sizepvg/2));
    y = interp1(rightunique, right(iright,2), u, 'cubic', 'extrap'); % spline
    right = [u; y]';
    
    % calculate distances & sign
    denum = (sqrt((D(k,1)-V(k,1))^2+ (D(k,2)-V(k,2))^2));
    numleft = (D(k,1)-V(k,1)).*(V(k,2)-left(:,2)) - (V(k,1)-left(:,1)).*(D(k,2)-V(k,2));
    numright = (D(k,1)-V(k,1)).*(V(k,2)-right(:,2)) - (V(k,1)-right(:,1)).*(D(k,2)-V(k,2));
    
    deltaleft = abs(numleft) ./denum;
    deltaright = abs(numright) ./ denum;
    
    signleft = left(:,1)* (-D(k,2) + V(k,2)) + left(:,2)*(-V(k,1) + D(k,1)) +...
        (D(k,2)*V(k,1) - V(k,2)*D(k,1));
    signleft = sign(-signleft);% positive - good, negative - bad
    
    signright = right(:,1)* (-D(k,2) + V(k,2)) + right(:,2)*(-V(k,1) + D(k,1)) +...
        (D(k,2)*V(k,1) - V(k,2)*D(k,1));
    signright = sign(signright); % positive - good, negative - bad
    
    
    % final form
    delta(:,k) = [flipud(deltaleft) ; deltaright];
    signdelta(:,k) = [flipud(signleft) ; signright];
    
    edges(k).left = left;
    edges(k).right = right;
    
    % glottal delta
    deltagl(:,k) = sqrt( (left(:,1) - right(:,1)).^2 + (left(:,2) - right(:,2)).^2 );
    signdeltagl(:,k)=sign(left(:,1) - right(:,1));
    
end
clear deltaleft deltaright denum ileft iright i j k left right leftunique rightunique
clear signleft signright u y 
end



function contours = getcontours(seg, edges)
% create matrix contours, same size as seg, logical indexing
% to be used for visual inspection

ee = zeros(size(seg));
siz = [size(seg,1) size(seg,2)];

for k = 1 : size(seg,3)
    try
        [x, y] = find(seg(:,:,k)==true);
        ind = sub2ind(siz, [floor(edges(k).left(:,1)) ; x] , [floor(edges(k).left(:,2)) ; y]);
        
        e = ee(:,:,k);
        e(ind) = 1;
        ind = sub2ind(siz, floor(edges(k).right(:,1)), floor(edges(k).right(:,2)));
        e(ind) = 1;
        ee(:,:,k) = e;
    catch
        k
    end
end

contours = logical(ee);
end

function edges = connectboundaries(seg, edges, D, V)
% connect vocal contours with dorsal and ventral points in each image

for k = 1 : size(seg,3)
    
    Ri = edges(k).Ri;
    left = edges(k).left;
    right = edges(k).right;
    Iv = edges(k).Iv;
    Id = edges(k).Id;
    
    dots = [D(k,:) ; Id ; Iv ; V(k,:)];
    dots(sum(isnan(dots),2)==1,:) = [];
    
    for m = 1 : 2: size(dots,1)-1
        coords = dots(m:m+1,:);
        if pdist(coords)>4
            [myline,mycoords,outmat,X,Y] = bresenham(seg(:,:,k), circshift(coords,[0 -1]),0);
            right = [right; [X;Y]'];
            left = [left; [X;Y]'];
        end
    end
    
    % sorted coordinates
    
    [bb, idx] = sort(left(:,1),'ascend');
    left(:,1) = left(idx,1);
    left(:,2) = left(idx,2);
    [bb, idx] = sort(right(:,1),'ascend');
    right(:,1) = right(idx,1);
    right(:,2) = right(idx,2);
    
    edges(k).Ri = Ri;
    edges(k).left = left;
    edges(k).right = right;
    edges(k).Iv = Iv;
    edges(k).Id = Id;
    
end


end

function [D, V] = linkendings(edges, To)
% coordinates [y-image, x-image]
% careful

D = zeros(size(edges,2), 2);
V = zeros(size(edges,2), 2);

% get the open states' endings
Dd = [edges(To).Id];
Dd = [Dd(1:2:end)'  Dd(2:2:end)'];
D(To,:) = Dd;

Vv = [edges(To).Iv];
Vv = [Vv(1:2:end)'   Vv(2:2:end)'];
V(To,:) = Vv;
clear Vv Dd

D = [D(To(1),:) ; D ; D(To(end),:) ];
D(end-1,:) = D(end,:);
V = [V(To(1),:) ; V ; V(To(end),:) ];
V(end-1,:) = V(end,:);
To = [1 To+1 size(edges,2)+1];

% interpolate in between
for k = 1 : length(To)-1
    ti = To(k)+1: To(k+1)-1;
    D(ti,:) = repmat(D(To(k),:), length(ti),1)...
        + repmat( ((D(To(k+1),:) - D(To(k),:))./(To(k+1) - To(k))), length(ti),1)...
        .* [(ti - To(k))' (ti - To(k))'];
    V(ti,:) = repmat(V(To(k),:), length(ti),1)...
        + repmat( ((V(To(k+1),:) - V(To(k),:))./(To(k+1) - To(k))), length(ti),1)...
        .* [(ti - To(k))' (ti - To(k))'];
end

D(1,:) = [];
D(end,:) = [];
V(1,:) = [];
V(end,:) = [];

end

function [edges , To] = getedges(seg, To)

% have everything in edges struct
p = cell(1, size(seg,3));
edges = struct('Ri', p, 'left', p, 'right', p, 'Iv', p, 'Id', p);

for k = 1 : size(seg,3)
    % get objects
    [boundaries, labels, numReg, a] = bwboundaries(seg(:,:,k), 'noholes');
    if numReg>0
        Ri = struct('a',cell(1,numReg), 'b', cell(1,numReg),...
            'I1',cell(1,numReg), 'I2',cell(1,numReg));
        left = cell(numReg,1);
        right = cell(numReg,1);
        % linear regression lines
        for l = 1: numReg
            obj = labels;
            obj(obj~=l) = 0;
            obj(obj~=0) = 1;
            if sum(obj(:))> 10
                % find regression line and sorted paths of each area object
                [Ri(l), left{l}, right{l}] = getlines(obj, boundaries{l});
            end
        end
        left = cell2mat(left);
        right = cell2mat(right);
        % what happens in between??
        if numReg>1
            [left, right] = getinbetween(seg(:,:,k), Ri, left, right);
        end
        
        % sorted coordinates
        if ~isempty(left)
            [bb, idx] = sort(left(:,1),'ascend');
            left(:,1) = left(idx,1);
            left(:,2) = left(idx,2);
            [bb, idx] = sort(right(:,1),'ascend');
            right(:,1) = right(idx,1);
            right(:,2) = right(idx,2);
            
            % calculate dorsal ventral points of the glottal area
            %             [x,y] = find(labels>0);
            %             minx = min(x);
            %             maxx = max(x);
            %             Id = [minx floor(mean(y(x==minx)))]; % dorsal - anterior part
            %
            %             Iv = [maxx floor(mean(y(x==maxx)))]; % ventral - posterior part
            
            Id = [Ri.I1];
            Id = [Id(1:2:end)' Id(2:2:end)'];
            [i,j] = find(Id==min(Id(:,1)));
            Id = Id(i,:); % dorsal
            
            Iv = [Ri.I2];
            Iv = [Iv(1:2:end)' Iv(2:2:end)'];
            [i,j] = find(Iv==max(Iv(:,1)));
            Iv = Iv(i,:); % ventral
            
            edges(k).Ri = Ri;
            edges(k).left = left;
            edges(k).right = right;
            edges(k).Iv = Iv;
            edges(k).Id = Id;
        end
        
    else  % if no contour detected
        Ri = [];
        left = [];
        right = [];
        Iv = [];
        Id = [];
    end
    % store everything
    
    if isempty(edges(k).Id) && sum(k==To)>0
        To(To==k) = [];
    end
    
end

end

function [left, right] = getinbetween(seg, Ri, left, right)

c = [[Ri.I2] [Ri.I1]];
if sum(isnan(c))==0 && isempty(c)==0
    coords = zeros(length(c)/2,2);
    coords(:,1) = c(1:2:end);
    coords(:,2) = c(2:2:end);
    [coords(:,1), idx] = sort(coords(:,1),'ascend'); % sorted
    coords(:,2) = coords(idx,2);
    coords([1 end],:) = [];
    for m = 1 : 2: size(coords,1)-1
        [myline,mycoords,outmat,X,Y] = bresenham(seg, circshift(coords(m:m+1,:),[0 -1]), 0);
        right = [right; [X;Y]'];
        left = [left; [X;Y]'];
    end
end
end

function [Ri, left, right] = getlines(obj, boundaries)
% calculate regression line for each area object
% a = coefficient
% b = intercept
% I1 = most dorsal point of object
% I2 = most ventral point of object
% keep notation by loscheller

Ri = struct('a',[], 'b',[], 'I1',[], 'I2',[]);

% find coordinates of object's pixels
[x,y] = find(obj==1);

% find intercept & coefficient
p = polyfit(x, y, 1);
Ri.a = p(1);
Ri.b = p(2);
f = polyval(p, x);
Ri.I1 = [unique(f(x==min(x))) min(x)];
Ri.I2 = [unique(f(x==max(x))) max(x)];

% calculate sorted paths

spath = boundaries(:,1).*(-Ri.I2(1) + Ri.I1(1)) + boundaries(:,2).*(Ri.I2(2) - Ri.I1(2))...
    + (Ri.I2(1)*Ri.I1(2) - Ri.I1(1)*Ri.I2(2));
spath(spath>=0) = 1;
spath(spath<0) = -1;

right = boundaries(spath~=1,:);
left = boundaries(spath==1,:);

% keep the same notation -- row, column of coordinate - treat as matrix
Ri.I2 = circshift(Ri.I2,[0 -1]);
Ri.I1 = circshift(Ri.I1,[0 -1]);
end

function To = getopenstates(seg)
% find open states in sequences, aka frames where glottal area is maximal

% get glottal cycles
s = sum(sum(seg,2));

% get frames where glottal area maximal
[peaksmax,To]=lmax(s(:),3);%solo para manual no pongo el 3
end

function [myline,mycoords,outmat,X,Y] = bresenham(mymat,mycoordinates,dispFlag)
% use of bresenham algorithm for line drawing, more accurate visualization
% BRESENHAM: Generate a line profile of a 2d image
%            using Bresenham's algorithm
% [myline,mycoords] = bresenham(mymat,mycoordinates,dispFlag)
%
% - For a demo purpose, try >> bresenham();
%
% - mymat is an input image matrix.
%
% - mycoordinates is coordinate of the form: [x1, y1; x2, y2]
%   which can be obtained from ginput function
%
% - dispFlag will show the image with a line if it is 1
%
% - myline is the output line
%
% - mycoords is the same as mycoordinates if provided.
%            if not it will be the output from ginput()
% Author: N. Chattrapiban
%
% Ref: nprotech: Chackrit Sangkaew; Citec
% Ref: http://en.wikipedia.org/wiki/Bresenham's_line_algorithm
%
% See also: tut_line_algorithm

if nargin < 1, % for demo purpose
    pxl = 20;
    mymat = 1:pxl^2;
    mymat = reshape(mymat,pxl,pxl);
    disp('This is a demo.')
end

if nargin < 2, % if no coordinate provided
    imagesc(mymat); axis image;
    disp('Click two points on the image.')
    %[mycoordinates(1:2),mycoordinates(3:4)] = ginput(2);
    mycoordinates = ginput(2);
end

if nargin < 3, dispFlag = 1; end

outmat = mymat;
mycoords = mycoordinates;

x = round(mycoords(:,1));
y = round(mycoords(:,2));
steep = (abs(y(2)-y(1)) > abs(x(2)-x(1)));

if steep, [x,y] = swap(x,y); end

if x(1)>x(2),
    [x(1),x(2)] = swap(x(1),x(2));
    [y(1),y(2)] = swap(y(1),y(2));
end

delx = x(2)-x(1);
dely = abs(y(2)-y(1));
error = 0;
x_n = x(1);
y_n = y(1);
if y(1) < y(2), ystep = 1; else ystep = -1; end
for n = 1:delx+1
    if steep,
        myline(n) = mymat(x_n,y_n);
        outmat(x_n,y_n) = 0;
        X(n) = x_n;
        Y(n) = y_n;
    else
        myline(n) = mymat(y_n,x_n);
        outmat(y_n,x_n) = 0;
        X(n) = y_n;
        Y(n) = x_n;
    end
    x_n = x_n + 1;
    error = error + dely;
    if bitshift(abs(error),1) >= delx, % same as -> if 2*error >= delx,
        y_n = y_n + ystep;
        error = error - delx;
    end
end
% -> a(y,x)
if dispFlag, imagesc(outmat); end
%plot(1:delx,myline)
end
function [q,r] = swap(s,t)
% function SWAP
q = t; r = s;
end


function [pvgColor]=pvgColorFc(pvg,signdelta)

   pvgColor_H=zeros(size(pvg));
   pvgColor_V=abs(pvg);
   pvgColor_S=abs(pvg);
   %pvgColor_S=ones(size(pvg));
   ind_blue=find(signdelta<0);
   ind_red=find(signdelta>0);
   pvgColor_H(ind_blue)=0.666;
   pvgColor_H(ind_red)=0;
   pvgColorHSV(:,:,1)=pvgColor_H;
   pvgColorHSV(:,:,2)=pvgColor_V;
   pvgColorHSV(:,:,3)=pvgColor_S;
   pvgColor=hsv2rgb(pvgColorHSV);
   for j=1:size(pvgColor,1)
       for i=1:size(pvgColor,2)
           if (pvgColor(j,i,1)==1 && pvgColor(j,i,2)==1 && pvgColor(j,i,3)==1)
               pvgColor(j,i,1)=0;
               pvgColor(j,i,2)=0;
               pvgColor(j,i,3)=0;
           end
       end
   end
end