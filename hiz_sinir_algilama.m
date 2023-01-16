function varargout = hiz_sinir_algilama(varargin)
% Semih AÞÇI
% 16006117061
% Bilgisayarlý Görü Ödevi
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @hiz_sinir_algilama_OpeningFcn, ...
                   'gui_OutputFcn',  @hiz_sinir_algilama_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end


function hiz_sinir_algilama_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);

function varargout = hiz_sinir_algilama_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

function pushbutton1_Callback(hObject, eventdata, handles)
%birinci butona basýlýnca olacak fonksiyonlar
global y
%kullanýcýdan açýlýr pencerede resim seçmesini saðladýk
[filename, pathname] = uigetfile('*.*', 'D:\Matlab deneme\ödev\SpeedLimitSignClassification-master\images\Stress');
    if isequal(filename,0) || isequal(pathname,0)
       disp('!!Kullanýcý resim seçmedi!!')
    else
       filename=strcat(pathname,filename);
       y=imread(filename);
       axes(handles.axes1)
       imshow(y);
    end
restoredefaultpath;
addpath(genpath(pwd));


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
global y %arayüzün diðer kýsýmlarýnda da bu deðiþken kullanýlacaðý için global olarak atadýk

y=imresize(y,[250 300]);%algýlam iþlemindeki boyutlara göre düzenleme yaptýk
y1=y;
%rgb renk kodlarýný ayrý ayrý matrisler halinde aldýk
r=double(y(:,:,1));
g=double(y(:,:,2));
b=double(y(:,:,3));
%gamma correction için iþlemler 
gam = 1.4
[rm,rn]=size(r);
[gm,gn]=size(g);
[bm,bn]=size(b);
outr=abs((1*r).^gam);
outg=abs((1*g).^gam);
outb=abs((1*b).^gam);
% kýrmýzý deðer için maxs ve min deðerlerini belirledik
maxr = max(outr(:));
minr = min(outr(:));
% yeþil deðer için maxs ve min deðerlerini belirledik
maxg = max(outg(:));
ming = min(outg(:));
% mavi deðer için maxs ve min deðerlerini belirledik
maxb = max(outb(:));
minb = min(outb(:));

for i=1:rm
    for j=1:rn
        outr(i,j) = (255*outr(i,j))/(maxr-minr);
    end
end
for a=1:gm
    for b=1:gn
        outg(a,b) = (255*outg(a,b))/(maxg-ming);
    end
end

for c=1:bm
    for d=1:bn
        outb(c,d) = (255*outb(c,d))/(maxb-minb);
    end
end

outr = uint8(outr);
outg = uint8(outg);
outb = uint8(outb);
%oluþan sonuçlarý tek bir imgede birleþtirdik
y=cat(3,outr,outg,outb);
axes(handles.axes2)
imshow(y)
title('Gamma düzeltmesi(gri ton)')
s1=size(y);
src_img=y;
if(numel(s1) > 2)
    y=rgb2gray(y);% imgeyi gri tonlamaya dönüþtürdük
end    
axes(handles.axes3)
imshow(y)
title('Gri resim')
axes(handles.axes4)
y=histeq(y);%histogram eþitleme yapýldý
imshow(y)
title('Histogram eþitleme ');

%% kýrmýzý renk MSER bulundu
R=double(src_img(:,:,1));
G=double(src_img(:,:,2));
B=double(src_img(:,:,3));
R=medfilt2(R,[5,5]);
G=medfilt2(G,[5,5]);
B=medfilt2(B,[5,5]);

ohmRB=max(R./(R+G+B),B./(R+G+B));

axes(handles.axes5)
imshow(uint8(ohmRB),[])
title('Kýrmýzý Renk Bölgesi tespiti')

I = uint8(mat2gray(ohmRB)) ;  

%% HOG için baðlantýlý bileþen analizi 
f=im2bw(ohmRB);
f=medfilt2(f);%MSER sonucunda çýkan resimdeki gürültüyü azaltýldý
axes(handles.axes6)
imshow(f)
title('Median filtreleme')   

connComp = bwconncomp(f); % Baðlý bileþenler bulundu 
stats = regionprops(connComp,'Area','Eccentricity','Solidity');%alan ve saðlamlýk bulundu
disp(stats)
%% dairesel alan algýlama
clear s
s=regionprops(f,{'Area';'EquivDiameter';'BoundingBox';'Eccentricity'})
[v ind]=max([s.Area]);
D=s(ind).EquivDiameter;
% 
A=pi.*D.^2.0/4.0;
% 
Diff=abs(A-s(ind).Area)
zk=imcrop(y1,s(ind).BoundingBox);
axes(handles.axes7)
imshow(zk)
title('Bulunan levha');
%bulunan alanýn içini boþaltma
s(ind).Eccentricity

zk1=imcrop(f,s(ind).BoundingBox);
yk=imfill((zk1),'holes');
axes(handles.axes8)
imshow(yk)
title('Tabelanýn içinin boþaltýlmýþ hali')


%% Çerçeve içine alma
clear s
Ibw1=yk;%önceki kýsýmda içi boþaltýlmýþ dairesel yapýyý aktardýk
s1  = regionprops(Ibw1,'MajorAxisLength','MinorAxisLength','Area','centroid');


ind=find([s1.Area]==max([s1.Area]));
centroids = cat(1, s1.Centroid);
Router=s1.MajorAxisLength./2.0;
Rinner=s1.MinorAxisLength./2.0;

[B,L] = bwboundaries(Ibw1,'noholes');
axes(handles.axes9)
imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on
for k = 1:length(B)
    boundary = B{k};
    plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
    %içi boþ dairesel bölgeyi plot komutu ile çerçeve içine alýyoruz
end



%% Þekil ayýrt etme
R=Router;
%oluþan Xc ve Yc'nin merkez noktalarýný belirledik
Xc=centroids(ind,1);
Yc=centroids(ind,2);
[m1,n1]=size(Ibw1);


Xi=boundary(:,2);%noktalar arasýnda sýnýr vektörü oluþturur
Yi=boundary(:,1);

eMIC= max(sqrt((Xi-Xc).^2 + (Yi-Yc).^2)-R);%enterpolasyon yapýldý
%tahmin yapmak için enterpolasyon yapýldý eðer enterpolasyon 6nýn
%altýndaysa þekil dairesel dýnýflandýrmaya giriyor fazla ise köþeli
%sýnýflandýrýlýyor
fprintf('eMIC Value is --> %3.2f\n',eMIC);

if(eMIC<6.0)
   
    fprintf('Çember bulundu\n');
    
    
    yk1=(zk(:,:,1)+zk(:,:,2)+zk(:,:,3))./3;
    
    yk1(yk1<=20)=0;
    yk1(yk1>20)=255;
    
    
    yk1=im2bw(rgb2gray(zk));
   axes(handles.axes10)
    imshow(yk1)
    yk1=imresize(yk1,[128 128]);
    
    axes(handles.axes10)
    imshow(yk1)
    fprintf('Çember bulundu\n');
    title('Çember bulundu')
    idk=1;
    
else
    
    idk=0;
    
%     figure,
    imshow(zk)
    fprintf('Üçgen Bulundu\n');
    title('Üçgen Bulundu ')
      
    
end


ik1=0;
if(idk==1)%eðer þekil dairesel ise eðittiðimiz veriyi görselde kullanarak bulma iþlemi yapacak
    
%% modelin kullanýmý ve sonuç elde etme

cellSize = 9 ;%SVM class sayýsý
hog=HOG(im2single(yk1));
load Train1.mat %9 adet SVM class yapýsýna sahip eðitilmiþ veri
%numel() içerisindeki vektörün kaç elemandan oluþtuðunu sayar
for ik=1:numel(H)
P(:,ik)=double(H{ik}(:));
end

PP=double(hog(:));
T1=cell2mat(T);

trainSet=double(P).*0.5;

trainClass=T1;
 testSet=double(PP).*0.5;
 testClass=1;



[model,OtherOutput1]=classificationTrain(trainSet,trainClass,'lsvm');    
[result21,OtherOutput1]=classificationPredict(model,trainSet,trainClass);
[result2,OtherOutput]=classificationPredict(model,testSet,1);

%SVM'deki sýnýflarýn çýktýlara uyarlanmasý 
if(result2==1)
    set(handles.text2,'String','Hýz sýnýrý----> 20 Km/h');
elseif(result2==2)
    set(handles.text2,'String','Hýz sýnýrý----> 30 Km/h');
elseif(result2==3)
    set(handles.text2,'String','Hýz sýnýrý----> 40 Km/h');
elseif(result2==4)
    set(handles.text2,'String','Hýz sýnýrý----> 50 Km/h');
elseif(result2==5)
    set(handles.text2,'String','Hýz sýnýrý----> 60 Km/h');
elseif(result2==6)
    set(handles.text2,'String','Hýz sýnýrý----> 70 Km/h');
elseif(result2==7)
    set(handles.text2,'String','Hýz sýnýrý----> 80 Km/h');
elseif(result2==8)
    set(handles.text2,'String','Hýz sýnýrý----> 90 Km/h');
elseif(result2==9)
    set(handles.text2,'String','Hýz sýnýrý----> 100 Km/h');   
else
    set(handles.text2,'String','Diðer');
    
end   
    ik1=1;
    
end
