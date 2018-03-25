clc
clear all
close all
tic

[model msz] = load_model();
load face_generate_id
ii=4;
alpha = face_id_shape(:,ii);
beta  = face_id_tex(:,ii);

shape  = coef2object( alpha, model.shapeMU, model.shapePC, model.shapeEV );
shape = reshape(shape, [ 3 prod(size(shape))/3 ])';
c=10;
c=c*pi/180;
x0=0;
z0=0;
shape_new(:,2) = (shape(:,2) - x0)*cos(c) - (shape(:,3) - z0)*sin(c) + x0;
shape_new(:,1) = shape(:,1);
shape_new(:,3) = (shape(:,2) - x0)*sin(c) + (shape(:,3) - z0)*cos(c) + z0;

plot3(shape_new(:,1),shape_new(:,2),shape_new(:,3),'.')
hold on 
plot3(0,0,z0,'*')
xlabel('x')
ylabel('y')
zlabel('z')
axis tight 
axis([-150000 150000 -150000 150000 -150000 150000])
%%
xa=min(shape_new(:,1));
xb=max(shape_new(:,1));
yb=max(shape_new(:,2));
ya=yb-(xb-xa);
d=floor((xb-xa));
yb=yb-0.1*d;
ya=ya-0.1*d;
d=d/70;
x1=xa;
y1=yb;
f(70,70)=0;

for ii=1:70;
x1=xa;
for jj=1:70;
xx=shape_new(:,1);
yy=shape_new(:,2);
num=find(xx>=x1 & xx<x1+d-1 & yy>=y1-d+1 & yy<y1);
zz=shape_new(num,3);
if size(zz,1)==0;
%     if ii>20 & ii<60 & jj>20 & jj<60 
%     f(ii,jj)=f(ii,jj-1);
%     else
    f(ii,jj)=0;
%     end
elseif size(zz,1)>0
f(ii,jj)=max(zz);
end
x1=x1+d;
end

y1=y1-d;
end

figure(2)
imshow(f,[])








toc