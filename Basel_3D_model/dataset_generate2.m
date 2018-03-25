clc
clear all
close all

[model msz] = load_model();
% face_id_shape=randn(msz.n_shape_dim,1000);
% face_id_tex=randn(msz.n_tex_dim,1000);
% save face_generate_id face_id_shape face_id_tex
load face_generate_id
%%
% angle_x=[pi/6,pi/6-(pi/36)*1,pi/6-(pi/36)*2,pi/6-(pi/36)*3,...
%          pi/6-(pi/36)*4,pi/6-(pi/36)*5,pi/6-(pi/36)*6,...
%          -pi/6,-pi/6+(pi/36)*1,-pi/6+(pi/36)*2,-pi/6+(pi/36)*3,...
%          -pi/6+(pi/36)*4,-pi/6+(pi/36)*5,-pi/6+(pi/36)*6];
angle_x=[15,15,10,10,10,5,5,5,0,0];
angle_x=[-angle_x,angle_x];
angle_x=(angle_x/180)*pi;
% (angle_x*180)/pi

% angle_y=[pi/12,pi/12-(pi/36)*1,pi/12-(pi/36)*2,pi/12-(pi/36)*3,...
%          -pi/12,-pi/12+(pi/36)*1,-pi/12+(pi/36)*2,-pi/12+(pi/36)*3];
angle_y=[10,10,5,5,5,0,0];
angle_y=[-angle_y,angle_y];
angle_y=(angle_y/180)*pi;
% (angle_y*180)/pi

light_angle_x=[0,10,20,30,40,50,60,70,80,90];
light_angle_x=[-light_angle_x,light_angle_x];
light_angle_x=(light_angle_x/180)*pi;
light_angle_y=[0,10,10,10,20,20,20,30,30,40, 0,10,10,10,20,20,20,30,40,50];
% light_angle_y=[-light_angle_y,light_angle_y];
light_angle_y=(light_angle_y/180)*pi;
%%

% P(80,80,112000)=0;
% S(1000,112000)=0;



for ii=225:500;
tic
alpha = face_id_shape(:,ii);
beta  = face_id_tex(:,ii);

shape  = coef2object( alpha, model.shapeMU, model.shapePC, model.shapeEV );
tex    = coef2object( beta,  model.texMU,   model.texPC,   model.texEV );

rp     = defrp;
rp.phi = [0,0];
rp.dir_light.dir = [0,0];
display_face2(shape, tex, model.tl, rp);

for jj=1:14
for kk=1:20

num=ceil(20*rand(1,2));
num2=ceil(3*rand(1));
view(180 + light_angle_x(num(1)) * 180 / pi, light_angle_y(num(2)) * 180 / pi);
L=camlight('headlight');
view(180 + angle_x(kk) * 180 / pi , angle_y(jj) * 180 / pi);
pause(0.1)

% print(1,'-djpeg',['D:\FDU\小罗\3DMM\dataset\s',num2str((ii-1)*112+(jj-1)*14+kk),'.jpg']);
print(1,'-dtiff',['D:\FDU\小罗\3DMM\dataset\s',num2str((ii)),'_',num2str((jj-1)*20+kk),'.tif']);
% print(1,'-dpng',['D:\FDU\小罗\3DMM\dataset\s',num2str((ii-1)*112+(jj-1)*14+kk),'.png']);

f0=imread(['D:\FDU\小罗\3DMM\dataset\s',num2str((ii)),'_',num2str((jj-1)*20+kk),'.tif']);
f0=im2double(f0);
f0=rgb2gray(f0);
[mm,nn]=find(f0~=1);

x_a=min(nn);
x_b=max(nn);
x_d=x_b-x_a;
if (angle_x(kk)*180)/pi<-9;
x_a = x_a + 0.05*x_d;
x_b = x_b + 0.05*x_d;
elseif (angle_x(kk)*180)/pi>9;
x_b = x_b - 0.05*x_d;
x_a = x_a - 0.05*x_d;
elseif (angle_x(kk)*180)/pi>2&&(angle_x(kk)*180)/pi<7;
x_b = x_b - 0.02*x_d;
x_a = x_a - 0.02*x_d;
elseif (angle_x(kk)*180)/pi>-7&&(angle_x(kk)*180)/pi<-2;
x_b = x_b + 0.02*x_d;
x_a = x_a + 0.02*x_d;
else

end

y_a=min(mm)+10+4*rand(1);
y_b=y_a+(x_b-x_a);

f0=f0(y_a:y_b,x_a:x_b);
f0(f0==1)=0.25*rand(1);

if num2==1
f0=imresize(f0,[100,100]);
%  if (angle_x(kk)*180)/pi>9;
%  f0=f0(11:90,1:80);
%  elseif (angle_x(kk)*180)/pi<-9;
%  f0=f0(11:90,21:100);
%  else
  f0=f0(15:94,11:90);
%  end

else
f0=imresize(f0,[90,90]);
f0=f0(8:87,6:85);
end

imwrite(f0,['D:\FDU\小罗\3DMM\dataset\ss',num2str((ii)),'_',num2str((jj-1)*20+kk),'.tif'])
% P(:,:,(ii-1)*112+(jj-1)*14+kk)=f0;
% S(ii,(ii-1)*112+(jj-1)*14+kk)=1;

set(L,'visible','off');

end
end
disp(ii)
toc
end
