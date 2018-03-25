clc
clear all
close all


[model msz] = load_model();
load face_generate_id
load depth_dataset1

angle_x=[15,10,5,0];
angle_x=[-angle_x,angle_x];
angle_x=(angle_x/180)*pi;

angle_y=[10,5,0];
angle_y=[-angle_y,angle_y];
angle_y=(angle_y/180)*pi;

for ii=101:224
tic
alpha = face_id_shape(:,ii);
beta  = face_id_tex(:,ii);

shape  = coef2object( alpha, model.shapeMU, model.shapePC, model.shapeEV );
shape = reshape(shape, [ 3 prod(size(shape))/3 ])';

for jj=1:6;
cy=angle_y(jj);
shape_new=face_shape_trans_y(shape,cy);

for kk=1:8
cx=angle_x(kk);
shape_new2=face_shape_trans(shape_new,cx);
f=face_depth_generate(shape_new2);
p{ii}{jj,kk}=f;
% imshow(f,[])
% pause(0.1)
disp([cy*180/pi,cx*180/pi])

end

end
save depth_dataset1 p
disp(ii)
toc
end




