clc
clear all
close all
tic
load depth_dataset1

for ii=1:224
for jj=1:6
for kk=1:8
f=p{ii}{jj,kk};
z_min=min(f(:));
z_max=max(f(:));
d=z_max-z_min;
background=z_min-2*rand(1)*d;
f(f==0)=background;
f=f-min(f(:));
f=f/max(f(:));
g1=ceil(11*rand(1));
g2=70-(10-g1+1);

% f=f(11:70,g1:g2);

if jj==4||jj==5;
f=f(5:64,g1:g2);
else
f=f(11:70,g1:g2);
end

P(:,:,(ii-1)*48+(jj-1)*8+kk)=f;
S(ii,(ii-1)*48+(jj-1)*8+kk)=1;
imshow(f)
pause(0.001)

end
end
disp(ii)
end

save depth_dataset_doubel_1 P S










toc