clc
clear all
close all


load anal_face_form_test

%%
P2(60,60,6720)=0;
for ii=1:24;
tic
for jj=1:280


f0=P_test(:,:,(ii-1)*280+jj);
f0=imresize(f0,[60,60]);
f0=im2double(f0);
P2(:,:,(ii-1)*280+jj)=f0;



end
disp(ii)
toc
end

save anal_face_form_test2 P2 S_test