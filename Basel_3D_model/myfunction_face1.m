clc
clear all
close all

[model msz] = load_model();
load D:/FDU/小罗/3DMM/dataset/s/04_attributes.mat

alpha = randn(msz.n_shape_dim, 1);
beta  = randn(msz.n_tex_dim, 1);
% alpha(:) =0; alpha(2)=8;
% beta(:) =0; beta(3)=50;
%%

% shape  = coef2object( alpha, model.shapeMU, model.shapePC, model.shapeEV );
% tex    = coef2object( beta,  model.texMU,   model.texPC,   model.texEV );

shape  = coef2object( alpha+0.1*age_shape(1:msz.n_shape_dim)+0.01*gender_shape(1:msz.n_shape_dim), model.shapeMU, model.shapePC, model.shapeEV );
tex    = coef2object( beta+0.1*age_tex(1:msz.n_shape_dim)+0.01*gender_tex(1:msz.n_shape_dim),  model.texMU,   model.texPC,   model.texEV );

% shape = coef2object( randn(msz.n_shape_dim, msz.n_seg), model.shapeMU, model.shapePC, model.shapeEV, model.segMM, model.segMB );
% tex   = coef2object( randn(msz.n_tex_dim,   msz.n_seg), model.texMU,   model.texPC,   model.texEV,   model.segMM, model.segMB );
rp     = defrp;
rp.phi = [0,0];
% rp.dir_light.dir = [0;0;1];
rp.dir_light.dir = [0,0];
% rp.dir_light.intens = 0.6*ones(3,1);
display_face2(shape, tex, model.tl, rp);










