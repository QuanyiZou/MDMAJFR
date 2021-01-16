function [Y_tar_predition] = DMDAJFR(X_src,Y_src,X_tar,Y_tar_pseudo,parameter)

%%% parameter.lambda;
 %parameter.beta;
 %parameter.noises;
 %parameter.layers;

  [X_src_globalx,X_tar_globalx,W] =GMDA(X_src,Y_src,X_tar,Y_tar_pseudo,parameter);
  [X_src_localx,X_tar_localx,Wc] =GMDA(X_src,Y_src,X_tar,Y_tar_pseudo,parameter);
  %X_src_new,: source joint feature representations;
   %X_tar_new : target joint feature representations,
  X_src_new=[X_src_globalx,X_src_localx]; 
  X_tar_new=[X_tar_globalx,X_tar_localx];
  [Y_tar_predition]= Pseudolable(X_src_new,  X_tar_new, Y_src);
end

