%tca_plus the extended version of transfer component analysis
%   src: source project with labels
%   tar: target project with labels
%Implemented according to Nam et al. 2013. Transfer defect learning.
function [src,tar] = tca_plus(src,tar)   
    data_src = src(:,1:end-1);
    data_tar = tar(:,1:end-1);
    
    % normalization
    [data_src,data_tar] = normal(data_src,data_tar,'heuristic');

    % transfering
    [data_src,data_tar] = tca(data_src,data_tar);
    
    src(:,1:end-1) = data_src;
    tar(:,1:end-1) = data_tar;
end