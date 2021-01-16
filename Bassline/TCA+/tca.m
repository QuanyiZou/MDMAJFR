%tca Transfer Component Analysis
%    X_src: source project
%    X_tar: target project
%    E_src: transferred source project
%    E_tar: transferred target project
% Implemented according to Pan et al. 2011. Domain adaptation via transfer component analysis

function [E_src, E_tar] = tca(X_src, X_tar)
    %% number of files in source and target projects
    n_tar = size(X_tar,1);
    n_src = size(X_src,1);
    dim = size(X_tar,2);

    %% build linear kernel
    X = [X_src; X_tar];
    L = [(1/(n_src*n_src))*ones(n_src, n_src) (-1/(n_src*n_tar))*ones(n_src, n_tar); ...
        (-1/(n_src*n_tar))*ones(n_tar, n_src) (1/(n_tar*n_tar))*ones(n_tar, n_tar)];
    L(isnan(L)) = 0;
    K = X*X';           
    K(isnan(K)) = 0;

    %% solution for TCA
    mu = 1;
    H = eye(n_src+n_tar) - 1/(n_src+n_tar) *ones(n_src+n_tar,1)*ones(n_src+n_tar,1)';
    forPinv = mu*eye(n_src+n_tar) + K*L*K;
    forPinv(isnan(forPinv)) = 0;
    Kc = pinv(forPinv)*K*H*K;
    Kc(isnan(Kc)) = 0;

    %% transferred source and target projects
    [V,D] = eig(Kc);
    eig_values = diag(D);
    [~,index_sorted] = sort(eig_values, 'descend');
    V = V(:, index_sorted);
    E_src = K(1:n_src,:)*V;
    E_tar = K(n_src+1:end,:)*V;
    E_src = E_src(:, 1:dim);
    E_tar = E_tar(:, 1:dim);
end


