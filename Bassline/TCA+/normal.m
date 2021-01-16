%normal normalizing source and target projects
%       src: source project without labels
%       tar: target project without labels
%       flag: type of normalization including: 
%             minmax, zscore, zscore_src, zscore_tar, heuristic
% The heuristic normalization method is implemented according to 
%       Nam et al. 2013. Transfer defect learning.
function [src,tar] = normal(src,tar,flag)  
    if strcmp(flag,'minmax')
        [src,tar] = n1_minmax(src,tar);
    elseif strcmp(flag,'zscore')
        [src,tar] = n2_zscore(src,tar);
    elseif strcmp(flag,'zscore_src')
        [src,tar] = n3_zscore_src(src,tar);
    elseif strcmp(flag,'zscore_tar')
        [src,tar] = n4_zscore_tar(src,tar);
    elseif strcmp(flag,'heuristic')
        v_src = dcv(pdist(src));
        v_tar = dcv(pdist(tar));
        
        st.mean     = ds(v_src.mean,v_tar.mean);
        st.median   = ds(v_src.median,v_tar.median);
        st.min      = ds(v_src.min,v_tar.min);
        st.max      = ds(v_src.max,v_tar.max);
        st.std      = ds(v_src.std,v_tar.std);
        st.num      = ds(v_src.num,v_tar.num);
        
        if st.mean==4 && st.std==4
        elseif (st.num==7 || st.num==1) && ...
                (st.min==7 || st.min==1) && ...
                (st.max==7 || st.max==1)
            [src,tar] = n1_minmax(src,tar);
        elseif (st.std==7 && st.num<4) || ...
                (st.std==1 && st.num>4)
            [src,tar] = n3_zscore_src(src,tar);
        elseif (st.std==7 && st.num==7) || ...
                (st.std==1 && st.num==1)
            [src,tar] = n4_zscore_tar(src,tar);
        else
            [src,tar] = n2_zscore(src,tar);
        end
    end
    src(isnan(src))=0;
    tar(isnan(tar))=0;
end

% Dataset Characteristic Vector (DCV)
function [r] = dcv(d)
    r.mean   = mean(d);
    r.median = median(d);
    r.min    = min(d);
    r.max    = max(d);
    r.std    = std(d);
    r.num    = size(d,1);
end

% Degree of Similarity (DS)
function [r] = ds(s,t)
    if s*1.6<t
        r = 7; % 'much more';
    elseif s*1.3<t && t<=s*1.6
        r = 6; % 'more';
    elseif s*1.1<t && t<=s*1.3
        r = 5; % 'slightly more'
    elseif (s*0.9<t && t<=s*1.1) || s==t
        r = 4; % 'same';
    elseif s*0.7<=t && t<s*0.9
        r = 3; % 'slightly less'
    elseif s*0.4<=t && t<s*0.7
        r = 2; % 'less'
    elseif t<s*0.4
        r = 1; % 'much less'
    end
end

function r = mtx(x,n)
    r = repmat(x,n,1);
end

function [src,tar] = n1_minmax(src,tar)
    m = size(src,1);
    n = size(tar,1);
    src = (src - mtx(min(src),m))./(mtx(max(src),m)-mtx(min(src),m));
    tar = (tar - mtx(min(tar),n))./(mtx(max(tar),n)-mtx(min(tar),n));
end

function [src,tar] = n2_zscore(src,tar)
    m = size(src,1);
    n = size(tar,1);
    src = (src - mtx(mean(src),m))./mtx(std(src),m);
    tar = (tar - mtx(mean(tar),n))./mtx(std(tar),n);
end

function [src,tar] = n3_zscore_src(src,tar)
    m = size(src,1);
    n = size(tar,1);
    src = (src - mtx(mean(src),m))./mtx(std(src),m);
    tar = (tar - mtx(mean(src),n))./mtx(std(src),n);
end

function [src,tar] = n4_zscore_tar(src,tar)
    m = size(src,1);
    n = size(tar,1);
    src = (src - mtx(mean(tar),m))./mtx(std(tar),m);
    tar = (tar - mtx(mean(tar),n))./mtx(std(tar),n);
end