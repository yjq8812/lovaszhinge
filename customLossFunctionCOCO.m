function [loss,subisin] = customLossFunctionCOCO(fn)
% define your (submodular) loss functions here
% loss: funciton handle
% subisin: whether the function is increasing
switch fn
    case 1
        beta = [1 0.6 0.5 0.4 0.7 0.8];
        loss = @(x)(beta*(x~=0)+1-exp(-length(find(x~=0))));
        subisin = 1;
    case 2
        alpha = 1;
        loss = @(x)(1-exp(-alpha*length(find(x~=0))));
        subisin = 1;
    case 3    
        beta = [1 0.6 0.5 0.4 0.7 0.8];
        lmax = 2.0;
        loss = @(x)(min(lmax,beta*(x~=0)));
        subisin = 1;
    otherwise
        error('\n This function number does not exist!!!\n');
end


end
