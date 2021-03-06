function err =  example()
% An example function for using Lovasz Hinge convex surrogate for
% submodular losses
% --
% Implemented by 
% Jiaqian Yu & Matthew B.Blashcko @ 2015
% 
% Please cite: 
% Yu, J. and M. B. Blaschko: Learning Submodular Losses with the Lovász Hinge.
%                   International Conference on Machine Learning (ICML), 2015.

% -----------------------------------------------------------------%
%                          Generate Data                           %
% -----------------------------------------------------------------%
[X, Y] = genData();
Xtrain = X(1:ceil(length(X)./3));
Ytrain = Y(1:ceil(length(X)./3));
Xtest  = X(ceil(2*length(X)./3)+1:end);
Ytest  = Y(ceil(2*length(X)./3)+1:end);
% -----------------------------------------------------------------%
%                          Set parameters                          %
% -----------------------------------------------------------------%

[oursubmodularloss,subIsIn] = customLossFunctionCOCO(1);% user-input (submodular) function

C = 1;% regularizer
      % cross-validation for choosing the C values need to be implented   

% -----------------------------------------------------------------%
%                             Training                             %
% -----------------------------------------------------------------%
% [Xtrainval,Ytrainval] = genData(cls,'trainval');
fprintf(['** Training...  **\n']);
[w,gap] = trainLovasz(Xtrain,Ytrain,oursubmodularloss,subIsIn,C);

% -----------------------------------------------------------------%
%                             Testing                              %
% -----------------------------------------------------------------%
fprintf(['**  Testing...  **\n']);
err = testEval(Xtest,Ytest,w,oursubmodularloss);
fprintf(['** Done  ! **\n']);
end

function [X,Y] = genData()
% Generate/Load your data here
% X : patterns, a cell in size of n*1; each cell in size of p*d
% Y : labels, a cell in size of n*1; each cell in size of p*1
% n : number of patterns
% p : size of bags
% d : dimension of feature vectors

% load one example generated from COCO dataset
load('COCOexample.mat','X','Y')
end


function [err] = testEval(X,Y,w,lossfn)
% Test time: calculate the empirical error values
for i=1:length(X)
    tempw = reshape(w,size(X{i},2),size(X{i},1)); % [d*p,1] to [d,p]
    for j=1:size(X{i},1) % sizeP
        ypred(j,1) = X{i,1}(j,:)*tempw(:,j);
    end
    errList(i) = lossfn(double(sign(ypred)~=Y{i}));
end

err = mean(errList);

end

function [w,gap] = trainLovasz(X,Y,lossfn,subIsIn,C)
% a wapper function for a method learning weight vector w;
% gap: the primal-dual gap during learning iteration;
% using the Lovasz Hinge as the convex surrogate operator
[w,model,iteration]=...
    implementLearning(X,Y,lossfn,'lovasz',C,subIsIn);
gap=iteration.gap;
end

function [w,gap] = trainSlackRescaling(X,Y,lossfn,C)
% a wapper function for a method learning weight vector w;
% gap: the primal-dual gap during learning iteration;
% using slack rescaling as the convex surrogate operator
[w,model,iteration]=...
    implementLearning(X,Y,lossfn,'slack',C,1);
gap=iteration.gap;
end

function [w,gap] = trainMarginRescaling(X,Y,lossfn,C)
% a wapper function for a method learning weight vector w;
% gap: the primal-dual gap during learning iteration;
% using margin rescaling as the convex surrogate operator
[w,model,iteration]=...
    implementLearning(X,Y,lossfn,'margin',C,1);
gap=iteration.gap;
end

