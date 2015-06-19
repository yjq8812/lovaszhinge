function COCOExperimentsLastInputLoss(fn,t)
%addpath([cd '/VOCcode']);

%[Xtrain,Ytrain] = genData(cls,'train')
seed1 = fn;
[X1,Y1] = generateCOCO(50,'train',seed1);
[X2,Y2] = generateCOCO(40,'val',seed1);

Xtraintest = [X1;X2];
Ytraintest = [Y1;Y2];
seed2 = t;
rng(seed2);
order = randperm(length(Xtraintest));
Xtraintest = Xtraintest(order);
Ytraintest = Ytraintest(order);
Xtrainval   = Xtraintest(1:ceil(2*length(order)./3));
Ytrainval   = Ytraintest(1:ceil(2*length(order)./3));
Xtest  = Xtraintest(ceil(2*length(order)./3)+1:end);
Ytest  = Ytraintest(ceil(2*length(order)./3)+1:end);

oursubmodularloss = customLossFunctionCOCO(fn);
hammingloss = @(x)(length(find(x~=0)));

% choose best C : train on train; test on val.
Cs = power(2,[-3:3]);
crosstime = 3;

err_lovasz = zeros(length(Cs),1);
err_hinge = zeros(length(Cs),1);
err_margin = zeros(length(Cs),1);
err_slack = zeros(length(Cs),1);

for k=1:crosstime
    fprintf(['******** Crosstime ' num2str(k) ' ... *******\n']);
    order = randperm(length(Xtrainval));
    Xtrainval = Xtrainval(order);
    Ytrainval = Ytrainval(order);
    
    Xtrain = Xtrainval(1:ceil(length(Xtrainval)/2));
    Ytrain = Ytrainval(1:ceil(length(Ytrainval)/2));
    Xval = Xtrainval(ceil(length(Xtrainval)/2)+1:end);
    Yval = Ytrainval(ceil(length(Xtrainval)/2)+1:end);
    
    err_l = zeros(length(Cs),1);
    err_h = zeros(length(Cs),1);
    err_m = zeros(length(Cs),1);
    err_s = zeros(length(Cs),1);
    for i=1:length(Cs)
        C = Cs(i);
        
        fprintf(['** Training on ',num2str(C),'  **\n']);
        fprintf(['   lovasz hinge learning submodular...    \n']);
        w.lovasz = trainLovasz(Xtrain,Ytrain,oursubmodularloss,C);
        fprintf(['   lovasz hinge learning 0-1 loss...     \n']);
        w.hinge = trainLovasz(Xtrain,Ytrain,hammingloss,C);
        fprintf(['   margin rescaling leargning submodular...    \n']);
        w.margin = trainMarginRescaling(Xtrain,Ytrain,oursubmodularloss,C);
        fprintf(['   slack rescaling learning submodular...    \n']);
        w.slack = trainSlackRescaling(Xtrain,Ytrain,oursubmodularloss,C);
        fprintf(['**  Validation...    \n']);
        err_l(i) = testEval(Xval,Yval,w.lovasz,oursubmodularloss);
        err_h(i) = testEval(Xval,Yval,w.hinge, hammingloss);
        err_m(i) = testEval(Xval,Yval,w.margin, oursubmodularloss);
        err_s(i) = testEval(Xval,Yval,w.slack, oursubmodularloss);
    end
    
    err_lovasz = err_l + err_lovasz;
    err_hinge  = err_h + err_hinge;
    err_margin = err_m + err_margin;
    err_slack  = err_s + err_slack;
end

err_lovasz = err_lovasz./crosstime;
err_hinge  = err_hinge./crosstime;
err_margin = err_m./crosstime;
err_slack  = err_s./crosstime;

ind = find(err_lovasz==min(err_lovasz),1,'first');
Cbest.lovasz = Cs(ind);
ind = find(err_hinge==min(err_hinge),1,'first');
Cbest.hinge = Cs(ind);
ind = find(err_margin==min(err_margin),1,'first');
Cbest.margin = Cs(ind);
ind = find(err_slack==min(err_slack),1,'first');
Cbest.slack = Cs(ind);

% re-train on trainval;

% [Xtrainval,Ytrainval] = genData(cls,'trainval');
 fprintf(['** Re-Training on training/validation set  **\n']);
[w.lovasz,gap.lovasz] = trainLovasz(Xtrainval,Ytrainval,oursubmodularloss,Cbest.lovasz);
[w.hinge,gap.hinge] = trainLovasz(Xtrainval,Ytrainval,hammingloss,Cbest.hinge);
[w.margin,gap.margin] = trainMarginRescaling(Xtrainval,Ytrainval,oursubmodularloss,Cbest.margin);
[w.slack,gap.slack] = trainSlackRescaling(Xtrainval,Ytrainval,oursubmodularloss,Cbest.slack);

% test on test
% [Xtest,Ytest] = genData(cls,'test');
% [Xtest,Ytest] = generateCOCO(50,'val');
fprintf(['**  Testing...    \n']);
err = zeros(4,2);
err(1,1) = testEval(Xtest,Ytest,w.lovasz,oursubmodularloss);
err(2,1) = testEval(Xtest,Ytest,w.hinge, oursubmodularloss);
err(3,1) = testEval(Xtest,Ytest,w.margin, oursubmodularloss);
err(4,1) = testEval(Xtest,Ytest,w.slack, oursubmodularloss);

err(1,2) = testEval(Xtest,Ytest,w.lovasz,hammingloss);
err(2,2) = testEval(Xtest,Ytest,w.hinge, hammingloss);
err(3,2) = testEval(Xtest,Ytest,w.margin, hammingloss);
err(4,2) = testEval(Xtest,Ytest,w.slack, hammingloss);

SaveName = strcat('COCO_Results_Func',num2str(fn),'_',num2str(t));
fid=fopen(strcat(SaveName,'.txt'),'w');
fprintf(fid,'Cbest_lovasz:  %f\n',Cbest.lovasz);
fprintf(fid,'Cbest_hinge:  %f\n',Cbest.hinge);
fprintf(fid,'Cbest_margin:  %f\n',Cbest.margin);
fprintf(fid,'Cbest_slack:  %f\n',Cbest.slack);
fprintf(fid,'Empirical Error Value:  \n');
fprintf(fid,'%f  %f  \n',err');

fclose(fid);
save(strcat(SaveName,'.mat'),...
'err','w','gap','Cbest','oursubmodularloss');
fprintf(['** Done with ' SaveName ' ! **\n']);
end


function [err] = testEval(X,Y,w,lossfn)
   
for i=1:length(X)
    tempw = reshape(w,size(X{i},2),size(X{i},1)); % [d*p,1] to [d,p]
    for j=1:size(X{i},1) % sizeP
        ypred(j,1) = X{i,1}(j,:)*tempw(:,j);
    end
    errList(i) = lossfn(double(sign(ypred)~=Y{i}));
%   before 20150209
%   errList(i) =lossfn(double(sign(X{i}*w)~=Y{i}));

end

err = mean(errList);

end


function [w,gap] = trainLovasz(X,Y,lossfn,C)

[w,model,iteration]=...
    implement_LearningP({'lovasz';[]},1,X,Y,{lossfn;[]},C,1);
gap=iteration.gap;
end

function [w,gap] = trainSlackRescaling(X,Y,lossfn,C)
[w,model,iteration]=...
    implement_LearningP({'slack';[]},1,X,Y,{lossfn;[]},C,1);
gap=iteration.gap;
end

function [w,gap] = trainMarginRescaling(X,Y,lossfn,C)
[w,model,iteration]=...
    implement_LearningP({'margin';[]},1,X,Y,{lossfn;[]},C,1);
gap=iteration.gap;
end

