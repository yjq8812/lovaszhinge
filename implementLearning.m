function [w, model,iteration]=implementLearning(patterns,labels,setlossFn,type,C,subIsIn)
% Canonical Classifier
% 
%   PATTERNS: input X
%   LABELS:   output Y
%   TYPE: 
%       the type of constraints formulation - no other type supported yet
%       "lovasz":   lovasz hinge
%       "margin":   margin rescaling
%       "slack":    slack rescaling
%   SETLOSSFN: the set functions that define the loss function
%   C: regularizer value
%   SUBISIN: if the submodular function is increasing, Lovasz Hinge uses
%               threshoding strategy
%   
%   calling:
%	[model,sparm,state] = GeneralClassifier(parm)
% --
% Implemented by 
% Jiaqian Yu & Matthew B.Blashcko @ 2015
% 

% ------------------------------------------------------------------
%                                                 Set SVM parameters
% ------------------------------------------------------------------

parm.patterns = patterns ;
parm.labels = labels ;

parm.C = C;
parm.setlossFn = setlossFn;

% parm.setlossFnSubIn = setlossFn{1}; % g
parm.submodFnIsIncreasing = subIsIn; % 1 if g is increasing (for thresholding of LH)
% parm.setlossFnSupDe = setlossFn{2}; % h, can be empty

parm.formulationType= type; % choose only for g
% parm.formulationTypeSub = type{1}; % choose only for g
% parm.formulationTypeSup = type{2}; % choose only for h, can be empty

parm.findMostViolatedLovasz = @violateLovasz;

parm.isgreedy = 1; % using greedy algorithm for margin&slack-rescaling
parm.findMostViolatedMargin = @violateMarginGreedy;
parm.findMostViolatedSlack = @violateSlackGreedy;


parm.sizeP=length(labels{1,1}); % size of bag / #label
parm.dim=size(patterns{1,1},2); % dimension of feature

parm.psiFn = @featureCB; % feature map function 
parm.sizePsi=parm.dim*parm.sizeP; % dimension of Psi in this multilabe task

[model,parm,state,iteration] = GeneralClassifier(parm);

w = model.w;


end

% --------------------------------------------------------------------
%                                                SVM struct callbacks
% --------------------------------------------------------------------

function phi = featureCB(param, x, y) 
% joint feature function
phi = zeros(1,length(y));
for j=1:length(y) % phi^j = x^j*y^j; x^j = p * dim; y = p * 1; phi = 1 * (dim*p)
    phi((j-1)*param.dim+1:j*param.dim)  = x(j,:).*y(j);    
end

end

function [gamma,deltaPsi] = violateLovasz(param, model, x, y,setfn)
% max w.r.t. permutations of sum_k s_{\pi_k}^i(f(\{\pi_1,...,\pi_k\}) - f(\{\pi_1,...,\pi_{k-1}\})
% where f is submodular loss function and s_k^i is the margin violation of the kth sample in bag i
% s^j = 1 - <w^j, phi^j(x,y)>

w = model.w;
s = zeros(1,param.sizeP);
psi= param.psiFn(param,x,y); 
for i=1:param.sizeP 
s(i) = 1 - psi(1,(i-1)*param.dim+1:i*param.dim)*w((i-1)*param.dim+1:i*param.dim,1); 
end

[~,ind] = sort(s,'descend'); 
gammak = zeros(length(y),1);

for k=1:length(ind)

    l = zeros(length(y),1);
    l(ind(1:k)) = 1;
    l_less = zeros(length(y),1);
    l_less(ind(1:k-1)) = 1;    
    gammak(ind(k)) = setfn(l)-setfn(l_less); 

end
if(param.submodFnIsIncreasing) % use thresholding if the submodular func is increasing
    y(s<=0)=0; % y is in the original order
end
gamma = sum(gammak.*double(y~=0)); % gamma is in the order ind

tempPsi = zeros(param.dim,param.sizeP);
for i=1:param.sizeP
    tempPsi(:,i)=(gammak(i)*y(i))*x(i,:)'; % in order ind
end;
deltaPsi=reshape(tempPsi,param.sizePsi,1); % in order ind

end


function yhat = violateMarginGreedy(param, model, x, y,setfn) 
% Greedy algorithm for finding max violated y
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
w = model.w;

% initialization
constraint_max = -inf;
yhat = y;

for i=1:length(y)
    temp_y = yhat;
    temp_y(i) = temp_y(i)*(-1);
    
    sumDelta = setfn(double(temp_y~=y));
    psi = param.psiFn(param,x,temp_y);
    const = 0;
    for j=1:param.sizeP % sum_j <w^j,phi(x^j,y^j)>
        const = const + psi(1,(j-1)*param.dim+1:j*param.dim)*w((j-1)*param.dim+1:j*param.dim,1);
    end
    constraint_new = sumDelta + const;% argmax_y delta(yi, y) + <psi(x,y), w>
    
    if constraint_new>=constraint_max
        constraint_max = constraint_new;
        yhat = temp_y;
    end
end
end


function yhat = violateSlackGreedy(param, model, x, y,setfn)
% Greedy algorithm selection for finding max violated y
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)

w = model.w;
% initialisation
constraint_max = -inf;
yhat=y; 

for i=1:length(y)
    
    temp_y = yhat;
    temp_y(i) = temp_y(i)*(-1);
    sumDelta = setfn(double(temp_y~=y));
    
    const = 0;
    psi = param.psiFn(param,x,temp_y);
    for j=1:param.sizeP % sum_j <w^j,phi(x^j,y^j)>
        const = const + psi(1,(j-1)*param.dim+1:j*param.dim)*w((j-1)*param.dim+1:j*param.dim,1);% sum <w^j,phi(x_i^j,y_i^j)>
    end
    constraint_new = sumDelta + sumDelta*const; % argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)  
                                                % == argmax_y delta(yi, y) (1 + <psi(x,y), w>)
    if constraint_new>=constraint_max
        constraint_max = constraint_new;
        yhat = temp_y;
    end
end

end