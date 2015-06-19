function [model,sparm,state,iteration] = GeneralClassifier(sparm, oldstate)

state = bundler(); % initialize state
state.lambda = 1 ./ (sparm.C);

if (~isfield(sparm,'convergenceThreshold'))
    sparm.convergenceThreshold = 0.008;
end
maxIterations = 1000;

sparm.w = zeros(sparm.sizePsi,1);
state.w = sparm.w;


model.w = state.w;

if (exist('oldstate','var'))
    for i=1:length(oldstate.b)
        if(oldstate.softVariables(i))
            state = bundler(state,oldstate.a(:,i),oldstate.b(i));
        end
    end
end

minIterations = 10;
numIterations = 0;

bestPrimalObjective = Inf;
bestState = state;
iteration.iter = 0;
iteration.gap = [];

while (((bestPrimalObjective - state.dualObjective)/state.dualObjective > sparm.convergenceThreshold ...
        || minIterations>0) && numIterations < maxIterations )
    
    
    numIterations = numIterations + 1;
    minIterations = minIterations - 1;
    
    switch sparm.formulationType
        case 'lovasz'
            [phi, b] = computeOneslackLovasz(sparm, model, sparm.patterns, sparm.labels,sparm.setlossFn);
        case 'margin'
            [phi, b] = computeOneslackMargin(sparm, model, sparm.patterns, sparm.labels,sparm.setlossFn);
        case 'slack'
            [phi, b] = computeOneslackSlack(sparm, model, sparm.patterns, sparm.labels,sparm.setlossFn);
        otherwise
            error('The type has not been well defined!')
    end
    
    if (norm(phi)==0)
        phi= zeros(size(state.w));
        fprintf('\n Warnign: No New Violated Constraint Added!!\n\n');
    end
    
    primalobjective = (state.lambda / 2) * (state.w' * state.w) + b - dot(state.w,phi);
    
    if (primalobjective <= bestPrimalObjective)
        bestPrimalObjective = primalobjective;
        bestState = state;
    end
    
    gap = (bestPrimalObjective - state.dualObjective) / state.dualObjective;
    
    fprintf([' %d primal objective: %f, best primal: %f, dual objective: %f, gap: %f\n'], ...
        numIterations, primalobjective, bestPrimalObjective, state.dualObjective,gap);
    
    state = bundler(state, phi, b);
    sparm.w = state.w;
    model.w = state.w;
    
    iteration.iter = numIterations;
    iteration.gap = [iteration.gap gap];
    
    if norm(model.w)==0
        fprintf('\n WARNING: Learned Weight Vector is Empty!!!\n\n');
    end
    
end

sparm.w = bestState.w;
model.w = bestState.w;

end



function [phi, b] = computeOneslackLovasz(sparm,model,X,Y,setfn)
phi = 0;
b = 0;

% For each pattern
for i = 1 : length(X);
    [gamma,deltaPsi] = sparm.findMostViolatedLovasz(sparm, model, X{i}, Y{i},setfn);
    if (gamma - dot(model.w,deltaPsi) > 0 )
        b = b + gamma;
        phi = phi + deltaPsi;
    end
end
end

function [phi, b] = computeOneslackMargin(sparm,model,X,Y,setfn)

phi = 0;
b = 0;

% For each pattern
for i = 1 : length(X);
    [tildeY] = sparm.findMostViolatedMargin(sparm, model, X{i}, Y{i},setfn);
    
    sumDelta = setfn(double(Y{i}~=tildeY));
    deltaPsi = sparm.psiFn(sparm, X{i} , Y{i})- sparm.psiFn(sparm, X{i}, tildeY);
    
    if (sumDelta - dot(model.w,deltaPsi)  > 0)
        b = b + sumDelta;
        phi = phi + deltaPsi;
    end
end

phi = phi';
end

function [phi, b] = computeOneslackSlack(sparm,model,X,Y,setfn)
phi = 0;
b = 0;

% For each pattern
for i = 1 : length(X);
    [tildeY] = sparm.findMostViolatedSlack(sparm, model, X{i}, Y{i},setfn);
    
    sumDelta = setfn(double(Y{i}~=tildeY));
    
    deltaPsi = sparm.psiFn(sparm, X{i}, Y{i}) - sparm.psiFn(sparm, X{i}, tildeY);
    
    if (sumDelta*(1-dot(model.w,deltaPsi)) >0)
        b = b + sumDelta;
        phi = phi + deltaPsi.*sumDelta;
        
    end
end

phi = phi';
end
