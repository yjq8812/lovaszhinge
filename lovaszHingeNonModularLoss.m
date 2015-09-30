function lovaszHingeNonModularLoss(f)
% implementation by Jiaqian Yu (jiaqian.yu@centralesupelec.com)
% J. Yu and M. B. Blaschko. Learning submodular losses with the Lovasz hinge. 
% In Proceedings of the International Conference on Machine Learning, 2015.
%
% This function plot the extension surface of Lovasz Hinge for a set funciton f:
% max_pi sum_i (l(a_j|0<j<=i)-l(a_k|0<k<i))*(1-g(x,y*)+g(x,y~))
% f: a 4-inputs set function with the order f(\emptyset), f({i_1}), f({i_2}), f({i_1,i_2})
% for a 2-outputs prediction problem.
%
%   x-axis: 1-g(y1*)
%   y-axis: 1-g(y2*)
%   e.g. pt f(1,2) at 1-g(y1*)=1,1-g(y2*)=0; etc
%   y1*=y1 ,y2*~=y2, f(1,2)
%   y1*~=y1,y2*=y2 , f(2,1)
% 

if(~exist('f','var'))
    f = [0 1;1 2];
end

if (length(find(f<0))>0)
    warning('Loss Function is supposed to be non-negative!')
end

[X,Y]=meshgrid([-1.6:0.1:1.6],[-1.6:0.1:1.6]);

output= zeros(size(X))+f(1,1);

pi_p1 = (f(1,2)-f(1,1))*X+(f(2,2)-f(1,2))*Y;
pi_p2 = (f(2,1)-f(1,1))*Y+(f(2,2)-f(2,1))*X;

if (f(2,2)>=f(1,2) && f(2,2)>=f(2,1))
    [rx,cx] = find(X<0);
    pi_p1(rx,cx) = (f(2,2)-f(1,2))*Y(rx,cx);
    pi_p2(rx,cx) = (f(2,1)-f(1,1))*Y(rx,cx);
    [ry,cy] = find(Y<0);
    pi_p1(ry,cy) = (f(1,2)-f(1,1))*X(ry,cy);
    pi_p2(ry,cy) = (f(2,2)-f(2,1))*X(ry,cy);
end

max_pi = max(pi_p1,pi_p2);
output = max(output,max_pi);%

% [rx,cx] = find(X==0);
% output = max(output,output(rx,cx));
% [ry,cy] = find(Y==0);
% output = max(output,output(ry,cy));
figure 

surf(X,Y,output);
xlabel('X')
ylabel('Y')
zlabel('Z')
hold on
scatter3([0 0 1 1]',[0 1 0 1]',f(:),'MarkerEdgeColor','k','MarkerFaceColor','r');
title(['f = [' num2str(f(1)) ' ' num2str(f(2)) '; ' num2str(f(3)) ' ' num2str(f(4)) ']'])
view([-37.5 30])
hold off

figure 
surf(X,Y,output);
xlabel('X')
ylabel('Y')
zlabel('Z')
hold on
scatter3([0 0 1 1]',[0 1 0 1]',f(:),'MarkerEdgeColor','k','MarkerFaceColor','r');
title(['f = [' num2str(f(1)) ' ' num2str(f(2)) '; ' num2str(f(3)) ' ' num2str(f(4)) ']'])
view([136.5 18])
hold off
end
