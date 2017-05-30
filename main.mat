function main
% This function solves an arbitrary second order stochastic ordinary 
% differential equation using the stochastic symmetric interior penalty
% method. It supports h, p, and s adaptivity and displays the solution
% along with first expansion field as its output.

% TO DO:
% - Higher order p adaptivity
% - Convergence plots
% - Second order output functionals if time allows, TEST in adjoint function

clr
% Define solver parameters
pquad = 5;      % Spatial quadrature rule
squad = 18;      % Stochastic quadrature rule
nelem = 12;     % Number of elements

% Define equation to be solved
A.nu = 0.001;         % Diffusion parameter
A.nuq = 0.0005;       % Diffusion parameter uncertainty
A.alpha = 0.0;       % Left boundary condition
A.alphaq = 0.00;     % Left boundary condition uncertainty
A.beta = -0;         % Right boundary condition
A.rhsfun = @(xx) ones(length(xx),1);    % Forcing function
A.rhsfunq = 0.0;     % Affine forcing function uncertainty term
A.testfun = @(xx) heaviside(xx - 0.5);  % Test function for output functional
A.N = 0;             % Advection term selection, 2 = Burgers, 1 = Linear advection, 0 = None
A.c = 1.0;           % Reaction term coefficient

% Initialize with linear spatial polynomials, and constant PC expansions on
% each element
% p = [1 1 1 1];
% s = [1 1 1 1];
p = 1*ones(1,nelem);
s = 1*ones(1,nelem);

% Define uniform mesh initially
xx = linspace(0,1,nelem+1);

% Set different types of adaptivity
h_adapt_flag = 1;
p_adapt_flag = 0;
s_adapt_flag = 1;

% Set number of DWR iterations, begin DWR loop
DWR_iter = 10;
Error = zeros(DWR_iter,1);
Err_Est = zeros(DWR_iter,1);
Error_p = zeros(DWR_iter,1);
Err_Est_p = zeros(DWR_iter,1);
Error_s = zeros(DWR_iter,1);
Err_Est_s = zeros(DWR_iter,1);
Ndof = zeros(DWR_iter,1);
for j = 1:DWR_iter
    % Calculate solution on coarse mesh, to be interpolated
    U = zeros(sum((p+1).*s),1);
    U1 = Newton(U,p,nelem,s,pquad,squad,xx,p,A);
    
    if h_adapt_flag == 1 || p_adapt_flag == 1
        % Interpolate solution on coarse mesh
        u1_int = linstretch(U1,p,s);
        % Define enriched p array
        p_inc = p + 1;
        
        % Calculate residual and jacobian for given U vector
        [Rs_int,RUselfs_int,RULs_int,RURs_int] = ResJac(u1_int,p_inc,nelem,s,pquad,squad,xx,p,A);
        % Reshape residual and jacobian tensors, Jacobian is used to find adjoint
        [Rf_int,RUf_int] = Rs2Rf(Rs_int,RUselfs_int,RULs_int,RURs_int,nelem,s,p_inc);
        
        % Calculate Adjoint with converged solution
        [z,E] = Adjoint(u1_int,RUf_int,p_inc,nelem,s,pquad,squad,xx,A);
        Error_p(j) = E;
        Err_Est_p(j) = abs(Rf_int'*z);
        
        % Calculate Dual Weighted Residual
        D = Rf_int.*z;
        DWR = zeros(nelem,1);
        for idx = 1:nelem
            DWR(idx) = sum(abs(D(sum(s(1:idx-1).*(p_inc(1:idx-1)+1))+1:sum(s(1:idx).*(p_inc(1:idx)+1)))));
        end
        
        % Mark elements with highest error estimates
        mark = zeros(nelem,1);
        DWR_sort = sort(abs(DWR));
        DWR_mark = DWR_sort(floor(0.85*nelem)+1:end);
        for n = 1:nelem
            for i = 1:length(DWR_mark)
                if DWR_mark(i) == abs(DWR(n))
                    mark(i) = n;
                end
            end
        end
        mark = (sort(nonzeros(mark)));
        % mark = 1:length(xx)-1; % Uncomment to mark all elements for refinement
        
        % Plot adjoint weighted residuals
%         figure(j)
%         hold on
%         plot(xx(1:end-1),abs(DWR),'-o')
%         plot(xx(mark),abs(DWR(mark)),'ro')
%         title('Dual Weighted Residual')
%         xlabel('\Omega')
%         ylabel('DWR')

    end
    
    if s_adapt_flag == 1
        % Define enriched stochastic expansion array
        s_inc = s + 1;
        
        % Stochastically extrapolate the solution
        u1_sint = zeros(sum((p+1).*(s_inc)),1);
        for elem = 1:nelem
            u1_sint(sum(s_inc(1:elem-1).*(p(1:elem-1)+1))+1:sum(s_inc(1:elem).*(p(1:elem)+1))) = [U1(sum(s(1:elem-1).*(p(1:elem-1)+1))+1:sum(s(1:elem).*(p(1:elem)+1))); zeros((p(elem)+1),1)];
        end
        
        % Calculate residual and jacobian for given stochastically enriched U vector
        [Rs_sint,RUselfs_sint,RULs_sint,RURs_sint] = ResJac(u1_sint,p,nelem,s_inc,pquad,squad,xx,p,A);
        % Reshape residual and jacobian tensors, Jacobian is used to find adjoint
        [Rf_sint,RUf_sint] = Rs2Rf(Rs_sint,RUselfs_sint,RULs_sint,RURs_sint,nelem,s_inc,p);
        
        % Calculate Adjoint with stochastically enriched solution
        [z_s,E] = Adjoint(u1_sint,RUf_sint,p,nelem,s_inc,pquad,squad,xx,A);
        Error_s(j) = E;
        Err_Est_s(j) = abs(Rf_sint'*z_s);
        
        % Calculate stochastic adjoint weighted residual
        Ds = Rf_sint.*z_s;
        DWRs = zeros(nelem,1);
        for idx = 1:nelem
            % disp(sum(s_inc(1:idx-1).*(p(1:idx-1)+1))+1:sum(s_inc(1:idx).*(p(1:idx)+1)))
            DWRs(idx) = sum(abs(Ds(sum(s_inc(1:idx-1).*(p(1:idx-1)+1))+1:sum(s_inc(1:idx).*(p(1:idx)+1)))));
            % Zs(idx) = sum(z_s(sum(s(1:idx-1).*(p+1))+1:sum(s(1:idx).*(p+1))));
        end
        
        % Mark elements with highest error estimates
        smark = zeros(nelem,1);
        DWRs_sort = sort(abs(DWRs));
        DWRs_mark = DWRs_sort(floor(0.85*nelem)+1:end);
        for n = 1:nelem
            for i = 1:length(DWRs_mark)
                if DWRs_mark(i) == abs(DWRs(n))
                    smark(i) = n;
                end
            end
        end
        smark = (sort(nonzeros(smark)));
        % smark = 1:length(xx) - 1; % Uncomment to mark all elements for refinement
        
        % Plot adjoint weighted residuals
        figure(j)
        hold on
        plot(xx(1:end-1),abs(DWRs),'-ko')
        plot(xx(smark),abs(DWRs(smark)),'ro')
        title('Dual Weighted Residual')
        xlabel('\Omega')
        ylabel('DWR')
    end
    Error(j) = Error_p(j) + Error_s(j);
    Err_Est(j) = Err_Est_p(j) + Err_Est_s(j);
    
        xx_prev = xx; p_prev = p; s_prev = s;
        if s_adapt_flag == 1
            s(smark) = s(smark) + 1;
        end
        if p_adapt_flag == 1
            p(mark) = p(mark) + 1;
        end
        if h_adapt_flag == 1
        [xx, s, p] = adaptmesh(xx,s,p,mark);
        end
        
    
    nelem = length(xx)-1;
    fprintf('Done %d of %d DWR iterations \n',j,DWR_iter)
    Ndof(j) = sum((p+1).*s);
end

%% Plotting
figure(j+1)
try
    plotcoefs(U1,p_prev,s_prev,xx_prev)
catch
    plotcoefs(U1,p,s,xx)
end
% ylim([-0.2 1.1])
title('Adaptive Solution with First Expansion Field')
figure
loglog(Ndof,Error,'-o',Ndof,Err_Est,'-o')
end

function [xx, s, p] = adaptmesh(xx,s,p,mark)
% The adaptmesh function takes in the 1D mesh corresponding to the problem
% along with the spatial and chaos expansion orders on each element, then
% splits marked elements in two, preserving the original polynomial orders.
%
% INPUTS:
%   xx   = nelem+1 by 1 array corresponding to node points for the mesh
%   s    = nelem by 1 array representing the number of PC expansion modes
%           on each element
%   p    = nelem by 1 array representing the polynomial order on each element
%   mark = 0.1*nelem by 1 array containting indicies for elements to be
%           h-refined
%
% OUTPUTS:
%   xx   = nelem+1+length(mark) by 1 array representing refined mesh
%   s    = nelem + length(mark) by 1 array representing number of PC
%              expansion modes on newly refined mesh
%   p    = nelem + length(mark) by 1 array representing polynomial order
%              for spatial basis functions on each element of refined mesh

% Loop through each element, placing new nodes between marked nodes of each
% mared element, and preserving the PC and spatial polynomial orders for
% each element
for i = 1:length(mark)
    xx = [xx(1:mark(i)), (xx(mark(i))+xx(mark(i)+1))/2, xx(mark(i)+1:end)];
    s = [s(1:mark(i)), s(mark(i)), s(mark(i)+1:end)];
    p = [p(1:mark(i)), p(mark(i)), p(mark(i)+1:end)];
    
    if i ~= length(mark)
        mark(i+1) = mark(i+1) + i;
    end
end
end

function U = Newton(U,p,nelem,s,pquad,squad,xx,pp,A)
% The Newton function takes some solution vector, evaluates the residual
% and iterates until the norm of said residual is below some tolerance
% (typically machine precision). 
%
% INPUTS:
%   U     = Ndof by 1 array of solution coefficients, typically zeros
%   p     = nelem by 1 array of spatial basis enriched polynomial order on each element
%   nelem = scalar, number of elements
%   s     = nelem by 1 array representing number of PC expansion modes on each element
%   pquad = scalar, number of deterministic quadrature points
%   squad = scalar, number of stochastic quadrature points
%   xx    = nelem+1 by 1 array, node locations in mesh
%   pp    = nelem by 1, non-enriched spatial polynomial order on each
%              element, used in penalty terms
%   A     = Structure containing relevant information about the equation
%              being solved, BCs, type of equation, forcing function, etc.
%
% OUTPUTS:
%   U     = Ndof by 1 array of converged solution coefficients
%              based on given mesh parameters

% Define maximum number of Newton iterations, 12 is likely plenty
max_iter = 12;
for iter = 1:max_iter
    
    % Calculate residual and jacobian for given U vector
    [Rs,RUselfs,RULs,RURs] = ResJac(U,p,nelem,s,pquad,squad,xx,pp,A);
    
    % Reshape residual and jacobian tensors
    [Rf,RUf] = Rs2Rf(Rs,RUselfs,RULs,RURs,nelem,s,p);
    
    
    % Calculate norm of the residual, used for convergence
    resnorm = norm(Rf(:),2);
    
    % Define tolerance and break if the residual is less than tolerance
    tol = 2e-12;
    % fprintf('Residual Norm = %e\n',resnorm) % Uncomment to print resnorm
    if (resnorm < tol)
        % fprintf('Solution has converged after %d iterations\n', iter-1) 
        break;
    end
    
    % Calculate Newton update
    dU = RUf\-Rf;
    U = U + dU;
    
end         % ===== End Newton Loop ======
% If solution does not converge, display error
if iter == max_iter
    error('Solution has failed to converge')
end
end

function [z,E] = Adjoint(U,RUf,p,nelem,s,pquad,squad,xx,A)
% The adjoint function determines the adjoint of the given solution, which
% represents the sensitivity to some output J(u) to small perturbations in
% the solution, it also returns an error estimate based on this adjoint.
%
% INPUTS:
%   U     = Ndof by 1 array of solution coefficients, where Ndof has been
%              updated based on an enriched p or s array
%   RUf   = Ndof by Ndof matrix representing the Jacobian of the problem,
%              that is the derivative of the residual 
%   p     = nelem by 1 array of (possibly enriched) spatial basis polynomial orders on each element
%   nelem = scalar, number of elements
%   s     = nelem by 1 array representing the (possibly enriched) number of PC expansion modes on each element
%   pquad = scalar, number of deterministic quadrature points
%   squad = scalar, number of stochastic quadrature points
%   xx    = nelem+1 by 1 array, node locations in mesh
%   A     = Structure containing relevant information about the equation
%              being solved, BCs, type of equation, forcing function, etc.
%
% OUTPUTS:
%   z     = Ndof by 1 array of coefficients representing adjoint solution
%   E     = scalar, error in output value for specified mesh configuration

% Set stochastic and deterministic quadrature rules
[sxq,swq] = quadrature_11(squad);
[xq,wq] = quadrature_11(pquad);
h = diff(xx)/2;

% Define test function for output functional
testfun = A.testfun;

% Preallocate memory for linearized output functional
J = zeros(sum((p+1).*s),1);

% Calculate Linearized output functional
m = 1;
for elem = 1:nelem
    [phi,~] = shape(p(elem),xq);
    for k = 1:s(elem)
        psij = lndshape(k,sxq,1);
        
        J(m:m+p(elem)) = phi'*(wq*h(elem).*testfun((xx(elem)+xx(elem+1))/2+xq*h(elem)))*(swq'*psij);
        m = m + p(elem) + 1;
    end
end

%TEST
% Jold = J;
% J = zeros(sum((p+1).*s),1);
% J2 = zeros(sum((p+1).*s),1);
% m = 1;
% for elem = 1:nelem
%     [phi,~] = shape(p(elem),xq);
%     for l = 1:length(sxq')
%         q = sxq(l);
%         wqs = swq(l);
%         Ucoef = zeros((p(elem)+1),1);
%         for k = 1:s(elem)
%             psij = lndshape(k,q,1);
%             Ucoef = Ucoef + U(sum(s(1:elem-1).*(p(1:elem-1)+1))+1 + (k-1)*(p(elem)+1):sum(s(1:elem-1).*(p(1:elem-1)+1))+1+p(elem)+ (k-1)*(p(elem)+1));
%             UQ = phi*Ucoef;
% %             J(m:m+p(elem)) = phi'*(wq*h(elem).*testfun((xx(elem)+xx(elem+1))/2+xq*h(elem)))*(swq'*psij);
%             J2(m:m+p(elem)) = J2(m:m+p(elem)) + 2*(phi'*(wqs.*wq*h(elem).*(UQ.*testfun(xx(elem)+xq*h(elem)).*psij)));
%             psij = lndshape(k,sxq,1);
%         
%         J(m:m+p(elem)) = phi'*(wq*h(elem).*testfun((xx(elem)+xx(elem+1))/2+xq*h(elem)))*(swq'*psij);
%         end
%         
%     end
%     m = m + p(elem) + 1;
% end
% ENDTEST

% Analytical integral, this value is calculated once with an expensive mesh
% then cheaper solutions are compared to it
J_true = 0.937347181443894;

% Calculate approximate value of linearized output functional
Jh = J'*U;

% Calculate error in J,without correct J_true this value is meaningless
E = abs(J_true - Jh);

% Calculate adjoiint
z = RUf'\J;
end

function [Rs,RUselfs,RULs,RURs] = ResJac(U,p,nelem,s,pquad,squad,xx,pp,A)
% This function calculates the stochastic residual and Jacobian for a given
% solution, mesh, and equation. 
%
% INPUTS:
%   U     = Ndof by 1 array of solution coefficients
%   p     = nelem by 1 array of (possibly enriched) spatial basis polynomial orders on each element
%   nelem = scalar, number of elements
%   s     = nelem by 1 array representing the (possibly enriched) number of PC expansion modes on each element
%   pquad = scalar, number of deterministic quadrature points
%   squad = scalar, number of stochastic quadrature points
%   xx    = nelem+1 by 1 array, node locations in mesh
%   pp    = nelem by 1, non-enriched spatial polynomial order on each
%              element, used in penalty terms
%   A     = Structure containing relevant information about the equation
%              being solved, BCs, type of equation, forcing function, etc.
%
% OUTPUTS:
%   Rs      = p+1 by s by nelem tensor, representing the residual of the
%                   problem for the given solution at each dof
%   RUselfs = p+1 by p+1 by s by s by nelem tensor representing the
%                   Jacobian of the problem calculated with test functions
%                   on the self element
%   RULs    = p+1 by p+1 by s by s by nelem tensor representing the
%                   Jacobian of the problem calculated with test functions
%                   on the left element
%   RURs    = p+1 by p+1 by s by s by nelem tensor representing the
%                   Jacobian of the problem calculated with test functions
%                   on the right element

% Preallocate memory for deterministic residual and Jacobian
R = zeros(max(p)+1, nelem);
RUself = zeros(max(p)+1,max(p)+1,nelem);
RUL = zeros(max(p)+1,max(p)+1,nelem);
RUR = zeros(max(p)+1,max(p)+1,nelem);

% Define stability constant, left boundary condition, and advection
% parameter
Cip = 10.0;
beta = A.beta;
N = A.N;

% Preallocate memory for stochastic residual and Jacobian tensors
Rs = zeros(max(p)+1,max(s),nelem);
RUselfs = zeros(max(p)+1,max(p)+1,max(s),max(s),nelem);
RULs = zeros(max(p)+1,max(p)+1,max(s),max(s),nelem);
RURs = zeros(max(p)+1,max(p)+1,max(s),max(s),nelem);

% Take forcing function from equation struct
rhsfun = A.rhsfun;

% Here the residual and jacobian are calculated on an element by
% element basis
for elem = 1:nelem
    % Define element boundaries and determine element size
    h = (xx(elem+1) - xx(elem))/2;
    
    %% VOLUME TERM %%
    % Evaluate deterministic and stochastic quadrature rules then
    % evaluate deterministic shape functions
    [xq,wq] = quadrature_11(pquad);
    [sxq,swq] = quadrature_11(squad);
    [shp,shpx] = shape(p(elem),xq);
    
    % Here the deterministic solver is evaulated at a state
    % corresponding to each stochastic quadrate point, evaulating all
    % quadrature points will correspond to the state vector along each
    % degree of stochastic expansion
    for l = 1:length(sxq')
        % Select quadrature point and weight to be evaluated
        q = sxq(l);
        wqs = swq(l);
        
        % Define stochastic input parameters
        nu = A.nu + A.nuq*q;
        alpha = A.alpha + A.alphaq*q;
        
        % Define shape function and weights on spatial element
        wqJ = wq*h;
        phi = shp;
        phix = 1/h*shpx;
        
        % Find coefficients relevant to the specific element by summing
        % over all stochastic expansions
        Ucoef = zeros(p(elem)+1,1);
        for jj = 1:s(elem)
            psij = lndshape(jj,q,1);
            Ucoef = Ucoef + U(sum(s(1:elem-1).*(p(1:elem-1)+1))+1 + (jj-1)*(p(elem)+1):sum(s(1:elem-1).*(p(1:elem-1)+1))+1+p(elem)+ (jj-1)*(p(elem)+1))*psij;
        end
        
        % Evaluate state at quad points, for nonlinear problems,
        % coefficients cannot be seperated from the basis
        UQ = phi*Ucoef;
        UQx = phix*Ucoef;
        
        % Define flux functions
        [f, f_u] = f_inv(UQ,N);
        
        % Evaluate residual on self
        % \nabla v nu\nabla u
        R(1:p(elem)+1,elem) = nu*phix'*(wqJ.*UQx);
        
        c = A.c;
        % vu
        R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) + c*phi'*(wqJ.*UQ);
        
        % -\nabla v finv
        R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) - phix'*(wqJ.*f);
        
        % -v f                                                        *
        R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) - phi'*(wqJ.*(rhsfun((xx(elem)+xx(elem+1))/2+xq*h) + A.rhsfunq*q));
        
        % Evaluate jacobian on self
        % \nabla v nu\nabla u
        RUself(1:p(elem)+1,1:p(elem)+1,elem) = nu*phix'*diag(wqJ)*phix;
        RUL(1:max(p)+1,1:max(p)+1,elem) = zeros(max(p)+1);
        RUR(1:max(p)+1,1:max(p)+1,elem) = zeros(max(p)+1);
        
        % vu
        RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) + c*phi'*diag(wqJ)*phi;
        
        % -\nabla v finv_u u
        RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) - phix'*diag(wqJ.*f_u)*phi;
        
        %% LEFT FACE TERM
        % Set up basis on self
        [shpp, shpxp] = shape(p(elem),-1);
        hp = h;
        phip = shpp;
        phixp = 1/hp*shpxp;
        UQp = phip*Ucoef;
        UQxp = phixp*Ucoef;
        
        % Define normal vector for left element
        np = -1.0; % outward pointing normal from self
        nm = 1.0; % outward pointing normal from the neighbor
        
        % For boundary element
        if (elem == 1)
            % Boundary Flux
            [H,H_up,~] = Hinv(UQp,alpha,np,N);
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) + H*phip';
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) + phip'*H_up*phip;
            
            % -nu\nabla u [v]
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) - nu*np*phip'*UQxp;
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) - nu*np*phip'*phixp;
            
            % (u+ - ug)[v]
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) + nu*Cip/hp*pp(elem)^2*np*np*phip'*UQp;
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) + nu*Cip/hp*pp(elem)^2*np*np*(phip)'*phip;
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) - nu*Cip/hp*pp(elem)^2*phip'*alpha;   % From ug
            
            % -nu\nabla v (u+ - ug)
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) - nu*np*phixp'*UQp;
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) - nu*np*phixp'*phip;
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) - nu*phixp'*alpha;          % From ug
            
        else
            % INTERIOR
            % Set up basis for neighbour element
            [shpm, shpxm] = shape(p(elem-1),1.0);
            hm = (xx(elem) - xx(elem-1))/2;
            phim = shpm;
            phixm = 1/hm*shpxm;
            
            % Determine coefficients for neighbour element
            Ucoefm = zeros(p(elem-1)+1,1);
            for jj = 1:s(elem-1)
                psij = lndshape(jj,q,1);
                Ucoefm = Ucoefm + U(sum(s(1:elem-2).*(p(1:elem-2)+1)) + 1 + (jj-1)*(p(elem-1)+1):sum(s(1:elem-2).*(p(1:elem-2)+1)) + (p(elem-1)+1) + (jj-1)*(p(elem-1)+1))*psij;
            end
            UQm = phim*Ucoefm;
            UQxm = phixm*Ucoefm;
            
            % Flux term
            [H,H_up,H_um] = Hinv(UQp,UQm,np,N);
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) + H*phip';
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) + phip'*H_up*phip;
            RUL(1:p(elem)+1,1:p(elem-1)+1,elem)  = RUL(1:p(elem)+1,1:p(elem-1)+1,elem)  + phip'*H_um*phim;
            
            % -nu{\nabla v} [ u ]
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) - nu*0.5*phixp'*(np*UQp + nm*UQm);
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) - nu*0.5*phixp'*np*phip;
            RUL(1:p(elem)+1,1:p(elem-1)+1,elem)  = RUL(1:p(elem)+1,1:p(elem-1)+1,elem)  - nu*0.5*phixp'*nm*phim;
            
            % -[ v ] {nu\nabla u}
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) - nu*0.5*np*phip'*(UQxp+UQxm);
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) - nu*0.5*phip'*np*phixp;
            RUL(1:p(elem)+1,1:p(elem-1)+1,elem)  = RUL(1:p(elem)+1,1:p(elem-1)+1,elem)  - nu*0.5*phip'*np*phixm;
            
            % Cip [ v ] [ u ]
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) + nu*Cip*((pp(elem)+pp(elem-1)))^2*np*phip'*(nm*UQm + np*UQp)/(hm+hp);
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) + nu*Cip/(hp+hm)*((pp(elem)+pp(elem-1)))^2*phip'*np*np*phip;
            RUL(1:p(elem)+1,1:p(elem-1)+1,elem)  = RUL(1:p(elem)+1,1:p(elem-1)+1,elem)  + nu*Cip/(hm+hp)*((pp(elem)+pp(elem-1)))^2*phip'*np*nm*phim;
            
        end
        %% RIGHT FACE TERM
        % Set up basis on self
        [shpp, shpxp] = shape(p(elem),1.0); % basis on self
        hp = h;
        phip = shpp;
        phixp = 1/hp*shpxp;
        UQp = phip*Ucoef;
        UQxp = phixp*Ucoef;
        
        % Define normal vectors
        np = 1.0; % outward pointing normal from self
        nm = -1.0; % outward pointing normal from right neighbour
        
        % For right boundary
        if (elem == nelem)
            % Boundary Flux
            [H,H_up,~] = Hinv(UQp,beta,np,N);
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) + H*phip';
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) + phip'*H_up*phip;
            
            % -nu\nabla u [v]
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) - nu*np*phip'*UQxp;
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) - nu*np*phip'*phixp;
            
            % (u+ - ug)[v]
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) + nu*Cip/hp*pp(elem)^2*np*np*phip'*UQp;
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) + nu*Cip/hp*pp(elem)^2*np*np*(phip)'*phip;
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) - nu*Cip/hp*pp(elem)^2*beta*phip';     % From ug
            
            % -nu\nabla v (u+ - ug)
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) - nu*np*phixp'*UQp;
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) - nu*np*phixp'*phip;
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) + nu*phixp'*beta;  % From ug
            
        else
            % INTERIOR
            % Set up basis on neighbour element
            [shpm, shpxm] = shape(p(elem+1),-1.0);
            hm = (xx(elem+2) - xx(elem+1))/2;
            phim = shpm;
            phixm = 1/hm*shpxm;
            
            % Determine state coefficients for neighbour element
            Ucoefm = zeros(p(elem+1)+1,1);
            for jj = 1:s(elem+1)
                psij = lndshape(jj,q,1);
                Ucoefm = Ucoefm + U(sum(s(1:elem).*(p(1:elem)+1)) + 1 + (jj-1)*(p(elem+1)+1):sum(s(1:elem).*(p(1:elem)+1)) + (p(elem+1)+1) + (jj-1)*(p(elem+1)+1))*psij;
            end
            UQm = phim*Ucoefm;
            UQxm = phixm*Ucoefm;
            
            % Flux term(v)
            [H,H_up,H_um] = Hinv(UQp,UQm,np,N);
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) + H*phip';
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) + phip'*H_up*phip;
            RUR(1:p(elem)+1,1:p(elem+1)+1,elem)  = RUR(1:p(elem)+1,1:p(elem+1)+1,elem)  + phip'*H_um*phim;
            
            % -{nu\nabla v} [ u ]
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) - nu*0.5*phixp'*(np*UQp + nm*UQm);
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) - nu*0.5*phixp'*np*phip;
            RUR(1:p(elem)+1,1:p(elem+1)+1,elem)  = RUR(1:p(elem)+1,1:p(elem+1)+1,elem)  - nu*0.5*phixp'*nm*phim;
            
            % -[ v ] {nu\nabla u}
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) - nu*0.5*np*phip'*(UQxp+UQxm);
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem) - nu*0.5*phip'*np*phixp;
            RUR(1:p(elem)+1,1:p(elem+1)+1,elem)  = RUR(1:p(elem)+1,1:p(elem+1)+1,elem)  - nu*0.5*phip'*np*phixm;
            
            % Cip [v] [u]
            R(1:p(elem)+1,elem) = R(1:p(elem)+1,elem) + nu*Cip*((pp(elem)+pp(elem+1)))^2*np*phip'*(nm*UQm + np*UQp)/(hm+hp);
            RUself(1:p(elem)+1,1:p(elem)+1,elem) = RUself(1:p(elem)+1,1:p(elem)+1,elem)   + nu*Cip/(hp+hm)*((pp(elem)+pp(elem+1)))^2*phip'*np*np*phip;
            RUR(1:p(elem)+1,1:p(elem+1)+1,elem)  = RUR(1:p(elem)+1,1:p(elem+1)+1,elem)    + nu*Cip/(hm+hp)*((pp(elem)+pp(elem+1)))^2*phip'*np*nm*phim;
            
        end       
        
        % Assemble complete spatio-stochastic residual and jacobian
        for j = 1:s(elem)
            psij = lndshape(j,q,1);
            Rs(1:p(elem)+1,j,elem) = Rs(1:p(elem)+1,j,elem) + R(1:p(elem)+1,elem)*psij*wqs;
            for k = 1:s(elem)
                psik = lndshape(k,q,1);
                RUselfs(1:p(elem)+1,1:p(elem)+1,j,k,elem) = RUselfs(1:p(elem)+1,1:p(elem)+1,j,k,elem) + RUself(1:p(elem)+1,1:p(elem)+1,elem)*psij*psik*wqs;
            end
            if elem ~= 1
                for k = 1:s(elem-1)
                    psik = lndshape(k,q,1);
                    RULs(1:p(elem)+1,1:p(elem-1)+1,j,k,elem) = RULs(1:p(elem)+1,1:p(elem-1)+1,j,k,elem) + RUL(1:p(elem)+1,1:p(elem-1)+1,elem)*psij*psik*wqs;
                end
            end
            if elem ~= nelem
                for k = 1:s(elem+1)
                    psik = lndshape(k,q,1);
                    RURs(1:p(elem)+1,1:p(elem+1)+1,j,k,elem) = RURs(1:p(elem)+1,1:p(elem+1)+1,j,k,elem) + RUR(1:p(elem)+1,1:p(elem+1)+1,elem)*psij*psik*wqs;
                end
            end
        end
    end   % ===== End Stochastic Quadrature Loop =====
end    % ===== End Element Loop =====
end

function [Rf,RUf] = Rs2Rf(Rs,RUselfs,RULs,RURs,nelem,s,p)
% This function converts the tensors Rs and RUs into a
% vector and matrix, respectively. This will allow us to determine our
% coefficients by solving system of linear equations
% 
% INPUTS:
%   Rs      = p+1 by s by nelem tensor, representing the residual of the
%                   problem for the given solution at each dof
%   RUselfs = p+1 by p+1 by s by s by nelem tensor representing the
%                   Jacobian of the problem calculated with test functions
%                   on the self element
%   RULs    = p+1 by p+1 by s by s by nelem tensor representing the
%                   Jacobian of the problem calculated with test functions
%                   on the left element
%   RURs    = p+1 by p+1 by s by s by nelem tensor representing the
%                   Jacobian of the problem calculated with test functions
%                   on the right element
%   nelem   = scalar, number of elements
%   s       = nelem by 1 array representing number of PC expansion modes on each element
%   p       = nelem by 1 array of spatial basis enriched polynomial order on each element
%
% OUTPUTS:
%   Rf      = Ndof by 1 == (s by p+1 by nelem) by 1 array, residual of the problem
%   RUf     = Ndof by Ndof matrix, Jacobian of the problem

% Preallocate memory for fully rearranged residual and Jacobian
Rf = zeros(sum((p+1).*s),1);
RUf = zeros(sum((p+1).*s));

% Build full block residual from residual tensor
    m = 1;
    
    for n = 1:nelem
        for j = 1:s(n)
            Rf(m:m+p(n)) = Rs(1:p(n)+1,j,n);
            m = m + (p(n)+1);
        end
    end
    
    % Build full block jacobian from jacobian tensor
    a = 1; b = 1;
    for n = 1:nelem
        if n == 1
            for j = 1:s(n)
                for i = 1:s(n)
                    RUf(a:a+p(n),b:b+p(n)) = RUselfs(1:p(n)+1,1:p(n)+1,i,j,n);
                    % spy(RUf)
                    a = a + (p(n)+1);
                end
                a = n*(p(n)+1)*s(n) - (p(n)+1)*s(n) + 1;
                b = b + (p(n)+1);
            end
            b = 1;
            for j = 1:s(n+1)
                for i = 1:s(n)
                    RUf(a:a+p(n),b+s(n)*(p(n)+1):b+s(n)*(p(n)+1)+p(n+1)) = RURs(1:p(n)+1,1:p(n+1)+1,i,j,n);
                    % spy(RUf)
                    a = a + (p(n)+1);
                end
                a = n*(p(n)+1)*s(n) - (p(n)+1)*s(n) + 1;
                b = b + (p(n+1)+1);
            end
            
        elseif n ~= nelem
            b = sum((p(1:n-1)+1).*s(1:n-1)) + 1;
            for j = 1:s(n)
                for i = 1:s(n)
                    RUf(a:a+p(n),b:b+p(n)) = RUselfs(1:p(n)+1,1:p(n)+1,i,j,n);
                    % spy(RUf)
                    a = a + (p(n)+1);
                end
                a = sum((p(1:n-1)+1).*s(1:n-1)) + 1;
                b = b + (p(n)+1);
            end
            b = sum((p(1:n-1)+1).*s(1:n-1)) + 1;
            for j = 1:s(n+1)
                for i = 1:s(n)
                    RUf(a:a+p(n),b+s(n)*(p(n)+1):b+s(n)*(p(n)+1)+p(n+1)) = RURs(1:p(n)+1,1:p(n+1)+1,i,j,n);
                    % spy(RUf)
                    a = a + (p(n)+1);
                end
                a = sum((p(1:n-1)+1).*s(1:n-1)) + 1;
                b = b + (p(n+1)+1);
            end
            b = sum((p(1:n-1)+1).*s(1:n-1)) + 1;
            for j = 1:s(n-1)
                for i = 1:s(n)
                    RUf(a:a+p(n),b-s(n-1)*(p(n-1)+1):b-s(n-1)*(p(n-1)+1)+p(n-1)) = RULs(1:p(n)+1,1:p(n-1)+1,i,j,n);
                    % spy(RUf)
                    a = a + (p(n)+1);
                end
                a = sum((p(1:n-1)+1).*s(1:n-1)) + 1;
                b = b + (p(n-1)+1);
            end
            
        else
            b = sum((p(1:n-1)+1).*s(1:n-1)) + 1;
            for j = 1:s(n)
                for i = 1:s(n)
                    RUf(a:a+p(n),b:b+p(n)) = RUselfs(1:p(n)+1,1:p(n)+1,i,j,n);
                    % spy(RUf)
                    a = a + (p(n)+1);
                end
                a = sum((p(1:n-1)+1).*s(1:n-1)) + 1;
                b = b + (p(n)+1);
            end
            b = sum((p(1:n-1)+1).*s(1:n-1)) + 1;
            for j = 1:s(n-1)
                for i = 1:s(n)
                    RUf(a:a+p(n),b-s(n-1)*(p(n-1)+1):b-s(n-1)*(p(n-1)+1)+p(n-1)) = RULs(1:p(n)+1,1:p(n-1)+1,i,j,n);
                    % spy(RUf)
                    a = a + (p(n)+1);
                end
                a = sum((p(1:n-1)+1).*s(1:n-1)) + 1;
                b = b + (p(n-1)+1);
            end
        end
        b = b + (sum(p(1:n-1)+1));
        a = sum((p(1:n)+1).*s(1:n)) + 1;
    end
end

function [f,f_u] = f_inv(u,N)
% Inviscid flux term, can be 0, linear or quadratic with respect to u, 
% N defines the advection term of the equation being solved
%
% INPUTS:
%   u   = p(elem)+1 by 1 double array representing approximate solution on a
%          specified element
%   N   = positive integer, no flux, linear flux or quadratic flux,
%          depending on problem {0, 1, 2}
%
% OUTPUTS:
%   f   = p(elem)+1 by 1 double, array representing flux function for the problem
%   f_u = p(elem)+1 by 1 double, array representing derivative of above

if N == 0
    f = 0;
    f_u = 0;
elseif N == 1
    f = u;
    f_u = ones(size(u));
elseif N == 2
    f = 0.5*u.*u;
    f_u = u;
else
    error('A.N should equal 0, 1, or 2.')
end
end

function [H,H_up,H_um] = Hinv(up,um,np,N)
% The function calculates the flux between elements within the domain, it
% also depends on the advection term N
%
% INPUTS:
%   up   = (p+1) by 1 double array representing approximate solution at quadrature
%              points on element of interest
%   um   = (p+1) by 1 double array representing approximate solution at quadrature
%              points on neighbouring element
%   np   = integer, outward pointing normal at boundary {-1, 1}
%   N    = positive integer, advection parameter, {0, 1, 2}
%
% OUTPUTS:
%   H    = (p+1) by 1 double array, residual flux term
%   H_up = (p+1) by 1 double array, Jacobian flux term on self
%   H_um = (p+1) by 1 double array, Jacobian flux term on neighbour

if N == 0
    H = 0;
    H_up = 0;
    H_um = 0;
elseif N == 1;
    H = 0.5*(np*up + np*um) - 0.5*abs(ones(length(up)))*(um - up);
    H_up = 0.5*1*(np + ones(length(up)));
    H_um = 0.5*1*(np - ones(length(um)));
elseif N == 2;
    H = 0.5*(0.5*np*up.*up + 0.5*np*um.*um) - 0.5*abs(up+um)*(um-up);
    H_up = 0.5*np*up - 0.5*sign(up+um)*(um-up) + 0.5*abs(up+um);
    H_um = 0.5*np*um - 0.5*sign(up+um)*(um-up) - 0.5*abs(up+um);
end
end

function [xq,wq] = quadrature_11(N)
% This function calculates N Gauss-Legendre quadrature nodes and weights 
% using the Golub-Welsch algorithm over (-1,1).
%
% INPUTS:
%   N  = positive integer, number of quadrature points
%
% OUTPUTS:
%   xq = N by 1 double, array of quadrature points
%   wq = N by 1 double, array of quadrature weights

J = zeros(N);
for i = 1:N;
    for j = 1:N;
        if i - j == 1
            J(i,j) = sqrt(j^2/(2*(j-1)+1)/(2*j+1));
        elseif i - j == -1
            J(i,j) = sqrt(i^2/(2*(i-1)+1)/(2*i+1));
        end
    end
end

[v,xq] = eig(J);

xq = diag(xq);
wq = (2*v(1,:).^2)';
end

function [shp,shpx] = shape(p,x)
% This function returns the Lagrange shape functions along with their 
% derivatives over x of order p.
%
% INPUTS:
%   p   = integer, polynomial order of shape functions
%   x   = M by 1 double array representing points where shape functions are
%          calculated, M is an arbitrary positive integer
%
% OUTPUTS:
%   shp  = M by p double precision floating point matrix, p Lagrange shape functions of order p
%   shpx = M by p double precision floating point matrix, derivatives of the above


switch p
    case 1
        np = length(x);
        shp(:,1) = (1.0 - x)/2;
        shp(:,2) = (1.0 + x)/2;
        shpx(:,1) = -ones(np,1)/2;
        shpx(:,2) = ones(np,1)/2;
    case 2
        shp(:,1) =  -1/2*x.*(1-x);
        shp(:,2) =  (1+x).*(1-x);
        shp(:,3) =  1/2*x.*(1+x);
        shpx(:,1) =  -1/2 + x;
        shpx(:,2) =  -2*x;
        shpx(:,3) =  1/2 + x;
    case 3
        shp(:,1) = -9/16*(x + 1/3).*(x - 1/3).*(x - 1);
        shp(:,2) = 27/16*(x + 1).*(x - 1/3).*(x - 1);
        shp(:,3) = -27/16*(x + 1).*(x + 1/3).*(x - 1);
        shp(:,4) = 9/16*(x + 1).*(x + 1/3).*(x - 1/3);
        shpx(:,1) = 1/16*(-27*x.^2 + 18*x + 1);
        shpx(:,2) = 9/16*(9*x.^2 - 2*x - 3);
        shpx(:,3) = -9/16*(9*x.^2 + 2*x - 3);
        shpx(:,4) = 1/16*(27*x.^2 + 18*x - 1);
        
    otherwise
        error('unsupported p');
end
end

function [leg,legx] = lndshape(N,x,stoc_flag)
% This function calculates the first N legendre polynomials evaluated at x,
% and returns either the Nth or 1 to Nth polynomials depending on the flag
% argument. 
%
% INPUTS:
%   N         = integer, Number of Legendre polynomials to calculate
%   x         = M by 1 double array representing points where shape functions are
%                   calculated, M is an arbitrary positive integer 
%   stoc_flag = boolean, 1 returns Nth polynomial, 0 returns 1 to Nth.
%
% OUTPUTS:
%   leg       = M by N or M by 1 array of 1 to N or Nth legendre
%                  polynomials evaluated at x
%   legx      = M by N or M by 1 array of 1 to N or Nth derivates of above

% Preallocate memory for legendre polynomials and their derivatives, hard
% code first two Legendre polynomials
leg = zeros(length(x),N);
leg(:,1) = ones(length(x),1);
leg(:,2) = x;
legx = zeros(length(x),N);
legx(:,1) = zeros(length(x),1);
legx(:,2) = ones(length(x),1);

% Use Bonnet's recursion formula to determines second to nth polynomials
for n = 1:N-2
    leg(:,n+2) = ((2*n + 1)*x.*leg(:,n+1) - n*leg(:,n))/(n+1);
    legx(:,n+2) = ((2*n + 1)*(leg(:,n+1) + x.*legx(:,n)) - n*legx(:,n))/(n+1);
end

% Return the Nth or first N Legendre polynomials
if stoc_flag == 1
    leg = leg(:,N);
    legx = legx(:,N);
end
end

function [u_int] = linstretch(u,p,s)
% This function interpolates the solution to a higher polynomial order on
% each element, the residual of the interpolated solution is used for error
% estimates
%
% INPUTS:
%   u = Ndof by 1 double precision array of solution coefficients
%   p = nelem by 1 int array of (non-enriched) spatial polynomial orders on each element
%   s = nelem by 1 int array of (non-enriched) polynomial chaos expansion modes on elements
%
% OUTPUTS:
%   u_int = Ndof' by 1 double array of interpolated solution coefficients,
%              here Ndof' corresponds to an enriched p or s array

% Preallocate memory for interpolated solution
u_int = zeros(sum(p+2),1);

% Interpolate each element independently
j = 1; k = 1;
for i = 1:length(p)
    
    % Solution in high PC modes must also be interpolated
    for o = 1:s(i)
        if p(i) == 1
            u_int(j:j+p(i)+1) = [u(k) (u(k+1) + u(k))/2 u(k+1)];
            j = j + p(i)+2;
        elseif p(i) == 2
            u_int(j:j+p(i)+1) = [shape(p(i),-1)*u(k:k+2) shape(p(i),-1/3)*u(k:k+2) shape(p(i),1/3)*u(k:k+2) shape(p(i),1)*u(k:k+2)];
            j = j + p(i)+2;
        end
        k = k + p(i) + 1;
    end
end
end

function plotcoefs(U,p,s,xx)
% This function plots the coefficients of the solution or adjoint using a
% Lagrange shape function basis.  
%
% INPUTS:
%   U  = Ndof by 1 double precision array of coefficients to be plotted
%   p  = nelem by 1 int array of polynomials degrees on each element
%   s  = nelem by 1 int array ofPC expansion modes on each element
%   xx = nelem+1 by 1 double array of node locations for each element
%
% OUTPUTS:
%   Plot of coefficients with respect to Lagrange basis functions over domain

% Define reference domain
x = -1:.1:1;

% Preallocate memory for solutions of each PC expansion mode on each element
soln = zeros(length(x),length(p),max(s));
xn = zeros(length(x),length(p));

% Build solution by summing basis coefficient products
j = 1;
for n = 1:length(p)
    for m = 1:s(n)
    solnt = zeros(length(x),p(n)+1);
    phi = shape(p(n),x);
        for i = 1:p(n)+1
        
        % Multiply coefficient by respective basis function
        solnt(:,i,m) = U(j)*phi(:,i);
        j = j + 1;
        
        % Sum over basis functions on each element
        soln(:,n,m) = sum(solnt(:,:,m),2);
        end
    end
    
    % Build array for solution to be plotted against
    xn(:,n) = linspace(xx(n),xx(n+1),length(soln(:,n,m)));
end
xn = reshape(xn,numel(xn),1);
solnf = zeros(numel(soln(:,:,1)),max(s));

% Store each stochastic expansion field separately
for m = 1:max(s)
    solnf(:,m) = reshape(soln(:,:,m),numel(soln(:,:,m)),1);
end
% Plot mean of solution
plot(xn,solnf(:,1),'k')

% If second expansion field exists and is non-zero, plot it as well
try
if norm(solnf(:,2)) > 1e-12
    hold on
    plot(xn,solnf(:,2),'--k')
end
catch
end
try
if norm(solnf(:,3)) > 1e-12
    hold on
    plot(xn,solnf(:,3),':k')
end
catch
end

% Set x axis ticks to show adapted mesh
ax = gca;
ax.XTick = xx;
ax.XTickLabel = {};
str = '0 ';
for j = 1:length(xx)-2
    if j ~= find(xx == (xx(1)+xx(end))/2)-1
        str = [str; '  '];
    else
        str_mid = num2str((xx(1)+xx(end))/2);
        str = [str; str_mid(2:3)];
    end
end
str = [str; '1 '];
ax.XTickLabel = str;
% ax.TickDir = 'out';
ax.TickLength = [0.02 0.02];

end