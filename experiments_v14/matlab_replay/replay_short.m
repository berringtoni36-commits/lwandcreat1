% Independent MATLAB RFF + sequential MCC + counted signed-sort prune probe.
% This short fixture contains a FORCED numerical actuator ramp, not a selector
% acceptance decision. The source selector is pinned in the fixture metadata.
set(groot, 'DefaultFigureVisible', 'off');
log_path = getenv('N14_REPLAY_LOG');
if ~isempty(log_path)
    diary(log_path);
end
f = load(getenv('N14_REPLAY_FIXTURE'));
result_path = getenv('N14_REPLAY_RESULT');
assert(~isempty(result_path));
x = f.x(:); d = f.d(:); v = f.v(:);
omega = f.omega;
phase_rff = f.rff_phase(:);
s_path = f.S_PATH(:);
N = numel(x); D = size(omega,1); M = size(omega,2);
assert(D == 500 && M == 20 && size(f.initial_A,1) == 25 && size(f.initial_B,1) == 20);
mu = double(f.mu); sigma = double(f.sigma); eps_value = double(f.eps);
proposal = double(f.proposal_index_zero_based);
train_samples = double(f.train_samples);
validation_samples = double(f.validation_samples);
ramp_samples = double(f.ramp_samples);
ramp_begin = proposal + 1 + train_samples + validation_samples; % zero-based
A = f.initial_A; B = f.initial_B; ybuf = zeros(4,1);
Ac = []; Bc = []; ycbuf = zeros(4,1); perm = [];
xb = zeros(M,1); zh = zeros(4,D); actual_ybuf = zeros(4,1);
active_drive = zeros(N,1); active_ys = zeros(N,1);
candidate_drive = nan(N,1); candidate_ys = nan(N,1);
actual_drive = zeros(N,1); actual_ys = zeros(N,1);
actual_error = zeros(N,1); actual_anr = zeros(N,1); gamma_forced = zeros(N,1);
check_indices = double(f.check_indices_one_based(:));
z_checks = zeros(numel(check_indices),D); q_checks = zeros(numel(check_indices),D);
A_checks = zeros(numel(check_indices),25,4); B_checks = zeros(numel(check_indices),20,4);
check_cursor = 1;
Ad = 0; Ae = 0;
prune_mults = nan; jacobi_sweeps = nan; truncation = nan;
candidate_A_at_proposal = []; candidate_B_at_proposal = [];
R4_A_at_proposal = []; R4_B_at_proposal = []; R4_ybuf_at_proposal = [];
for n = 1:N
    xb = [x(n); xb(1:end-1)];
    z = sqrt(2/D)*cos(omega*xb+phase_rff);
    zh = [z.'; zh(1:3,:)];
    q = zh(3,:).' + 0.5*zh(4,:).';
    dv = d(n)+v(n);
    [A,B,ybuf,ya,ysa] = factor_step(A,B,ybuf,z,q,dv,mu,sigma,eps_value,s_path);
    active_drive(n) = ya; active_ys(n) = ysa;
    if ~isempty(Ac)
        [Ac,Bc,ycbuf,yc,ysc] = factor_step(Ac,Bc,ycbuf,z(perm),q(perm),dv,mu,sigma,eps_value,s_path);
        candidate_drive(n) = yc; candidate_ys(n) = ysc;
    end
    if n == proposal + 1
        R4_A_at_proposal = A; R4_B_at_proposal = B; R4_ybuf_at_proposal = ybuf;
        [Ac,Bc,perm,prune_mults,jacobi_sweeps,truncation] = counted_prune(A,B);
        candidate_A_at_proposal = Ac; candidate_B_at_proposal = Bc;
        ycbuf = ybuf;
    end
    zero_n = n-1;
    if zero_n < ramp_begin
        mix = 0;
    else
        mix = min(1,(zero_n-ramp_begin+1)/ramp_samples);
    end
    gamma_forced(n) = mix;
    y_actual = ya;
    if zero_n >= ramp_begin
        y_actual = (1-mix)*ya+mix*candidate_drive(n);
    end
    actual_ybuf = [y_actual; actual_ybuf(1:3)];
    ys = actual_ybuf.'*s_path;
    e = dv-ys;
    actual_drive(n) = y_actual; actual_ys(n) = ys; actual_error(n) = e;
    Ad = .999*Ad+.001*abs(d(n));
    Ae = .999*Ae+.001*abs(e);
    actual_anr(n) = 20*log10((Ae+1e-12)/(Ad+1e-12));
    if check_cursor <= numel(check_indices) && n == check_indices(check_cursor)
        z_checks(check_cursor,:) = z.';
        q_checks(check_cursor,:) = q.';
        A_checks(check_cursor,:,:) = A;
        B_checks(check_cursor,:,:) = B;
        check_cursor = check_cursor+1;
    end
end
assert(check_cursor == numel(check_indices)+1);
save(result_path, 'active_drive','active_ys','candidate_drive','candidate_ys', ...
    'actual_drive','actual_ys','actual_error','actual_anr','gamma_forced', ...
    'z_checks','q_checks','A_checks','B_checks','perm','prune_mults', ...
    'jacobi_sweeps','truncation','candidate_A_at_proposal', ...
    'candidate_B_at_proposal','R4_A_at_proposal','R4_B_at_proposal', ...
    'R4_ybuf_at_proposal','-v7');
fprintf('N14 MATLAB short replay: N=%d, proposal=%d, counted=%d, sweeps=%d, tail=%.12g\n', ...
    N, proposal, prune_mults, jacobi_sweeps, truncation);
if ~isempty(log_path)
    diary('off');
end


function [A,B,ybuf,y,ys] = factor_step(A,B,ybuf,z,q,dv,mu,sigma,eps_value,s_path)
    d2 = size(B,1); d1 = size(A,1);
    Z = reshape(z,d2,d1);
    Q = reshape(q,d2,d1);
    y = sum(B.*(Z*A),'all');
    ybuf = [y; ybuf(1:end-1)];
    ys = ybuf.'*s_path;
    UB = Q*A;
    e1 = dv-sum(B.*UB,'all');
    psi1 = e1*exp(-(e1*e1)/(2*sigma*sigma));
    nb = sum(UB.*UB,'all');
    B = B + ((mu/2)*psi1/(eps_value+nb))*UB;
    UA = Q.'*B;
    e2 = dv-sum(A.*UA,'all');
    psi2 = e2*exp(-(e2*e2)/(2*sigma*sigma));
    na = sum(UA.*UA,'all');
    A = A + ((mu/2)*psi2/(eps_value+na))*UA;
end


function [A2,B2,p,total,sweeps,trunc] = counted_prune(A,B)
    d1 = size(A,1); d2 = size(B,1); R = size(A,2);
    W = B*A.';
    w = reshape(W,[],1);
    [~,p] = sort(w,'ascend');
    Wp = reshape(w(p),d2,d1);
    [Q,Rmat,qr_mults] = qr_mgs2(Wp.');
    [U,s,V,sweeps,jacobi_mults] = jacobi_svd(Rmat);
    target = 2;
    scale = sqrt(s(1:target)).';
    A2 = (Q*U(:,1:target)).*scale;
    B2 = V(:,1:target).*scale;
    trunc = sqrt(sum(s(target+1:end).^2)/sum(s.^2));
    total = qr_mults+jacobi_mults + ...
        size(Q,1)*size(Q,2)*target + (d1+d2)*target + ...
        d1*d2*R + numel(s);
    assert(norm(Wp-B2*A2.','fro')/norm(Wp,'fro') < 1);
end


function [Q,R,count] = qr_mgs2(matrix)
    [m,n] = size(matrix);
    Q = zeros(m,n); R = zeros(n,n);
    frob = sqrt(sum(matrix(:).^2));
    count = m*n + 1;
    tol = 1e-13*max(1,frob);
    for j=1:n
        v = matrix(:,j);
        for pass=1:2
            for i=1:j-1
                coeff = Q(:,i).'*v;
                count = count+m;
                R(i,j) = R(i,j)+coeff;
                v = v-coeff*Q(:,i);
                count = count+m;
            end
        end
        norm_v = sqrt(v.'*v);
        count = count+m;
        if norm_v>tol
            R(j,j)=norm_v;
            Q(:,j)=v/norm_v;
        end
    end
end


function [left,singular,right,sweeps,count] = jacobi_svd(square)
    work = square;
    n = size(work,1);
    assert(size(work,2)==n);
    right = eye(n); count = 0; converged = false;
    tol = 1e-12;
    for sweep=1:32
        changed = false;
        for p=1:n
            for q=p+1:n
                cp=work(:,p); cq=work(:,q);
                app=cp.'*cp; aqq=cq.'*cq; apq=cp.'*cq;
                count=count+3*n+2;
                if abs(apq)<=tol*sqrt(app*aqq)
                    continue;
                end
                changed = true;
                tau=(aqq-app)/(2*apq);
                t=sign_nonzero(tau)/(abs(tau)+sqrt(1+tau*tau));
                c=1/sqrt(1+t*t); s=c*t;
                count=count+1+1+2;
                work(:,p)=c*cp-s*cq;
                work(:,q)=s*cp+c*cq;
                vp=right(:,p); vq=right(:,q);
                right(:,p)=c*vp-s*vq;
                right(:,q)=s*vp+c*vq;
                count=count+8*n;
            end
        end
        if ~changed
            converged = true;
            break;
        end
    end
    assert(converged,'Jacobi SVD did not converge');
    singular = sqrt(sum(work.*work,1));
    count=count+n*n;
    left=zeros(n,n);
    for j=1:n
        if singular(j)>1e-14
            left(:,j)=work(:,j)/singular(j);
        end
    end
    [singular,order]=sort(singular,'descend');
    singular=singular(:);
    left=left(:,order); right=right(:,order);
    sweeps=sweep;
end


function value = sign_nonzero(x)
    if x>=0
        value=1;
    else
        value=-1;
    end
end
