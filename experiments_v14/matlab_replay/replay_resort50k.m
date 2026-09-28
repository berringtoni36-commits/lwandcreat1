% Independent N14 dev replay of periodic causal R2 signed-sort resort.
% MATLAB computes all controller outputs and states; no Python trace is loaded.
set(groot,'DefaultFigureVisible','off');
log_path=getenv('N14_REPLAY_LOG');
if ~isempty(log_path), diary(log_path); end
f=load(getenv('N14_REPLAY_FIXTURE'));
result_path=getenv('N14_REPLAY_RESULT');assert(~isempty(result_path));
x=f.x(:);d=f.d(:);v=f.v(:);omega=f.omega;ph=f.rff_phase(:);
s_path=f.S_PATH(:);T=numel(x);D=size(omega,1);M=size(omega,2);
assert(T==400000 && D==500 && M==20 && numel(d)==T && numel(v)==T);
D1=25;D2=20;LS=4;SHARED=D*M+D+D*LS;
mu=double(f.mu);TRAIN=double(f.train_samples);VALIDATE=double(f.validation_samples);
RAMP=double(f.ramp_samples);FIRST=double(f.first_propose);
INTERVAL=double(f.propose_interval);RESORT=double(f.resort_interval);
COOLDOWN=double(f.cooldown);assert(RESORT==50000);
SPECTRAL=double(f.spectral_threshold);DETECT_REL=double(f.detect_rel);
DETECT_HOLD=double(f.detect_hold);
active=struct('A',f.initial_A,'B',f.initial_B,'R',4,'ybuf',zeros(LS,1));
candidate=[];retiring=[];
active_perm=(1:D).';candidate_perm=[];retiring_perm=[];
phase=0;phase_end=-1;ramp_left=0;post_ramp_check=0;
next_prune=FIRST;cooldown_until=0;
slow=0;fast=0;hold=0;check_active=0;check_candidate=0;check_count=0;
actual_ybuf=zeros(LS,1);xb=zeros(M,1);zh=zeros(LS,D);Ad=0;Ae=0;
drive=zeros(T,1);physical_error=zeros(T,1);ys_actual=zeros(T,1);
anr=zeros(T,1);R=zeros(T,1);candidate_R=zeros(T,1);retiring_R=zeros(T,1);
resident_factor_coeffs=zeros(T,1);gamma=zeros(T,1);exp_evals=zeros(T,1);
cost_core=zeros(T,1);cost_candidate=zeros(T,1);cost_retiring=zeros(T,1);
cost_management=zeros(T,1);cost_reorder=zeros(T,1);cost_total=zeros(T,1);
post_ramp_output_error=nan(T,1);
proposal_n=[];proposal_from_R=[];proposal_reorder=[];proposal_tail=[];
proposal_sweeps=[];proposal_veto=[];
decision_n=[];decision_from_R=[];decision_accepted=[];
validation_ratio=[];validation_count=[];ramp_start=[];ramp_end=[];
for n=1:T
    zero_n=n-1;
    xb=[x(n);xb(1:end-1)];
    z=sqrt(2/D)*cos(omega*xb+ph);
    zh=[z.';zh(1:LS-1,:)];
    q=zh(3,:).'+0.5*zh(4,:).';
    executed_rank=active.R;
    executed_candidate=0;if ~isempty(candidate),executed_candidate=candidate.R;end
    executed_retiring=0;if ~isempty(retiring),executed_retiring=retiring.R;end
    want_active=~isempty(candidate)||~isempty(retiring)||post_ramp_check>0;
    [active,ya,ys_a]=branch_output(active,z(active_perm),s_path,want_active);
    cost_core(n)=kron_cost(executed_rank,D1,D2,M,LS);
    ys_c=0;yc=0;
    if ~isempty(candidate)
        [candidate,yc,ys_c]=branch_output(candidate,z(candidate_perm),s_path,true);
        cost_candidate(n)=kron_cost(executed_candidate,D1,D2,M,LS)-SHARED+4;
    end
    yr=0;
    if ~isempty(retiring)
        [retiring,yr,~]=branch_output(retiring,z(retiring_perm),s_path,true);
        cost_retiring(n)=kron_cost(executed_retiring,D1,D2,M,LS)-SHARED+4;
    end
    mix=1;
    if ~isempty(retiring)
        mix=(RAMP-ramp_left+1)/RAMP;
    end
    gamma(n)=mix;
    yout=ya;
    if ~isempty(retiring)
        yout=mix*ya+(1-mix)*yr;
    end
    actual_ybuf=[yout;actual_ybuf(1:LS-1)];
    ys=actual_ybuf.'*s_path;
    ys_actual(n)=ys;
    e=d(n)-ys+v(n);
    dv=e+ys;
    assert(abs(dv-(d(n)+v(n)))<1e-10);
    if post_ramp_check>0
        post_ramp_output_error(n)=ys_a-ys;
        post_ramp_check=post_ramp_check-1;
    end
    active=factor_update(active,q(active_perm),dv,mu,D1,D2);
    exp_evals(n)=exp_evals(n)+2;
    if ~isempty(candidate)
        candidate=factor_update(candidate,q(candidate_perm),dv,mu,D1,D2);
        exp_evals(n)=exp_evals(n)+2;
    end
    if ~isempty(retiring)
        retiring=factor_update(retiring,q(retiring_perm),dv,mu,D1,D2);
        exp_evals(n)=exp_evals(n)+2;
    end
    drive(n)=yout;physical_error(n)=e;
    Ad=.999*Ad+.001*abs(d(n));Ae=.999*Ae+.001*abs(e);
    anr(n)=20*log10((Ae+1e-12)/(Ad+1e-12));
    robust=abs(e);
    if slow>0
        robust=min(abs(e),3*max(slow,1e-3));
    end
    fast=.98*fast+.02*robust;
    if fast>slow*(1+DETECT_REL) && zero_n>1000
        hold=hold+1;
    else
        hold=0;
    end
    if hold<DETECT_HOLD
        slow=.9995*slow+.0005*robust;
    end
    cost_management(n)=12;
    if ~isempty(candidate)||~isempty(retiring)||~isnan(post_ramp_output_error(n))
        cost_management(n)=cost_management(n)+4;
    end
    if ~isempty(retiring)
        cost_management(n)=cost_management(n)+2;
        ramp_left=ramp_left-1;
        if ramp_left==0
            active.ybuf=actual_ybuf;
            post_ramp_check=3;
            retiring=[];retiring_perm=[];
        end
    end
    if ~isempty(candidate)
        if phase==2 % validation uses causal reconstructed dv, never d directly
            cap=max(3*slow,1e-3);
            check_active=check_active+min(abs(dv-ys_a),cap);
            check_candidate=check_candidate+min(abs(dv-ys_c),cap);
            check_count=check_count+1;
            cost_management(n)=cost_management(n)+4;
        end
        if zero_n>=phase_end
            if phase==1
                phase=2;phase_end=zero_n+VALIDATE;
                check_active=0;check_candidate=0;check_count=0;
            else
                ratio=check_candidate/max(check_active,1e-12);
                accepted=ratio<=10^(.5/20);
                decision_n(end+1,1)=zero_n;
                decision_from_R(end+1,1)=active.R;
                decision_accepted(end+1,1)=double(accepted);
                validation_ratio(end+1,1)=ratio;
                validation_count(end+1,1)=check_count;
                this_ramp_start=nan;this_ramp_end=nan;
                if accepted
                    retiring=active;retiring_perm=active_perm;
                    ramp_left=RAMP;
                    this_ramp_start=zero_n+1;this_ramp_end=zero_n+RAMP;
                    active=candidate;active_perm=candidate_perm;
                    active.ybuf=actual_ybuf;
                    next_prune=zero_n+RESORT;
                end
                ramp_start(end+1,1)=this_ramp_start;
                ramp_end(end+1,1)=this_ramp_end;
                candidate=[];candidate_perm=[];phase=0;
                cooldown_until=zero_n+COOLDOWN;hold=0;
            end
        end
    end
    if isempty(candidate)&&isempty(retiring)&&zero_n>=cooldown_until
        if zero_n>=next_prune && (active.R>2 || (RESORT>0 && active.R==2))
            old=active.R;
            [candidate,candidate_perm,rc,sweeps,trunc]=n14_counted_prune(active,active_perm);
            next_prune=T; % once arm
            cost_reorder(n)=cost_reorder(n)+rc;
            cost_management(n)=cost_management(n)+2*old*(D1+D2);
            proposal_n(end+1,1)=zero_n;
            proposal_from_R(end+1,1)=old;
            proposal_reorder(end+1,1)=rc;
            proposal_tail(end+1,1)=trunc;
            proposal_sweeps(end+1,1)=sweeps;
            proposal_veto(end+1,1)=double(isempty(candidate)||trunc>SPECTRAL);
            if isempty(candidate)||trunc>SPECTRAL
                candidate=[];candidate_perm=[];
                cooldown_until=zero_n+COOLDOWN;
            else
                phase=1;phase_end=zero_n+TRAIN;
            end
            hold=0;
        end
    end
    R(n)=executed_rank;candidate_R(n)=executed_candidate;
    retiring_R(n)=executed_retiring;
    resident_factor_coeffs(n)=(executed_rank+executed_candidate+executed_retiring)*(D1+D2);
    cost_total(n)=cost_core(n)+cost_candidate(n)+cost_retiring(n)+ ...
        cost_management(n)+cost_reorder(n)+4;
end
physical_fir_max_abs=max(abs(filter(s_path,1,drive)-(d+v-physical_error)));
valid_post=post_ramp_output_error(~isnan(post_ramp_output_error));
post_ramp_max_abs=0;
if ~isempty(valid_post),post_ramp_max_abs=max(abs(valid_post));end
assert(post_ramp_max_abs<1e-10);
assert(numel(proposal_n)==8 && numel(decision_n)==7);
assert(all(decision_accepted==1) && all(proposal_veto==0));
assert(~isempty(candidate) && phase==1); % eighth proposal still training at T
segment_anr=zeros(4,1);segment_mults=zeros(4,1);
for j=1:4
    lo=(j-1)*100000+1;hi=j*100000;
    segment_anr(j)=mean(anr(hi-4999:hi));
    segment_mults(j)=mean(cost_total(lo:hi));
end
mean_mults=mean(cost_total);
save(result_path,'drive','physical_error','ys_actual','anr','R','candidate_R', ...
    'retiring_R','resident_factor_coeffs','gamma','exp_evals','cost_core', ...
    'cost_candidate','cost_retiring','cost_management','cost_reorder', ...
    'cost_total','post_ramp_output_error','proposal_n','proposal_from_R', ...
    'proposal_reorder','proposal_tail','proposal_sweeps','proposal_veto', ...
    'decision_n','decision_from_R','decision_accepted', ...
    'validation_ratio','validation_count','ramp_start','ramp_end', ...
    'segment_anr','segment_mults','mean_mults','physical_fir_max_abs', ...
    'post_ramp_max_abs','-v7');
fprintf('N14 resort MATLAB replay: T=%d proposals=%d decisions=%d pending=%d mean_cost=%.8f physical_gap=%.3g\n', ...
    T,numel(proposal_n),numel(decision_n),~isempty(candidate),mean_mults,physical_fir_max_abs);
if ~isempty(log_path),diary('off');end

function [ctrl,y,ys] = branch_output(ctrl,z,s_path,want_secondary)
Z=reshape(z,size(ctrl.B,1),size(ctrl.A,1));
y=sum(ctrl.B.*(Z*ctrl.A),'all');
ctrl.ybuf=[y;ctrl.ybuf(1:end-1)];
ys=nan;
if want_secondary,ys=ctrl.ybuf.'*s_path;end
end

function ctrl = factor_update(ctrl,q,dv,mu,d1,d2)
Q=reshape(q,d2,d1);A=ctrl.A;B=ctrl.B;
UB=Q*A;e1=dv-sum(B.*UB,'all');
psi1=e1*exp(-(e1*e1)/(2*2*2));nb=sum(UB.*UB,'all');
B=B+((mu/2)*psi1/(1e-8+nb))*UB;
UA=Q.'*B;e2=dv-sum(A.*UA,'all');
psi2=e2*exp(-(e2*e2)/(2*2*2));na=sum(UA.*UA,'all');
A=A+((mu/2)*psi2/(1e-8+na))*UA;
ctrl.A=A;ctrl.B=B;
end

function cost = kron_cost(rank,d1,d2,M,LS)
D=d1*d2;
cost=D*M+D+D*LS+(rank*D+rank*d2)+2*rank*D+3*rank*(d1+d2)+8;
end
