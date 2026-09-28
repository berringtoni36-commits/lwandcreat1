% Independently audit the physical output, ANR and every recorded decision.
% This does not rerun the factor adaptation or candidate initialization.
set(groot,'DefaultFigureVisible','off');
f = load(getenv('AKT_FIXTURE'));
T = numel(f.d);
Ad = 0; Ae = 0;
anrmax = 0; ysmax = 0; costmax = 0;
ybuf = zeros(1,4);
for n = 1:T
    ybuf(2:4) = ybuf(1:3);
    ybuf(1) = f.actuator_output(n);
    ys = ybuf(3) + .5*ybuf(4);
    ysmax = max(ysmax,abs(ys-f.ys_expected(n)));
    e = f.d(n)+f.v(n)-ys;
    Ad = .999*Ad+.001*abs(f.d(n));
    Ae = .999*Ae+.001*abs(e);
    anr = 20*log10((Ae+1e-12)/(Ad+1e-12));
    anrmax = max(anrmax,abs(anr-f.anr_expected(n)));
    costmax = max(costmax,abs(f.total_mults(n)-f.core_mults(n) ...
        -f.candidate_mults(n)-f.management_mults(n)));
end
wrong_decisions = 0;
for j = 1:numel(f.decision_sample)
    old_C = 12508+1655*f.decision_old_R(j);
    new_C = 12508+1655*f.decision_new_R(j);
    old_J = f.decision_old_loss(j)+f.lambda_cost*old_C/14504;
    new_J = f.decision_new_loss(j)+f.lambda_cost*new_C/14504;
    predicted = new_J < old_J-f.margin;
    wrong_decisions = wrong_decisions + (predicted ~= logical(f.decision_accepted(j)));
end
result = struct('max_secondary_error',ysmax,'max_anr_error',anrmax, ...
                'max_cost_sum_error',costmax,'wrong_decisions',wrong_decisions, ...
                'decisions',numel(f.decision_sample));
save(getenv('AKT_RESULT'),'result');
disp(result);
assert(ysmax<1e-5 && anrmax<1e-3 && costmax==0 && wrong_decisions==0);
