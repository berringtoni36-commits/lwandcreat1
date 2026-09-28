% Independent short-sequence MATLAB implementation of the v12 growth manager.
% Input arrays and initial factors come from probe.py; MATLAB computes every
% RFF feature, filtered regressor, factor update and structural decision.
set(groot, 'DefaultFigureVisible', 'off');
f = load(getenv('AKT_FIXTURE'));
f.warmup = double(f.warmup);
f.train_samples = double(f.train_samples);
f.validation_samples = double(f.validation_samples);
f.ramp_samples = double(f.ramp_samples);
f.cooldown = double(f.cooldown);
T = numel(f.x);
D = size(f.Om, 1);
M = size(f.Om, 2);
A = f.A0;
B = f.B0;
ya = zeros(1, 4);
actual_ybuf = zeros(1, 4);
Ac = [];
Bc = [];
yc = zeros(1, 4);
xb = zeros(M, 1);
zh = zeros(4, D);
z_trace = zeros(T, D);
q_trace = zeros(T, D);
trace = zeros(T, 10);
actual_ys = zeros(T, 1);
phase = 0; % idle=0, train=1, validate=2, ramp=3
age = 0;
next_probe = f.warmup;
validation_sum = [0, 0];
proposal_sample = NaN;
validation_start_sample = NaN;
decision_sample = NaN;
decision_accept = NaN;
old_loss = NaN;
new_loss = NaN;
old_score = NaN;
new_score = NaN;
switch_sample = NaN;
for n = 1:T
    % Reference and filtered-x RFF are computed here from the saved physical
    % reference and frequency/phase initialization, not imported z/q arrays.
    xb = [f.x(n); xb(1:end-1)];
    z = sqrt(2 / D) * cos(f.Om * xb + f.ph(:));
    zh = [z.'; zh(1:3, :)];
    q = zh(3, :) + 0.5 * zh(4, :);
    z_trace(n, :) = z.';
    q_trace(n, :) = q;
    dv = f.d(n) + f.v(n);
    phase_start = phase;
    old_r = size(A, 2);
    candidate_r = size(Ac, 2);
    [A, B, ya, ys_old, y_old] = factor_step(A, B, ya, z, q.', dv, f);
    y_actual = y_old;
    gamma = 0;
    if candidate_r > 0
        [Ac, Bc, yc, ys_new, y_new] = factor_step(Ac, Bc, yc, z, q.', dv, f);
        if phase_start == 1
            age = age + 1;
            if age >= f.train_samples
                phase = 2;
                age = 0;
                validation_start_sample = n - 1;
            end
        elseif phase_start == 2
            errors = [dv - ys_old, dv - ys_new];
            validation_sum = validation_sum - expm1(-0.5 * (errors / f.sigma).^2);
            age = age + 1;
            if age >= f.validation_samples
                losses = validation_sum / f.validation_samples;
                costs = [kron_cost(old_r, M, size(A, 1), size(B, 1)), ...
                    kron_cost(candidate_r, M, size(A, 1), size(B, 1))];
                cref = full_cost(D, M);
                scores = losses + f.lambda_cost * costs / cref;
                accepted = all(isfinite(scores)) && scores(2) < scores(1) - f.margin;
                decision_sample = n - 1;
                decision_accept = double(accepted);
                old_loss = losses(1);
                new_loss = losses(2);
                old_score = scores(1);
                new_score = scores(2);
                if accepted
                    phase = 3;
                    age = 0;
                else
                    Ac = [];
                    Bc = [];
                    phase = 0;
                    age = 0;
                    next_probe = n + f.cooldown;
                end
            end
        elseif phase_start == 3
            age = age + 1;
            gamma = age / f.ramp_samples;
            y_actual = (1 - gamma) * y_old + gamma * y_new;
            if age >= f.ramp_samples
                A = Ac;
                B = Bc;
                ya = yc;
                Ac = [];
                Bc = [];
                phase = 0;
                age = 0;
                next_probe = n + f.cooldown;
                switch_sample = n - 1;
            end
        end
    end
    actual_ybuf = [y_actual, actual_ybuf(1:3)];
    actual_ys(n) = actual_ybuf(3) + 0.5 * actual_ybuf(4);
    % Python's n is zero-based; its proposal condition n+1>=next_probe.
    if phase_start == 0 && n >= next_probe
        assert(isempty(Ac) && old_r == 1, 'Unexpected second proposal');
        Ac = [A, f.growth_a_at_proposal(:)];
        Bc = [B, zeros(size(B, 1), 1)];
        yc = ya;
        phase = 1;
        age = 0;
        validation_sum = [0, 0];
        proposal_sample = n - 1;
    end
    ca = NaN;
    cb = NaN;
    if ~isempty(Ac)
        ca = sum(Ac(:));
        cb = sum(Bc(:));
    end
    trace(n, :) = [old_r, candidate_r, y_actual, actual_ys(n), ...
        gamma, phase_start, sum(A(:)), sum(B(:)), ca, cb];
end

expected = f.trace_expected;
both_nan = isnan(expected) & isnan(trace);
finite_mask = isfinite(expected) & isfinite(trace);
assert(all(both_nan(:) | finite_mask(:)), 'Nonfinite trace mismatch');
delta = abs(trace - expected);
delta(~finite_mask) = 0;
column_max = max(delta, [], 1);
result = struct();
result.max_z_error = max(abs(z_trace(:) - f.z_expected(:)));
result.max_q_error = max(abs(q_trace(:) - f.q_expected(:)));
result.max_trace_error = max(delta(:));
result.max_trace_error_by_column = column_max;
result.max_actual_secondary_error = max(abs(actual_ys(:) - expected(:, 4)));
result.max_A_final_error = max(abs(A(:) - f.A_final(:)));
result.max_B_final_error = max(abs(B(:) - f.B_final(:)));
result.proposal_sample = proposal_sample;
result.validation_start_sample = validation_start_sample;
result.decision_sample = decision_sample;
result.decision_accept = decision_accept;
result.switch_sample = switch_sample;
result.old_loss_error = abs(old_loss - f.old_loss);
result.new_loss_error = abs(new_loss - f.new_loss);
result.old_score_error = abs(old_score - f.old_score);
result.new_score_error = abs(new_score - f.new_score);
result.structure_match = proposal_sample == f.proposal_sample && ...
    validation_start_sample == f.validation_start_sample && ...
    decision_sample == f.decision_sample && decision_accept == f.decision_accept;
if decision_accept
    result.structure_match = result.structure_match && switch_sample == f.switch_sample;
else
    result.structure_match = result.structure_match && ...
        isnan(switch_sample) && f.switch_sample == -1;
end
save(getenv('AKT_RESULT'), 'result', 'trace', 'z_trace', 'q_trace', ...
    'actual_ys', 'A', 'B');
disp(result);
assert(result.structure_match && result.max_trace_error < 1e-9 && ...
    result.max_z_error < 1e-12 && result.max_q_error < 1e-12 && ...
    result.max_A_final_error < 1e-9 && result.max_B_final_error < 1e-9, ...
    'Independent dynamic replay differs from Python');

function [A, B, ybuf, ys, y] = factor_step(A, B, ybuf, z, q, dv, f)
    Z = reshape(z, size(B, 1), size(A, 1));
    Q = reshape(q, size(B, 1), size(A, 1));
    y = sum(B .* (Z * A), 'all');
    ybuf = [y, ybuf(1:3)];
    ys = ybuf(3) + 0.5 * ybuf(4);
    UB = Q * A;
    e1 = dv - sum(B .* UB, 'all');
    psi1 = e1 * exp(-e1^2 / (2 * f.sigma^2));
    nb = sum(UB.^2, 'all');
    B = B + (f.mu / 2) * psi1 / (f.eps + nb) * UB;
    UA = Q.' * B;
    e2 = dv - sum(A .* UA, 'all');
    psi2 = e2 * exp(-e2^2 / (2 * f.sigma^2));
    na = sum(UA.^2, 'all');
    A = A + (f.mu / 2) * psi2 / (f.eps + na) * UA;
end

function count = kron_cost(R, M, D1, D2)
    D = D1 * D2; Ls = 4;
    count = D * M + D + D * Ls + (R * D + R * D2) + ...
        2 * R * D + 3 * R * (D1 + D2) + 8;
end

function count = full_cost(D, M)
    count = D * M + D + D * 4 + 4 * D + 4;
end
