% Independent MATLAB replay of v10/v12 fixed R=4 factor updates.
set(groot,'DefaultFigureVisible','off');
fixture = getenv('AKT_FIXTURE');
result_path = getenv('AKT_RESULT');
assert(~isempty(fixture) && ~isempty(result_path));
f = load(fixture);
T = numel(f.x);
A = f.A0; B = f.B0;
xb = zeros(1,20);
zh = zeros(4,500);
ybuf = zeros(1,4);
zmax = 0; qmax = 0; ymax = 0; ysmax = 0;
for n = 1:T
    xb(2:end) = xb(1:end-1);
    xb(1) = f.x(n);
    z = sqrt(2/500) * cos(f.Om * xb.' + f.ph(:));
    zh(2:4,:) = zh(1:3,:);
    zh(1,:) = z.';
    q = zh(3,:).' + 0.5 * zh(4,:).';
    zmax = max(zmax, max(abs(z - f.z_expected(n,:).')));
    qmax = max(qmax, max(abs(q - f.q_expected(n,:).')));
    Z = reshape(z,20,25);
    Q = reshape(q,20,25);
    y = sum(B .* (Z*A),'all');
    ybuf(2:4) = ybuf(1:3);
    ybuf(1) = y;
    ys = ybuf(3) + .5*ybuf(4);
    ymax = max(ymax, abs(y - f.y_expected(n)));
    ysmax = max(ysmax, abs(ys - f.ys_expected(n)));
    UB = Q*A;
    e1 = f.d(n)+f.v(n)-sum(B.*UB,'all');
    psi1 = e1*exp(-e1^2/8);
    B = B + (.1*psi1/(1e-8+sum(UB.^2,'all')))*UB;
    UA = Q.'*B;
    e2 = f.d(n)+f.v(n)-sum(A.*UA,'all');
    psi2 = e2*exp(-e2^2/8);
    A = A + (.1*psi2/(1e-8+sum(UA.^2,'all')))*UA;
end
result = struct('max_z_error',zmax,'max_q_error',qmax, ...
                'max_y_error',ymax,'max_ys_error',ysmax, ...
                'max_A_error',max(abs(A-f.A_final),[],'all'), ...
                'max_B_error',max(abs(B-f.B_final),[],'all'));
save(result_path,'result');
disp(result);
assert(max(cell2mat(struct2cell(result))) < 1e-8);
