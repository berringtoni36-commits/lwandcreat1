function [candidate,new_perm,total,sweeps,trunc] = n14_counted_prune(active,old_perm)
% MATLAB implementation of frozen 60F6 signed-sort QR/Jacobi prune + ledger.
d1=size(active.A,1); d2=size(active.B,1); old_R=size(active.A,2); D=d1*d2;
W=active.B*active.A.'; w=reshape(W,[],1);
[~,p]=sort(w,'ascend');
new_perm=old_perm(p);
Wp=reshape(w(p),d2,d1);
[Q,R,qr_mults]=qr_mgs2(Wp.');
[U,s,V,sweeps,jacobi_mults]=jacobi_svd(R);
target=2; scale=sqrt(s(1:target)).';
A=(Q*U(:,1:target)).*scale;
B=V(:,1:target).*scale;
trunc=sqrt(sum(s(target+1:end).^2)/sum(s.^2));
candidate=active;
candidate.A=A; candidate.B=B; candidate.R=target;
candidate.ybuf=active.ybuf;
total=qr_mults+jacobi_mults + ...
    size(Q,1)*size(Q,2)*target + (d1+d2)*target + ...
    D*old_R + numel(s);
assert(norm(Wp-B*A.','fro')/norm(Wp,'fro')<1);
end

function [Q,R,count] = qr_mgs2(matrix)
[m,n]=size(matrix); Q=zeros(m,n);R=zeros(n,n);
frob=sqrt(sum(matrix(:).^2));count=m*n+1;
tol=1e-13*max(1,frob);
for j=1:n
    v=matrix(:,j);
    for pass=1:2
        for i=1:j-1
            coeff=Q(:,i).'*v;count=count+m;
            R(i,j)=R(i,j)+coeff;
            v=v-coeff*Q(:,i);count=count+m;
        end
    end
    nv=sqrt(v.'*v);count=count+m;
    if nv>tol
        R(j,j)=nv;Q(:,j)=v/nv;
    end
end
end

function [left,singular,right,sweeps,count] = jacobi_svd(square)
work=square;n=size(work,1);assert(size(work,2)==n);
right=eye(n);count=0;tol=1e-12;converged=false;
for sweep=1:32
    changed=false;
    for p=1:n
        for q=p+1:n
            cp=work(:,p);cq=work(:,q);
            app=cp.'*cp;aqq=cq.'*cq;apq=cp.'*cq;
            count=count+3*n+2;
            if abs(apq)<=tol*sqrt(app*aqq)
                continue;
            end
            changed=true;
            tau=(aqq-app)/(2*apq);
            if tau>=0
                sign_tau=1;
            else
                sign_tau=-1;
            end
            t=sign_tau/(abs(tau)+sqrt(1+tau*tau));
            c=1/sqrt(1+t*t);s=c*t;count=count+4;
            work(:,p)=c*cp-s*cq;
            work(:,q)=s*cp+c*cq;
            vp=right(:,p);vq=right(:,q);
            right(:,p)=c*vp-s*vq;
            right(:,q)=s*vp+c*vq;count=count+8*n;
        end
    end
    if ~changed
        converged=true;break;
    end
end
assert(converged,'Jacobi SVD did not converge');
singular=sqrt(sum(work.*work,1));count=count+n*n;
left=zeros(n,n);
for j=1:n
    if singular(j)>1e-14
        left(:,j)=work(:,j)/singular(j);
    end
end
[singular,order]=sort(singular,'descend');singular=singular(:);
left=left(:,order);right=right(:,order);sweeps=sweep;
end
