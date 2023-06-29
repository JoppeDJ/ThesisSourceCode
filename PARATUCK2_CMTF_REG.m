function [Wres, D2res, Vtres, D1res, Ztres, Htres, cD1res, cD2res] = PARATUCK2_CMTF_REG(Jac, F, bf1, bf1d, bf2, bf2d, r1, r2, samples)
    
    % Sets the seed for reproducibility of results
    rng(12422);

    [I, J, K] = size(Jac);
    d1 = length(bf1d);
    d2 = length(bf2d);

    lambda =  0.001;
    lambda1 = 0.0; % niet gebruikt
    lambda2 = 0.0; % niet gebruikt
    
    cD1 = zeros(r1*d1,1);
    cD2 = zeros(r2*(d2+1),1);
    
    W = randn(I, r2);
    D2 = randn(K, r2);
    Vt = randn(r2, r1);
    D1 = randn(K, r1);
 
    tic

    Zt = updateZt(Jac, W, D1, Vt, D2, I, J, K, r1);
    
    %%toc
    %%tic

    D1 = updateD1(Jac, W, Vt, Zt, D2, K, I, r1);
    
    %%toc
    %%tic

    Vt = updateVt(Jac, W, D1, D2, Zt, I, J, K, r1, r2); 
    
    %%toc
    %%tic

    D2 = updateD2(Jac, W, Vt, Zt, D1, K, I, r1, r2);
    
    %%toc
    %%tic

    Ht = updateH(F,W);
    
    %%toc
    %%tic

    W = updateW(Jac, F, Ht, D2, Vt, D1, Zt, I, J, K, r2, lambda);
    
    toc

    lastError = 1e5;
    minError = 1;

    for i=1:25
       
        Zt = updateZt(Jac, W, D1, Vt, D2, I, J, K, r1);

        D1 = updateD1(Jac, W, Vt, Zt, D2, K, I, r1);
        
        % Projectie strategie
        %tic

        [cD1, D1] = update_cD1(D1, Zt, samples, bf1d, K, r1, d1, lambda1);
        
        %toc

        Vt = updateVt(Jac, W, D1, D2, Zt, I, J, K, r1, r2);        

        D2 = updateD2(Jac, W, Vt, Zt, D1, K, I, r1, r2);
        
        Ht = updateH(F,W);
        
        %tic

        % Projectie strategie
        [cD2, D2, Ht] = update_cD2(Ht, D2, Vt, Zt, samples, cD1, bf1, bf2, bf2d, K, r2, d2, lambda2, lambda);
        
        %toc

        W = updateW(Jac, F, Ht, D2, Vt, D1, Zt, I, J, K, r2, lambda);
        

        apprJac = zeros(I, J, K);
        for j=1:K
            apprJac(:,:,j) = ...
                W * diag(D2(j,:)) * Vt * diag(D1(j,:)) * Zt;
        end
        
        error = frob(Jac - apprJac)^2 / frob(Jac)^2;

        i
        Jerror = error
        Ferror = frob(F-W*Ht)^2 / frob(F)^2

        if(Ferror < minError)
            Wres = W;
            D2res = D2;
            Vtres = Vt;
            D1res = D1; 
            Ztres = Zt;
            Htres = Ht;
            cD1res = cD1; 
            cD2res = cD2;

            minError = Ferror;
        end
        
        if(mod(i,5) == 0 && lambda < 1)
             lambda = lambda * 3;
        end
        lastError = error;
    end

    minError
end

function [Ht] = updateH(F,W)
    Ht = W \ F;
end

function [W] = updateW(X, F, Ht, D2, Vt, D1, Zt, I, J, K, r2, lambda)
    %unfoldX = zeros(I, J*K);
    %for i=1:K
    %    unfoldX(:, (i-1) * J + 1 : i * J) = X(:,:,i);
    %end

    Fw = zeros(r2, J*K);
    for i=1:K
        Fw(:,(i-1) * J + 1 : i * J) = ...
            diag(D2(i,:)) * Vt * diag(D1(i,:)) * Zt;
    end

    W = [tens2mat(X,1,[2 3]) lambda*F] / [Fw lambda*Ht];
end

function [D2] = updateD2(X, W, Vt, Zt, D1, K, I, r1, r2)
%     D2 = zeros(K, r2);
%     for k=1:K
%         Fk = Zt' * diag(D1(k,:)) * Vt';
%         xk = reshape(X(:,:,k), numel(X(:,:,k)), 1);
% 
%         D2(k,:) = (kr(Fk, W) \ xk)';
%     end

    rowList = zeros(K * r1 * r2 * I,1);
    colList = zeros(K * r1 * r2 * I,1);
    valList = zeros(K * r1 * r2 * I,1);
    currentIdx = 1;

    Jtest = zeros(K * I * r1,1);
    
    for i=1:K
        
        B = Vt * diag(D1(i,:));
        C = kr(B', W);

        for j=1: I*r1
            for k=1:r2
                rowList(currentIdx) = (i-1) * r1 * I + j;
                colList(currentIdx) = (i-1) * r2 + k;
                valList(currentIdx) = C(j,k);

                currentIdx = currentIdx + 1;
            end
        end

        temp = X(:,:,i) * pinv(Zt);
        Jtest((i-1) * I * r1 + 1 : i * I * r1, :) = ...
            reshape(temp, [],1);
    end
    
    Z = sparse(rowList, colList, valList);

    d2 = Z \ Jtest; 

    D2 = reshape(d2, r2, K)'; 
end

function [Vt] = updateVt(X, W, D1, D2, Zt, I, J, K, r1, r2)
%     Z = zeros(I*J*K, r1 * r2);
%     for i=1:K
%        Z((i-1) * I * J + 1 : i * I * J, :) = ...
%            kron(Zt'*diag(D1(i,:)), W*diag(D2(i,:)));
%     end
% 
%     vt = Z \ tens2vec(X,1:3); 
% 
%     Vt = reshape(vt, r2, r1);

    
    rowList = zeros(K*r1*I*r2,1);
    colList = zeros(K*r1*I*r2,1);
    valList = zeros(K*r1*I*r2,1);

    currentIdx = 1;

    Jtest = zeros(K * r1 * I, 1);
    for i=1:K

        C = W*diag(D2(i,:));

        for j=1:r1
            for k=1:I
                for l=1:r2
                    rowList(currentIdx) = (i-1) * r1* I + (j-1) * I + k;
                    colList(currentIdx) = (j-1) * r2 + l;
                    valList(currentIdx) = D1(i,j) * C(k,l);

                    currentIdx = currentIdx + 1;
                end
            end
        end

        temp = X(:,:,i) * pinv(Zt);
        Jtest((i-1)*r1*I + 1 : i *r1*I ,:) = ...
            reshape(temp, [], 1);
    end
    
    Z = sparse(rowList, colList, valList);

    vt = Z \ Jtest; 

    Vt = reshape(vt, r2, r1);
end

function [D1] = updateD1(X, W, Vt, Zt, D2, K, I, r1)
%     D1 = zeros(K, r1);
%     for k=1:K
%         Fk = W * diag(D2(k,:)) * Vt;
%         xk = reshape(X(:,:,k), numel(X(:,:,k)), 1);
% 
%         D1(k,:) = (kr(Zt', Fk) \ xk)';
%     end
    
    rowList = zeros(K * r1 * r1 * I,1);
    colList = zeros(K * r1 * r1 * I,1);
    valList = zeros(K * r1 * r1 * I,1);
    currentIdx = 1;

    Jtest = zeros(K * I * r1,1);
    
    Ir1 = eye(r1);
    for i=1:K
        
        A = W * diag(D2(i,:)) * Vt;
        C = kr(Ir1, A);

        for j=1: I*r1
            for k=1:r1
                rowList(currentIdx) = (i-1) * r1 * I + j;
                colList(currentIdx) = (i-1) * r1 + k;
                valList(currentIdx) = C(j,k);

                currentIdx = currentIdx + 1;
            end
        end

        temp = X(:,:,i) * pinv(Zt);
        Jtest((i-1) * I * r1 + 1 : i * I * r1, :) = ...
            reshape(temp, [],1);
    end
    
    Z = sparse(rowList, colList, valList);

    d1 = Z \ Jtest; 

    D1 = reshape(d1, r1, K)'; 
end

function [Zt] = updateZt(X, W, D1, Vt, D2, I, J, K, r1)

    F = zeros(I * K, r1);
    for i=1:K
        F((i-1) * I + 1 : i * I,:) = ...
            W*diag(D2(i,:))*Vt*diag(D1(i,:));
    end

    Zt = F \ tens2mat(X, [1 3], 2);
end

function [cD1, D1] = update_cD1(D1, Zt, samples, bf1d, K, r1, d1, lambda1)
    % Projectiestrategie
    rowList = zeros(d1*K*r1,1);
    colList = zeros(d1*K*r1,1);
    valList = zeros(d1*K*r1,1);
    currentIdx = 1;
    for l=1:r1
        for j=1:K
            xl = Zt(l,:) * samples(:,j);

            for k=1:d1
                func = bf1d{k};
                grad = func(xl);

                rowList(currentIdx) = ((l-1)*K) + j;
                colList(currentIdx) = ((l-1)*d1) + k;
                valList(currentIdx) = grad;

                currentIdx = currentIdx + 1;
            end
        end
    end
    
    X = sparse(rowList, colList, valList);

    cD1 = X \ reshape(D1, numel(D1), 1);

    for l=1:r1
        D1(:,l) = X(((l-1)*K) + 1:l * K, ((l-1)*d1) + 1:l * d1) * cD1((l-1) * d1 + 1: l * d1);
    end
end

function [cD2, D2, Ht] = update_cD2(Ht, D2, Vt, Zt, samples, cD1, bf1, bf2, bf2d, K, r2, d2, lambda2, lambda)
    % Projectiestrategie
    rowListX = zeros((d2+1)*K*r2,1);
    colListX = zeros((d2+1)*K*r2,1);
    valListX = zeros((d2+1)*K*r2,1);

    rowListY = zeros((d2+1)*K*r2,1);
    colListY = zeros((d2+1)*K*r2,1);
    valListY = zeros((d2+1)*K*r2,1);

    currentIdx = 1;
    for l=1:r2
        for j=1:K
            inner_sample = Zt * samples(:,j);
            sample = applyFlexibleFunctions(inner_sample, cD1, bf1);

            xl = Vt(l,:) * sample;
            
            rowListX(currentIdx) = ((l-1)*K) + j;
            colListX(currentIdx) = ((l-1)*(d2+1)) + 1;
            valListX(currentIdx) = 0;

            rowListY(currentIdx) = ((l-1)*K) + j;
            colListY(currentIdx) = ((l-1)*(d2+1)) + 1;
            valListY(currentIdx) = 1;

            currentIdx = currentIdx + 1;
            for k=2:(d2+1)
                funcf = bf2{k-1};
                funcd = bf2d{k-1};

                grad = funcd(xl);
                f = funcf(xl);
                
                rowListX(currentIdx) = ((l-1)*K) + j;
                colListX(currentIdx) = ((l-1)*(d2+1)) + k;
                valListX(currentIdx) = grad;

                rowListY(currentIdx) = ((l-1)*K) + j;
                colListY(currentIdx) = ((l-1)*(d2+1)) + k;
                valListY(currentIdx) = f;

                currentIdx = currentIdx + 1;
            end
        end
    end
    
    X = sparse(rowListX, colListX, valListX);
    Y = sparse(rowListY, colListY, valListY);

    cD2 = [X; lambda * Y] \ ...
        [reshape(D2, numel(D2), 1); lambda * reshape(Ht', numel(Ht), 1)];

    for l=1:r2
        D2(:,l) = X(((l-1)*K) + 1:l * K, ((l-1)*(d2+1)) + 1:l * (d2+1)) * cD2((l-1) * (d2+1) + 1: l * (d2+1));
        Ht(l,:) = (Y(((l-1)*K) + 1:l * K, ((l-1)*(d2+1)) + 1:l * (d2+1)) * cD2((l-1) * (d2+1) + 1: l * (d2+1)))';
    end
end

function [result] = applyFlexibleFunctions(inputVec, c, bf)
    d = length(bf);
    n = length(inputVec);
    result = zeros(n,1);

    for i=1:n
        val = 0;
        for j=1:d
            x = inputVec(i);
            func = bf{j};
            f = func(x);

            val = val + c((i-1)*d + j) * f;
        end

        result(i) = val;
    end
end
