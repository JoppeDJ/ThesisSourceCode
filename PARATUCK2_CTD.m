function [Wres, D2res, Vtres, D1res, Ztres, cD1res, cD2res] = PARATUCK2_CTD(Jac, bf1, bf1d, bf2d, r1, r2, samples)
%PARATUCK2 Computes PARATUCK2 decomposition of given three-way tensor
%taking into account the neural network structure.
    
    % Sets the seed for reproducibility of results
    rng(58162);

    [I, J, K] = size(Jac);
    d1 = length(bf1d);
    d2 = length(bf2d);
    
    cD1 = zeros(r1*d1,1);
    cD2 = zeros(r2*d2,1);
    
    W =randn(I, r2);
    D2 = randn(K, r2);
    Vt = randn(r2, r1);
    D1 = randn(K, r1);
    Zt = randn(r1, J);

    Zt = updateZt(Jac, W, D1, Vt, D2, I, J, K, r1);

    D1 = updateD1(Jac, W, Vt, Zt, D2, K, I, r1);
    
    Vt = updateVt(Jac, W, D1, D2, Zt, I, J, K, r1, r2);        

    D2 = updateD2(Jac, W, Vt, Zt, D1, K, I, r1, r2);

    W = updateW(Jac, D2, Vt, D1, Zt, I, J, K, r2);

    lastError = 1e5;
    minError = 1;

    iterations = 0;
    for i=1:25
       
        Zt = updateZt(Jac, W, D1, Vt, D2, I, J, K, r1);

        D1 = updateD1(Jac, W, Vt, Zt, D2, K, I, r1);
        
        % Projectie strategie
        [cD1, D1] = update_cD1(D1, Zt, samples, bf1d, K, r1, d1);

        Vt = updateVt(Jac, W, D1, D2, Zt, I, J, K, r1, r2);        

        D2 = updateD2(Jac, W, Vt, Zt, D1, K, I, r1, r2);
        
        % Projectie strategie
        [cD2, D2] = update_cD2(D2, Vt, Zt, samples, cD1, bf1, bf2d, K, r2, d2);
        
        W = updateW(Jac, D2, Vt, D1, Zt, I, J, K, r2);
    
        apprJac = zeros(I, J, K);
        for j=1:K
            apprJac(:,:,j) = ...
                W * diag(D2(j,:)) * Vt * diag(D1(j,:)) * Zt;
        end
        
        i
        Jerror = frob(Jac - apprJac)^2 / frob(Jac)^2

        if(Jerror < minError)
            Wres = W;
            D2res = D2;
            Vtres = Vt;
            D1res = D1; 
            Ztres = Zt;
            cD1res = cD1; 
            cD2res = cD2;

            minError = Jerror;
        end

%         if(Jerror < 0.0001 || abs(Jerror-lastError) < 0.000005)
%             break
%         end

        lastError = Jerror;
        iterations = iterations + 1;
    end
    minError
end

% Optimalisatie mogelijk: element per element output berekenen i.p.v.
% in 1 keer zoals nu gebeurd.
function [W] = updateW(X, D2, Vt, D1, Zt, I, J, K, r2)
    unfoldX = zeros(I, J*K);
    for i=1:K
        unfoldX(:, (i-1) * J + 1 : i * J) = X(:,:,i);
    end

    F = zeros(r2, J*K);
    for i=1:K
        F(:,(i-1) * J + 1 : i * J) = ...
            diag(D2(i,:)) * Vt * diag(D1(i,:)) * Zt;
    end

    W = unfoldX / F; % Paper gebruikt F^T (lijkt verkeerd)
end

function [D2] = updateD2(X, W, Vt, Zt, D1, K, I, r1, r2)
%     D2 = zeros(K, r2);
%     for k=1:K
%         Fk = Zt' * diag(D1(k,:)) * Vt';
%         xk = reshape(X(:,:,k), numel(X(:,:,k)), 1);
% 
%         D2(k,:) = (kr(Fk, W) \ xk)'; % transpose van kathri-rao nodig (?)
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

% Optimalisatie mogelijk: element per element output berekenen i.p.v.
% in 1 keer zoals nu gebeurd.
function [Vt] = updateVt(X, W, D1, D2, Zt, I, J, K, r1, r2)
%     x = zeros(I*J*K, 1);
%     for k=1:K
%         x((k-1) * I * J + 1 : k * I * J, :) = reshape(X(:,:,k), 1, numel(X(:,:,k)));
%     end
% 
%     Z = zeros(I*J*K, r1 * r2);
%     for i=1:K
%         Z((i-1) * I * J + 1 : i * I * J, :) = ...
%             kron(Zt'*diag(D1(i,:)), W*diag(D2(i,:)));
%     end
% 
%     vt = Z \ x; 
% 
%     Vt = reshape(vt, r2, r1); % Afhankelijk van hoe reshape gebeurd kan dit fout zijn
    
    rowList = zeros(K*r1*I*r2,1);
    colList = zeros(K*r1*I*r2,1);
    valList = zeros(K*r1*I*r2,1);

    currentIdx = 1;

    %Z = zeros(K * r1 * I, r1 * r2);

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

        %Z((i-1) * r1 * I + 1 : i * r1 * I, :) = ...
        % kron(diag(D1(i,:)), W*diag(D2(i,:)));

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
%         Fk = W * diag(D2(k,:))' * Vt;
%         xk = reshape(X(:,:,k)', numel(X(:,:,k)), 1);
% 
%         D1(k,:) = (kr(Fk, Zt') \ xk)'; % transpose van kathri-rao nodig (?)
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

% Optimalisatie mogelijk: element per element output berekenen i.p.v.
% in 1 keer zoals nu gebeurd.
function [Zt] = updateZt(X, W, D1, Vt, D2, I, J, K, r1)
    unfoldX = zeros(I*K, J);
    for i=1:K
        unfoldX((i-1) * I + 1 : i * I,:) = X(:,:,i);
    end

    F = zeros(I * K, r1);
    for i=1:K
        F((i-1) * I + 1 : i * I,:) = ...
            W*diag(D2(i,:))*Vt*diag(D1(i,:));
    end

    Zt = F \ unfoldX; % Paper gebruikt F^T (lijkt verkeerd)
end

function [cD1, D1] = update_cD1(D1, Zt, samples, bf1d, K, r1, d1)
    % Projectiestrategie
    %X = sparse(r1*K, r1*d1);
    rowList = zeros(d1*K*r1,1);
    colList = zeros(d1*K*r1,1);
    valList = zeros(d1*K*r1,1);
    currentIdx = 1;
    for l=1:r1
        %Xl = zeros(K, d1);
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

        %X(((l-1)*K) + 1:l * K, ((l-1)*d1) + 1:l * d1) = Xl;
    end
    
    X = sparse(rowList, colList, valList);

    s2 = size(X, 2);
    cD1 = X \ reshape(D1, numel(D1), 1);
    
    for l=1:r1
        D1(:,l) = X(((l-1)*K) + 1:l * K, ((l-1)*d1) + 1:l * d1) * cD1((l-1) * d1 + 1: l * d1);
    end
end

function [cD2, D2] = update_cD2(D2, Vt, Zt, samples, cD1, bf1, bf2d, K, r2, d2)
    % Projectiestrategie
    X = zeros(r2*K, r2*d2);
    for l=1:r2
        Xl = zeros(K, d2);
        for j=1:K
            inner_sample = Zt * samples(:,j);
            sample = applyFlexibleFunctions(inner_sample, cD1, bf1);

            xl = Vt(l,:) * sample;

            for k=1:d2
                func = bf2d{k};
                grad = func(xl);
                Xl(j,k) = grad;
            end
        end
        X(((l-1)*K) + 1:l * K, ((l-1)*d2) + 1:l * d2) = Xl;
    end
    
    s2 = size(X, 2);
    cD2 = X \ reshape(D2, numel(D2), 1);
    for l=1:r2
        D2(:,l) = X(((l-1)*K) + 1:l * K, ((l-1)*d2) + 1:l * d2) * cD2((l-1) * d2 + 1: l * d2);
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
