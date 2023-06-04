warning("off")

bf1 = {@theta1, @theta2, @theta3, @theta4};%, @theta5};%, @theta6};%, @theta7, @theta8, @theta9, @theta10};
bf1d = {@theta1d, @theta2d, @theta3d, @theta4d};%, @theta5d};%, @theta6d};
bf2 = {@theta1, @theta2, @theta3, @theta4};%, @theta5};%, @theta6};%, @theta7, @theta8, @theta9, @theta10};
bf2d = {@theta1d, @theta2d, @theta3d, @theta4d};%, @theta5d};%, @theta6d};%, @theta7, @theta8, @theta9, @theta10};

r1 = 100;
r2 = 80;

tic
[We, D2e, Vte, D1e, Zte, Ht, cD1e, cD2e] = PARATUCK2_CMTF_REG(Jac, F, bf1, bf1d, bf2, bf2d, r1, r2, inputs);
toc
%% Save parameters

%save("Parameters_results/CMTF_PT2_SEED/DGDH/12422/CMTF_PT2_80_80_2_2", "We", "D2e", "Vte", "D1e", "Zte", "Ht", "cD1e", "cD2e")
save("Parameters_results/CMTF_PT2_SEED/Third_row_of_seeds/CMTF_PT2_100_80", "We", "D2e", "Vte", "D1e", "Zte", "Ht", "cD1e", "cD2e")

%% Functions

function [f] = theta1(x)
    f = x;
end

function [f] = theta1d(x)
    f = 1;
end

function [f] = theta2(x)
    f = x^2;
end

function [f] = theta2d(x)
    f = 2*x;
end

function [f] = theta3(x)
    f = x^3;
end

function [f] = theta3d(x)
    f = 3*x^2;
end

function [f] = theta4(x)
    f = x^4;
end

function [f] = theta4d(x)
    f = 4*x^3;
end

function [f] = theta5(x)
    f = x^5;
end

function [f] = theta5d(x)
    f = 5*x^4;
end

function [f] = theta6(x)
    f = x^6;
end

function [f] = theta6d(x)
    f = 6*x^5;
end