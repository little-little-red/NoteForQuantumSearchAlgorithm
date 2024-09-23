% Grover Search Algorithm (Standard + Modofied)
% Made for the Paper: "Enhancing Grover's Search Algorithm: A Modified Approach to Increase the Probability of Good States"
% https://link.springer.com/article/10.1007/s11227-024-06142-5
% Dr. Ismael Abdulrahman, ismael.abdulrahman@epu.edu.iq
clear; clc;
%%============================Input Parameters=============================
n                   = 10;                          % Number of qubits
N                   = 2^n;                         % All possible combinations of states
MaxItr              = floor(pi/4*sqrt(N))*2;       % Max. number of iterations
%%==========================Initialization=================================
GSA_Amplitude       = zeros(N, MaxItr);
oracle              = zeros(N, MaxItr);
Init_n              = zeros(N, MaxItr);
theta               = zeros(MaxItr, 1);
amplitude           = zeros(MaxItr, 1);
%%========================== Gates (1-qubit) ==============================
X                          = [0 1; 1 0];
H                          = 1/sqrt(2) * [1 1 ; 1 -1];
Z                          = [1 0; 0 -1];
%%========================= Registers (n-qubit) ===========================
Hn                         = H;
Xn                         = sparse(X);
for k = 1:n-1
    Hn                     = kron(Hn, H);
    Xn                     = kron(Xn, X);
end
In                         = speye(N);
InZ                        = In;
InZ(end-1:end, end-1:end)  = Z;
CZ                         = InZ;
%%========================= Initial Input States ==========================
ket0                       = ([1 0])';
Init_ket                   = (ket0);
for k = 1:n-1
    Init_ket               = kron(Init_ket, ket0);
end
%%========================= Searching Iterations ==========================
GSA                        = zeros(MaxItr, 2);
CR2                        = speye(N);
for Mthd = 1:2             % Comparison between the standard and modified versions
    for k = 1:MaxItr
        if k == 1
            Init_n           = Hn * Init_ket;
            amplitude(k)     = (2*k+1)/(2*k-1)/sqrt(N);
        else
            Init_n           = GSA_Amplitude(:, k-1);
            amplitude(k)     = (2*k+1)/(2*k-1);
        end
        if Mthd == 1
            theta(k)         = 0; 
        elseif Mthd == 2
            theta(k)         = 2*atan(1-4/N) + amplitude(k);
        end
        R                           = [cos(theta(k)/2) -sin(theta(k)/2); sin(theta(k)/2) cos(theta(k)/2)] * Z;   % Contribution of the paper (generating new rotation-around-y-axis gate)
        CR2(end-1:end, end-1:end)   = R;
        oracle                      = CZ * Init_n;                                          % Oracle-i   
        GSA_Amplitude(:, k)         = - Hn*Xn * CR2 * Xn*Hn * oracle;                           % Grover difussion operator- i (reflection about the mean)
    end
    GSA(:,Mthd)                     = GSA_Amplitude(end, :)';                                        
end
%%=====================Plotting Probability vs Iteration===================
figure(1)
bar(GSA.^2);
xlabel('Iteration','FontSize',18)
ylabel('Probability','FontSize',18)
lgd = legend('Standard GSA','Modified GSA','Location', 'northwest');
fontsize(lgd,12,'points')
grid on
set(gca,'FontSize',16)
text(MaxItr/10,0.75, sprintf('n =  %d', n),'Color','k','FontSize',18, AffectAutoLimits="on")
