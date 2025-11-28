x1=-1:0.1:1;
x2=-1:0.1:1;
[X1,X2]=meshgrid(x1,x2);
fx=X1.^2+X2.^2;
contour (X1,X2,fx,[0.25 0.5],'g');
hold on
plot([-1,1],[0,0])
hold on
plot([0,0],[-1,1])
hold on
%%
x1=-1:0.1:1;
x2=-1:0.1:1;
[X1,X2]=meshgrid(x1,x2);
f=x1+x2-1;
fx=X1.^2+X2.^2;
contour (X1,X2,f,fx,[0.25 0.5],'g');
hold on
plot([-1,1],[0,0])
hold on
plot([0,0],[-1,1])
hold on
%%
x1=-1:0.1:1;
x2=-1:0.1:1;
[X1,X2]=meshgrid(x1,x2);
n=1;
hold on
contour(X1,X2,fx,[0 0],'r');
hold on
contour(X1,X2,fx),[0.1 0.1,'b'];

hold on 
plot(0.5,0.5,'*')
hold on
plot([0.5 1],[0.5 1]),'--x'
%%
a=[1 5 4; 2 4 6; 2 10 8];
b=rref(d);
c=[0;0;0];
d=[a c]
%%
a=[5 -12 9;1 -2 3;0 -2 6];
b=det(a);
disp(b)
%%
% Define the matrix A
A = [-4 14 0; -3 9 0; -1 1 4];

% Calculate the reduced row echelon form of A
R = rref(A);

% Display the result
disp('The reduced row echelon form of the matrix A is:');
disp(R);
%%
A = [-4 14 0;-1 9 0;-1 1 4];
b=eig(A)
%%
% Define the matrix A
A = [-4 14 0; -1 9 0; -1 1 4];

% Calculate the eigenvalues and eigenvectors
[V, D] = eig(A);

% Display the eigenvalues and eigenvectors
disp('The eigenvalues of the matrix A are:');
disp(diag(D));

disp('The eigenvectors of the matrix A are:');
disp(V);
%%
% Define the matrix A
A = [1 2 2 -4 -6; 2 4 4 5 14; 2 4 4 4 12];

% Compute the reduced row echelon form of A
rref_A = rref(A);

% Display the reduced row echelon form
disp('The reduced row echelon form (RREF) of matrix A is:');
disp(rref_A);

% Compute the null space of the transpose of A (which gives us the row null space of A)
RNS = null(A', 'r');

% Display the row null space
disp('The row null space (RNS) of matrix A is:');
disp(RNS);
%%
format short
clear all;
A = [1 2 2 4 10;4 8 8 5 18;0 0 0 -3 -6]
rank_A = rank(A);
[m, n] = size(A);
A(2,:) = A(2,:) - 4*A(1,:)
A(2,:) = A(2,:) / (-11)
A(1,:) = A(1,:) - 4*A(2,:)
A(3,:) = A(3,:) + 3*A(2,:)
[R, p] = rref(A);  % p gives index of independent columns
r = length(p);     % r is rank of matrix A

% Display the RREF of A and the pivot columns
disp('RREF of A:');
disp(R);
disp('Pivot columns:');
disp(p);

% Retrieve r non-zero rows from R
R = R(1:r, :);
% Find the index of dependent columns
f = setdiff(1:n, p);
% Initialize the null space matrix N
N = zeros(n, n - r);
% Retrieve dependent columns from R and put in rows specified by p
N(p, :) = -R(1:r, f); 
N(f, :) = eye(n - r);

% Display the null space matrix N
disp('Right Null Space Matrix N:');
disp(N);
%%
% Define the matrix A
A = [1 2 2 -4 -6; 2 4 4 5 14; 2 4 4 4 12];

% Compute the reduced row echelon form of A
rref_A = rref(A);

% Display the reduced row echelon form
disp('The reduced row echelon form (RREF) of matrix A is:');
disp(rref_A);

% Compute the right null space of A
RNS = null(A);

% Display the right null space
disp('The right null space (RNS) of matrix A is:');
disp(RNS);
%%
format short
clear all;
clc;
A = [1 2 2 -4 -6;2 4 4 5 14;2 4 4 4 12]
rank_A = rank(A);
[m, n] = size(A);
A(2,:) = A(2,:) - 4*A(1,:)
A(2,:) = A(2,:) / (-11)
A(1,:) = A(1,:) - 4*A(2,:)
A(3,:) = A(3,:) + 3*A(2,:)
[R, p] = rref(A);  % p gives index of independent columns
r = length(p);     % r is rank of matrix A

% Display the RREF of A and the pivot columns
disp('RREF of A:');
disp(R);
disp('Pivot columns:');
disp(p);

% Retrieve r non-zero rows from R
R = R(1:r, :);
% Find the index of dependent columns
f = setdiff(1:n, p);
% Initialize the null space matrix N
N = zeros(n, n - r);
% Retrieve dependent columns from R and put in rows specified by p
N(p, :) = -R(1:r, f); 
N(f, :) = eye(n - r);

% Display the null space matrix N
disp('Right Null Space Matrix N:');
disp(N);
%%
a=[3 0 0; 2 1 2; -1 0 2];
det(a)
eig(a)
%%
clc;
a=[ 0 0 0 9; 0 0 5 0; 0 4 0 0;8 0 0 0];
eig(a)
%%
clc;
A=[-6 18 -18; -1 3 1; 2 -6 10];
[eigenvectors,eigenvalues]=eig(A)
%%
clc;
function [C,R] = cr(A)
    [R, j] = rref(A);  
    r = length(j);                    
    R = R(1:r,:);                     
    C = A(:,j);
end

A=[1 1 1 -3 -4;4 4 4 -4 0;3 3 3 -5 -4];
[C,R]=cr(A)
