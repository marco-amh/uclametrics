# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 00:12:32 2021

@author: marco
"""
import pandas as pd
import numpy as np
from scipy.linalg import inv
os.chdir('C://Users//marco//Desktop//Projects//Bayesian_VAR')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

y = Puller.Banxico(serie="SR16734", name="IGAE", plot=False)
p = Puller.Banxico(serie="SP1", name="Inflation", plot=False)
r = Puller.Banxico(serie="SF3270", name="Interest_rate", plot=False)
m = Puller.Banxico(serie="SF1", name="Money", plot=False)
df = pd.concat([y, p, r, m], axis=1).dropna()


N = df.shape[1];
L = 2;   #number of lags in the VAR

def lagger(p,mtx):
    df1 = pd.DataFrame() # Support matrix
    for i in range(0,p+1):
        tmp = mtx.shift(i)
        tmp.columns = tmp.columns + '_'+ str(i)
        df1 = pd.concat([df1,tmp], axis=1)
    return(df1.dropna())

df = lagger(p=2,mtx=df)


Y = df.iloc[:,:N]
X = df.iloc[:,N:];
X['Cons'] = 1
T = X.shape[0];

# Move to the first column cons
first_column = X.pop('Cons')
X.insert(0, 'Cons', first_column)

#compute standard deviation of each series residual via an ols regression
#to be used in setting the prior
#first variable
y=Y.iloc[:,0:1].to_numpy();
x=X.iloc[:,[0,1]].to_numpy();
b0 = inv(x.T @ x) @ (x.T @ y)
s1 = np.sqrt(((y - x @ b0).T @ (y- x @ b0))/(y.shape[0]-2)) 

#second variable
y=Y.iloc[:,1:2].to_numpy();
x=X.iloc[:,[0,2]].to_numpy();
b0 = inv(x.T @ x) @ (x.T @ y)
s2 = np.sqrt(((y - x @ b0).T @ (y- x @ b0))/(y.shape[0]-2)) 

#third variable
y=Y.iloc[:,2:3].to_numpy();
x=X.iloc[:,[0,3]].to_numpy();
b0 = inv(x.T @ x) @ (x.T @ y)
s3 = np.sqrt(((y - x @ b0).T @ (y- x @ b0))/(y.shape[0]-2)) 

#fourth variable
y=Y.iloc[:,3:4].to_numpy();
x=X.iloc[:,[0,4]].to_numpy();
b0 = inv(x.T @ x) @ (x.T @ y)
s4 = np.sqrt(((y - x @ b0).T @ (y- x @ b0))/(y.shape[0]-2)) 


# Parameters to control the prior
lamda1 = 0.1 #tightness prior on the AR coefficients
lamda3 = 0.05 # tightness of prior on higher lags
lamda4 = 1 # tightness of prior on the constant term
# Specify the prior mean of the coefficients of the two equations of the VAR

B0 = np.zeros(shape=(N*L+1,N))

for i in range(1,N+1):
    B0[i:i+1,i-1:i]=0.95;
B0

B0 = B0.reshape(-1,1)

# Specify the prior variance of vec(B)
H=np.eye(N*(N*L+1),N*(N*L+1))
H.shape

# Small for coefficients we want close to zero
H[2,2]=1e-9;
H[3,3]=1e-9;
H[4,4]=1e-9;
H[6,6]=1e-9;
H[7,7]=1e-9;
H[8,8]=1e-9;
# for others like the normal conjugate prior
# fist equation
H[0,0]=(s1*lamda4)**2;
H[1,1]=(lamda1)**2;
H[5,5]=(lamda1/(2**lamda3))**2;
# Second equation
H[9,9]=(s2*lamda4)**2;
H[10,10]=((s2*lamda1)/s1)**2;
H[11,11]=(lamda1)**2;
H[12,12]=((s2*lamda1)/s3)**2;
H[13,13]=((s2*lamda1)/s4)**2;
H[14,14]=((s2*lamda1)/(s1*(2**lamda3)))**2;
H[15,15]=(lamda1/(2**lamda3))**2;
H[16,16]=((s2*lamda1)/(s3*(2**lamda3)))**2;
H[17,17]=((s2*lamda1)/(s4*(2**lamda3)))**2;
# Third equation
H[18,18]=(s3*lamda4)**2;
H[19,19]=((s3*lamda1)/s1)**2;
H[20,20]=((s3*lamda1)/s2)**2;
H[21,21]=(lamda1)**2;
H[22,22]=((s3*lamda1)/s4)**2;
H[23,23]=((s3*lamda1)/(s1*(2**lamda3)))**2;
H[24,24]=((s3*lamda1)/(s2*(2**lamda3)))**2;
H[25,25]=(lamda1/(2**lamda3))**2;
H[26,26]=((s3*lamda1)/(s4*(2**lamda3)))**2;
# Fourth equation
H[27,27]=(s4*lamda4)**2;
H[28,28]=((s4*lamda1)/s1)**2;
H[29,29]=((s4*lamda1)/s2)**2;
H[30,30]=((s4*lamda1)/s3)**2;
H[31,31]=(lamda1)**2;
H[32,32]=((s4*lamda1)/(s1*(2**lamda3)))**2;
H[33,33]=((s4*lamda1)/(s2*(2**lamda3)))**2;
H[34,34]=((s4*lamda1)/(s3*(2**lamda3)))**2;
H[35,35]=(lamda1/(2**lamda3))**2;
pd.DataFrame(H)

Y = Y.to_numpy()
X = X.to_numpy()

# prior scale matrix for sigma the VAR covariance
S = np.eye(N);

# prior degrees of freedom
alpha = N+1;

#starting values for the Gibbs sampling algorithm
Sigma = np.eye(N);
betaols = (inv(X.T @ X) @ (X.T @ Y)).reshape(-1,1)

Reps=40000;
burn=30000;
out1=[]; # will store IRF of R 
out2=[]; # will store IRF of GB 
out3=[]; # will store IRF of U 
out4=[]; # will store IRF of P 

#def stability(beta,n,l):

coef = beta.reshape(N*L+1,N,-1)
FF = np.zeros(shape=(N*L, N*L));
FF[N+1-1:(N*L), 1-1:(N*(L-1))] = np.eye(N*(L-1),N*(L-1));
temp = beta.reshape(N*L+1,N,-1)
temp = temp[2-1:N*L+1, 1-1:N].T;
FF[1-1:N, 1-1:N*L] = temp;
ee = np.max(np.abs(np.linalg.eig(FF)));
S = ee>1;

i=1;
for j in range(1,Reps):
#step 1 draw the VAR coefficients
    M=inv(inv(H)+np.kron(inv(Sigma), X.T @ X)) @ (inv(H) @ B0 + np.kron(inv(Sigma),X.T @ X) @ betaols);
    V=inv(inv(H)+np.kron(inv(Sigma),X.T @ X));
#check for stability of the VAR
    check=-1;
    while check<0:
        beta = M+(np.random.normal(1, N*(N*L+1)) * np.linalg.cholesky(V)).T;
        CH = stability(beta,N,L);
        if CH==0:
            check = 10;
#draw sigma from the IW distribution
    e=Y-X*reshape(beta,N*L+1,N);
#scale matrix
    scale=e.T @ e + S;
    Sigma=IWPQ(T+alpha,inv(scale));

    if j>burn
    #impulse response using a cholesky decomposition
        A0=chol(Sigma);
        v=zeros(60,N);
        v(L+1,2)=-1; %shock the government bondyield
        yhat=zeros(60,N);
        for i in range(3,60)
            yhat(i,:)=[0 yhat(i-1,:) yhat(i-2,:)]*reshape(beta,N*L+1,N)+v(i,:)*A0;

    out1=[out1 yhat(3:end,1)];
    out2=[out2 yhat(3:end,2)];
    out3=[out3 yhat(3:end,3)];
    out4=[out4 yhat(3:end,4)];



# Notes:
# T is traspose and is ' in matlab
# inv is the inverse or inv in matlab
# @ is matrix multiplication or * in matlab


data=xlsread('\data\data
             3US.xls'); 
N=size(data,2);
L=2;   #number of lags in the VAR
Y=data;
X=[ones(size(Y,1),1) lag0(data,1) lag0(data,2) ];
Y=Y(3:end,:);
X=X(3:end,:);
T=rows(X);


#compute standard deviation of each series residual via an ols regression
#to be used in setting the prior
#first variable
y=Y(:,1);
x=X(:,1:2); 
b0=inv(x'*x)*(x'*y);
s1=sqrt(((y-x*b0)'*(y-x*b0))/(rows(y)-2));  %std of residual standard error
%second variable
y=Y(:,2);
x=X(:,[1 3]); 
b0=inv(x'*x)*(x'*y);
s2=sqrt(((y-x*b0)'*(y-x*b0))/(rows(y)-2));  
%third variable
y=Y(:,3);
x=X(:,[1 4]); 
b0=inv(x'*x)*(x'*y);
s3=sqrt(((y-x*b0)'*(y-x*b0))/(rows(y)-2));  
%fourth variable
y=Y(:,4);
x=X(:,[1 5]); 
b0=inv(x'*x)*(x'*y);
s4=sqrt(((y-x*b0)'*(y-x*b0))/(rows(y)-2));


%parameters to control the prior
lamda1=0.1;  %tightness prior on the AR coefficients
lamda3=0.05;   %tightness of prior on higher lags 
lamda4=1;  %tightness of prior on the constant term

%specify the prior mean of the coefficients of the Two equations of the VAR

B0=zeros((N*L+1),N);
for i=1:N
    B0(i+1,i)=0.95;
end
B0=vec(B0);
%Specify the prior variance of vec(B)
H=eye(N*(N*L+1),N*(N*L+1)); 
%small for coefficients we want close to zero
H(3,3)=1e-9;
H(4,4)=1e-9;
H(5,5)=1e-9;
H(7,7)=1e-9;
H(8,8)=1e-9;
H(9,9)=1e-9;
%for others like the normal conjugate prior
%ist equation
H(1,1)=(s1*lamda4)^2;
H(2,2)=(lamda1)^2;
H(6,6)=(lamda1/(2^lamda3))^2;
%second equation
H(10,10)=(s2*lamda4)^2;
H(11,11)=((s2*lamda1)/s1)^2;
H(12,12)=(lamda1)^2;
H(13,13)=((s2*lamda1)/s3)^2;
H(14,14)=((s2*lamda1)/s4)^2;
H(15,15)=((s2*lamda1)/(s1*(2^lamda3)))^2;
H(16,16)=(lamda1/(2^lamda3))^2;
H(17,17)=((s2*lamda1)/(s3*(2^lamda3)))^2;
H(18,18)=((s2*lamda1)/(s4*(2^lamda3)))^2;
%third equation
H(19,19)=(s3*lamda4)^2;
H(20,20)=((s3*lamda1)/s1)^2;
H(21,21)=((s3*lamda1)/s2)^2;
H(22,22)=(lamda1)^2;
H(23,23)=((s3*lamda1)/s4)^2;
H(24,24)=((s3*lamda1)/(s1*(2^lamda3)))^2;
H(25,25)=((s3*lamda1)/(s2*(2^lamda3)))^2;
H(26,26)=(lamda1/(2^lamda3))^2;
H(27,27)=((s3*lamda1)/(s4*(2^lamda3)))^2;
%fourth equation
H(28,28)=(s4*lamda4)^2;
H(29,29)=((s4*lamda1)/s1)^2;
H(30,30)=((s4*lamda1)/s2)^2;
H(31,31)=((s4*lamda1)/s3)^2;
H(32,32)=(lamda1)^2;
H(33,33)=((s4*lamda1)/(s1*(2^lamda3)))^2;
H(34,34)=((s4*lamda1)/(s2*(2^lamda3)))^2;
H(35,35)=((s4*lamda1)/(s3*(2^lamda3)))^2;
H(36,36)=(lamda1/(2^lamda3))^2;





%prior scale matrix for sigma the VAR covariance
S=eye(N);
%prior degrees of freedom
alpha=N+1;

%starting values for the Gibbs sampling algorithm
Sigma=eye(N);
betaols=vec(inv(X'*X)*(X'*Y));

Reps=40000;
burn=30000;
out1=[]; %will store IRF of R 
out2=[]; %will store IRF of GB 
out3=[]; %will store IRF of U 
out4=[]; %will store IRF of P 
i=1;
for j=1:Reps

%step 1 draw the VAR coefficients
M=inv(inv(H)+kron(inv(Sigma),X'*X))*(inv(H)*B0+kron(inv(Sigma),X'*X)*betaols);
V=inv(inv(H)+kron(inv(Sigma),X'*X));
%check for stability of the VAR
check=-1;
while check<0
beta=M+(randn(1,N*(N*L+1))*chol(V))';
CH=stability(beta,N,L);
if CH==0
    check=10;
end
end

%draw sigma from the IW distribution
e=Y-X*reshape(beta,N*L+1,N);
%scale matrix
scale=e'*e+S;
Sigma=IWPQ(T+alpha,inv(scale));

if j>burn
    #impulse response using a cholesky decomposition
    A0=chol(Sigma);
    v=zeros(60,N);
    v(L+1,2)=-1; %shock the government bondyield
   yhat=zeros(60,N);
   for i=3:60
    yhat(i,:)=[0 yhat(i-1,:) yhat(i-2,:)]*reshape(beta,N*L+1,N)+v(i,:)*A0;
end
out1=[out1 yhat(3:end,1)];
out2=[out2 yhat(3:end,2)];
out3=[out3 yhat(3:end,3)];
out4=[out4 yhat(3:end,4)];
end

end


subplot(2,2,1)
plot([prctile(out1,[50 16 84],2) zeros(size(out3,1),1)]);
title('Response of the Federal Funds rate');
axis tight

subplot(2,2,2)
plot([prctile(out2,[50 16 84],2) zeros(size(out3,1),1)]);
title('Response of the Government Bond Yield');
axis tight


subplot(2,2,3)
plot([prctile(out3,[50 16 84],2) zeros(size(out3,1),1)]);
title('Response of the Unemployment Rate');
axis tight


subplot(2,2,4)
plot([prctile(out4,[50 16 84],2) zeros(size(out3,1),1)]);
title('Response of Inflation');
axis tight
legend('Median Response','Upper 84%','Lower 16%','Zero Line');

