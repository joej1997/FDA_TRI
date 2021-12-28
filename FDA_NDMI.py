# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 01:09:36 2021

@author: joeja
"""

#import packages
import numpy as np
import matplotlib.pyplot as pp
from numpy import random
import torch
import torch.optim as optim


#create data 100 datapoints, 24 long
n=1000
l= 24
p=2
t= np.arange(0,l,1)
real_delay = np.random.randint(2,l+1,p)


Xlist=list()
for i in range(p):
    Xlist.append(np.zeros((n,l)))

for i in range(n):
    for j in range(p):
        curVec = np.sin(t*2*np.pi/12) + np.random.normal(size=l)
        Xlist[j][i,:] = curVec


#fit b-spline function to each X

#create b-spline basis
def Belem0(u,i,knots):
    u_i=knots[i]
    u_ip1=knots[i+1]
    toreturn=np.zeros((len(u)))
    toreturn[np.logical_and(u_i <= u,u_ip1> u)] =1
    return toreturn
            
def Bspline_elm(u,i,knots,order):
    if order == 0:
        return Belem0(u,i,knots)
    else:
        u_i=knots[i]
        u_ip1=knots[i+1]
        u_ipp= knots[i+order]
        u_ippp1 = knots[i+order+1]
        return ((u-u_i)/(u_ipp-u_i))*Bspline_elm(u,i,knots,order-1)+((u_ippp1-u)/(u_ippp1-u_ip1))*Bspline_elm(u,i+1,knots,order-1)

def get_basis_mat(order,t,knots):
    basis_mat= np.zeros((len(t),len(knots) - order-1))
    for i in range(len(knots) - order-1):
        basis_mat[:,i]= Bspline_elm(t,i,knots,order)
    return basis_mat

def get_A(basis_mat2,D22,lam):
    return basis_mat2+lam*D22

def get_hat(basis_mat,A):
    matinv=np.linalg.inv(A)
    toreturn = basis_mat @ matinv @ np.transpose(basis_mat)
    return toreturn

def get_coefs(basis_mat,A,y):
    b=np.matmul(np.transpose(basis_mat),y)
    return np.linalg.solve(A,b)

def get_GCV(y_mat,lam_seq,basis_mat,basis_mat2,D22):
    GCV_mat=np.zeros((len(lam_seq),y_mat.shape[0]))
    for i in range(GCV_mat.shape[0]):
        lam=lam_seq[i]
        A= get_A(basis_mat2,D22,lam)
        hat=get_hat(basis_mat,A)
        tr=np.matrix.trace(hat)
        GCV_BOT= (hat.shape[0]-tr)**2
        #get GCV for each observation and average
        for j in range(GCV_mat.shape[1]):
            y=y_mat[j,:]
            coefs=get_coefs(basis_mat,A,y)
            y_hat= np.matmul(basis_mat,coefs)
            GCV_TOP = np.sum((y-y_hat)**2)
            GCV_mat[i,j]=GCV_TOP/GCV_BOT
    return GCV_mat


#intialize parameters
l= 24
order=3
t = np.arange(0,l,1)
internal_knots=20
knots = np.linspace(0, 23, internal_knots)
knot_diff= knots[1]-knots[0]
knots= np.linspace(0-3*knot_diff,(l-1)+3*knot_diff,internal_knots+6)

#create bspline basis matrix
basis_mat= get_basis_mat(order,t,knots)
ns=basis_mat.shape[1]

pp.figure()
pp.plot(t,basis_mat)
pp.xlim([0, l-1])

#create better second difference operator for P-splines
diag1=np.diag(np.zeros((ns))-2.0,0)
diag2=np.diag(np.zeros((ns-1))+1.0,1)
diag3=np.diag(np.zeros((ns-1))+1.0,-1)
D2=diag1+diag2+diag3
D2[0,0]=-1
D2[-1,-1]=-1
del diag1,diag2,diag3

#create sqaured norm matricies
basis_mat2=np.transpose(basis_mat) @ basis_mat
D22= np.transpose(D2) @ D2
      
# fit one curve
lam=0
y=Xlist[1][0,:]
A= get_A(basis_mat2,D22,lam)
hat=get_hat(basis_mat,A)
coefs=get_coefs(basis_mat,A,y)
tnew=np.arange(0,l-1,.01)
y_hat= np.matmul(get_basis_mat(order,tnew,knots),coefs)
pp.figure()
pp.plot(tnew,y_hat,label="y_hat")
pp.plot(t,y,label="y")
pp.legend(loc="lower left")
pp.xlabel("time")
pp.ylabel("y")

# test sequence of lambdas on each X variable
lam_seq=10.0**np.arange(-5,5,.01)
results= get_GCV(Xlist[0], lam_seq, basis_mat, basis_mat2, D22)
pp.figure()
pp.plot(lam_seq,results) #plot example
pp.xscale("log")

#plot mean
GCV_avg=np.mean(results,axis=1)
pp.figure()
pp.plot(lam_seq,GCV_avg)
pp.xscale("log")
pp.xlabel("lambda")
pp.ylabel("average GCV")

# get results for all Xs
best_lam = np.zeros((len(Xlist)))
for i in range(len(Xlist)):
    results= get_GCV(Xlist[i], lam_seq, basis_mat, basis_mat2, D22)
    GCV_avg=np.mean(results,axis=1)
    best_lam[i]=lam_seq[np.argmin(GCV_avg)]



#save matricies of smoothed Xs on finer grid
n_fg=200
fg=np.linspace(0, l-1,n_fg)
fg_basis=get_basis_mat(order,fg,knots)
Xsmooth_list = list()

for j in range(len(Xlist)):
    curX_smooth = np.zeros((n,n_fg))
    for i in range(n):
        lam=best_lam[j]
        y=Xlist[j][i,:]
        A= get_A(basis_mat2,D22,lam)
        hat=get_hat(basis_mat,A)
        coefs=get_coefs(basis_mat,A,y)
        curX_smooth[i,:]= np.matmul(fg_basis,coefs)
    Xsmooth_list.append(torch.tensor(curX_smooth))

#plot smoothed curve with data
y=Xlist[1][0,:]
y_hat = Xsmooth_list[1][0,:]
pp.figure()
pp.plot(fg,y_hat,label="y_hat")
pp.plot(t,y,label="y")
pp.legend(loc="lower left")
pp.xlabel("time")
pp.ylabel("y")



del curX_smooth, lam, A, hat, coefs, results, y, y_hat, tnew, lam_seq, j, i, GCV_avg, curVec


betaMat = torch.zeros(n_fg,p)
betaMat[int(-real_delay[0]*n_fg/l):,0]=torch.sin(2*torch.pi*torch.tensor(fg[int(-real_delay[0]*n_fg/l):])/l)    #1
betaMat[int(-real_delay[1]*n_fg/l):,1]=torch.sin(2*torch.pi*torch.tensor(fg[int(-real_delay[1]*n_fg/l):])/l)    #1
pp.figure()
pp.plot(betaMat)

response=torch.zeros(n)
h=(l-1)/(n_fg-1)
for i in range(n):
    for j in range(p):
        toIntegrate =  Xsmooth_list[j][i,:] * betaMat[:,j]
        response[i] += h*torch.sum(toIntegrate[1:-1])+(h/2)*(toIntegrate[0]+toIntegrate[-1])+ 0.1*np.random.normal()


#make U matrix from "Locally Sparse Estimator for Funtional Linear Regression Models"
Ulist=list()
for x in range(p):
    curU = torch.zeros(n,ns,dtype=torch.float64)
    for i in range(n):
        for j in range(ns):
            toIntegrate = Xsmooth_list[x][i,:] * fg_basis[:,j]
            curU[i,j] = h*torch.sum(toIntegrate[1:-1])+(h/2)*(toIntegrate[0]+toIntegrate[-1])
    Ulist.append(curU)
    

###################################################################################
####################################################################################
##############################    sequential thresholding   ##########################3
####################################################################################
#####################################################################################

# define eval function
def EvalFun(b,Xlist,y,D22,tau,Ulist):
    yhat = 0
    sb_term = 0
    for i in range(len(Xlist)):
        yhat += Ulist[i] @ b[:,i]
        sb_term += b[:,i] @ D22 @ b[:,i]
    sse_term= torch.sum((y-yhat)**2)
    sb_term = sb_term*tau

    all_term = sse_term + sb_term
    return sse_term, sb_term, all_term



#create second difference operator for P-splines
D22=torch.tensor(D22,requires_grad=False)

#initalize parameters
b = torch.zeros(ns,p,dtype=torch.float64)
response= torch.tensor(response,requires_grad=False)
fg_basis=get_basis_mat(order,fg,knots)
fg_basis=torch.tensor(fg_basis,requires_grad=False,dtype=torch.float64)
lam = 1e-4
tau = 8e01
punish= lam*torch.transpose(torch.transpose(torch.ones(b.shape[0],b.shape[1]),0,1) 
                           *torch.arange(ns,0,-1, dtype=torch.float64),0,1)
h=(l-1)/(n_fg-1)

# for gradient descent
delta      =3e-5
numiter = 20000
prev_all_term=0

for i in range(numiter): 
    # Make the coefficients a parameter that we can compute gradients 
    b = torch.tensor(b, requires_grad=True)
    
    #calculate loss
    sse_term, sb_term, all_term = EvalFun(b=b,Xlist=Xsmooth_list,
        y=response, D22=D22,tau=tau, Ulist=Ulist)
    
    # Compute gradients
    all_term.backward()
    gradLossB = b.grad
    b=b.detach().clone()
    
    # update the parameters
    b -= delta*gradLossB
    b[torch.abs(b)<punish]=0  
    
    #stopping criteria
    if np.abs(prev_all_term-all_term.item())<1e-15:
        break
    prev_all_term= all_term.item()
        
    if i % 100 == 0:
        print(i, all_term.item(),sse_term.item(),sb_term.item())


# look at results
yhat=0
for i in range(len(Ulist)):
    yhat += Ulist[i] @ b[:,i]
    
beta = torch.zeros(n_fg,p)
for i in range(p):
    beta[:,i] = fg_basis @ b[:,i]

pp.figure()
pp.scatter(response.detach(),yhat.detach())

pp.figure()
pp.plot(fg,beta.detach())
pp.xlabel("t (months)")
pp.ylabel("Beta")
pp.title("Sequential Thresholding")


pp.figure()
pp.plot(b.detach())
pp.xlabel("basis function")
pp.ylabel("b")


# test exact answer
yhat=0
betaMat = torch.zeros(len(fg_basis),len(Xsmooth_list),dtype=torch.float64)
betaMat[int(-real_delay[0]*n_fg/l):,0]=torch.sin(2*torch.pi*torch.tensor(fg[int(-real_delay[0]*n_fg/l):])/l)  #1
betaMat[int(-real_delay[1]*n_fg/l):,1]=torch.sin(2*torch.pi*torch.tensor(fg[int(-real_delay[1]*n_fg/l):])/l)  #1
h=(l-1)/(n_fg-1)
pp.figure()
pp.plot(fg,betaMat)
pp.title("True Beta")
pp.ylabel("Beta")
pp.xlabel("t (months)")

for i in range(len(Xsmooth_list)):
    toIntegrate = Xsmooth_list[i] * betaMat[:,i]
    yhat += h*torch.sum(toIntegrate[:,1:-1],1)+(h/2)*(toIntegrate[:,0]+toIntegrate[:,-1])

pp.figure()
pp.scatter(response.detach(),yhat.detach())



###################################################################################
####################################################################################
##############################    ridge penalty with soft thresholding   ###########
####################################################################################
#####################################################################################

# define eval function
def EvalFun(b,Xlist,y,D22,Z,lam,tau,Ulist):
    yhat = 0
    sb_term = 0
    ppb_term = 0
    for i in range(len(Xlist)):
        yhat += Ulist[i] @ b[:,i]
        sb_term += b[:,i] @ D22 @ b[:,i]
        ppb_term += torch.norm(Z @ b[:,i],p="fro")**2
    sse_term= torch.sum((y-yhat)**2)
    sb_term = sb_term*tau
    ppb_term = ppb_term*lam

    all_term = sse_term + sb_term + ppb_term
    return sse_term, sb_term, ppb_term, all_term



#create second difference operator for P-splines
D22=torch.tensor(D22,requires_grad=False)

#initalize parameters
b = torch.zeros(ns,p,dtype=torch.float64)
response= torch.tensor(response,requires_grad=False)
fg_basis=get_basis_mat(order,fg,knots)
fg_basis=torch.tensor(fg_basis,requires_grad=False,dtype=torch.float64)
lam = 5e-3
tau = 5e1
soft_t= 3e-2
h=(l-1)/(n_fg-1)

Z= torch.diag(torch.arange(ns,0,-1, dtype=torch.float64))

# for gradient descent
delta      =3e-5
numiter = 50000

for i in range(numiter): 
    # Make the coefficients a parameter that we can compute gradients 
    b = torch.tensor(b, requires_grad=True)
    
    #calculate loss
    sse_term, sb_term, ppb_term, all_term = EvalFun(b=b,Xlist=Xsmooth_list,
        y=response, D22=D22,Z=Z,lam=lam,tau=tau, Ulist=Ulist)
    
    # Compute gradients
    all_term.backward()
    gradLossB = b.grad
    b=b.detach().clone()
    
    # update the parameters
    b -= delta*gradLossB
        
    #stopping criteria
    if np.abs(prev_all_term-all_term.item())<1e-15:
        break
    prev_all_term= all_term.item()
    
    if i % 1000 == 0:
        print(i, all_term.item(),sse_term.item(),sb_term.item(),ppb_term.item())


#apply soft thresholding
b[torch.abs(b)<soft_t]=0  

# look at results
yhat=0
for i in range(len(Ulist)):
    yhat += Ulist[i] @ b[:,i]
    
beta = torch.zeros(n_fg,p)
for i in range(p):
    beta[:,i] = fg_basis @ b[:,i]

pp.figure()
pp.scatter(response.detach(),yhat.detach())

pp.figure()
pp.plot(fg,beta.detach())
pp.xlabel("t (months)")
pp.ylabel("Beta")
pp.title("Frobenius Penalty with Soft Thresholding")


pp.figure()
pp.plot(b.detach())
pp.xlabel("basis function")
pp.ylabel("b")




###################################################################################
####################################################################################
##############################   normal multiple functional linear regression #########
####################################################################################
#####################################################################################

# define eval function
def EvalFun(b,Xlist,y,D22,tau,Ulist):
    yhat = 0
    sb_term = 0
    for i in range(len(Xlist)):
        yhat += Ulist[i] @ b[:,i]
        sb_term += b[:,i] @ D22 @ b[:,i]
    sse_term= torch.sum((y-yhat)**2)
    sb_term = sb_term*tau

    all_term = sse_term + sb_term
    return sse_term, sb_term, all_term



#create second difference operator for P-splines
D22=torch.tensor(D22,requires_grad=False)

#initalize parameters
b = torch.zeros(ns,p,dtype=torch.float64)
response= torch.tensor(response,requires_grad=False)
fg_basis=get_basis_mat(order,fg,knots)
fg_basis=torch.tensor(fg_basis,requires_grad=False,dtype=torch.float64)
tau = 8e2
punish= lam*torch.transpose(torch.transpose(torch.ones(b.shape[0],b.shape[1]),0,1) 
                           *(torch.arange(ns,0,-1, dtype=torch.float64)-1),0,1)
h=(l-1)/(n_fg-1)

# for gradient descent
delta      =3e-5
numiter = 50000

for i in range(numiter): 
    # Make the coefficients a parameter that we can compute gradients 
    b = torch.tensor(b, requires_grad=True)
    
    #calculate loss
    sse_term, sb_term, all_term = EvalFun(b=b,Xlist=Xsmooth_list,
        y=response, D22=D22,tau=tau, Ulist=Ulist)
    
    # Compute gradients
    all_term.backward()
    gradLossB = b.grad
    b=b.detach().clone()
    
    # update the parameters
    b -= delta*gradLossB
    
    #stopping criteria
    if np.abs(prev_all_term-all_term.item())<1e-15:
        break
    prev_all_term= all_term.item()
        
    if i % 1000 == 0:
        print(i, all_term.item(),sse_term.item(),sb_term.item())


# look at results
yhat=0
for i in range(len(Ulist)):
    yhat += Ulist[i] @ b[:,i]
    
beta = torch.zeros(n_fg,p)
for i in range(p):
    beta[:,i] = fg_basis @ b[:,i]

pp.figure()
pp.scatter(response.detach(),yhat.detach())

pp.figure()
pp.plot(fg,beta.detach())
pp.xlabel("t (months)")
pp.ylabel("beta")
pp.title("Multiple Functional Linear Regression")


pp.figure()
pp.plot(b.detach())
pp.xlabel("basis function")
pp.ylabel("b")




###################################################################################
####################################################################################
##############################    ridge penalty fast   ###########
####################################################################################
#####################################################################################

#make big versions of matricies
response= torch.tensor(response,dtype=torch.float64)
U_big=Ulist[0]
D2_big=torch.tensor(D2,dtype=torch.float64)
Z_big=torch.diag(torch.tile(torch.arange(ns,0,-1, dtype=torch.float64),(p,)))
for i in range(1,p):
    U_big=torch.cat((U_big, Ulist[i]), 1)
    D2_big=torch.cat((D2_big,torch.tensor(D2,dtype=torch.float64)),1)

#initalize parameters
lam = 1e0
tau = 5e1
soft_t= 5e-2

# get A and d
d = torch.transpose(U_big,0,1) @ response
A = torch.transpose(U_big, 0, 1) @ U_big + lam*torch.transpose(D2_big,0,1) @ D2_big + tau * torch.transpose(Z_big,0,1) @ Z_big
b = torch.linalg.solve(A,d)
b = torch.transpose(torch.reshape(b,(p,ns)),0,1)

#apply soft thresholding
b[torch.abs(b)<soft_t]=0  

# look at results
fg_basis=get_basis_mat(order,fg,knots)
fg_basis=torch.tensor(fg_basis,requires_grad=False,dtype=torch.float64)

yhat=0
for i in range(len(Ulist)):
    yhat += Ulist[i] @ b[:,i]
    
beta = torch.zeros(n_fg,p)
for i in range(p):
    beta[:,i] = fg_basis @ b[:,i]

pp.figure()
pp.scatter(response.detach(),yhat.detach())

pp.figure()
pp.plot(fg,beta.detach())
pp.xlabel("t (months)")
pp.ylabel("Beta")
pp.title("Frobenius Penalty with Soft Thresholding")


pp.figure()
pp.plot(b.detach())
pp.xlabel("basis function")
pp.ylabel("b")


###################################################################################
####################################################################################
##############################    normal fast   ####################################
####################################################################################
#####################################################################################

#make big versions of matricies
response= torch.tensor(response,dtype=torch.float64)
U_big=Ulist[0]
D2_big=torch.tensor(D2,dtype=torch.float64)
for i in range(1,p):
    U_big=torch.cat((U_big, Ulist[i]), 1)
    D2_big=torch.cat((D2_big,torch.tensor(D2,dtype=torch.float64)),1)

#initalize parameters
lam = 1e-5

# get A and d
d = torch.transpose(U_big,0,1) @ response
A = 0*torch.transpose(U_big, 0, 1) @ U_big + lam*torch.transpose(D2_big,0,1) @ D2_big
b = torch.linalg.solve(A,d)
b = torch.transpose(torch.reshape(b,(p,ns)),0,1)

# look at results
fg_basis=get_basis_mat(order,fg,knots)
fg_basis=torch.tensor(fg_basis,requires_grad=False,dtype=torch.float64)

yhat=0
for i in range(len(Ulist)):
    yhat += Ulist[i] @ b[:,i]
    
beta = torch.zeros(n_fg,p)
for i in range(p):
    beta[:,i] = fg_basis @ b[:,i]

pp.figure()
pp.scatter(response.detach(),yhat.detach())

pp.figure()
pp.plot(fg,beta.detach())
pp.xlabel("t (months)")
pp.ylabel("Beta")
pp.title("Multiple Functional Linear Regression")


pp.figure()
pp.plot(b.detach())
pp.xlabel("basis function")
pp.ylabel("b")





