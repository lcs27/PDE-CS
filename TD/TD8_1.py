# This Python program contains the program for Q.VIII.2
import numpy as np


def upperlisation(A):
    n = np.size(A,0)
    if n != np.size(A,1):
        raise ValueError('Not a square matrix')
    U = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(i,n,1):
            U[i,j]=A[i,j]
    return U

def lowerlisation(A):
    n = np.size(A,0)
    if n != np.size(A,1):
        raise ValueError('Not a square matrix')
    L = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(i+1):
            L[i,j]=A[i,j]
    return L

def solve_Lx(L,b):
    n = np.size(L,0)
    if n != np.size(L,1):
        raise ValueError('Not a square matrix')
    x = np.zeros(shape=(n,1))
    for i in range(n):
        sum = 0
        for j in range(i):
            sum += L[i,j]*x[j,0]
        x[i,0] = (b[i] - sum) / L[i,i]
    return x
    

def solve_Ux(U,b):
    n = np.size(U,0)
    if n != np.size(U,1):
        raise ValueError('Not a square matrix')
    x = np.zeros(shape=(n,1))
    for i in range(n-1,-1,-1):
        sum = 0
        for j in range(n-1,i,-1):
            sum += U[i,j]*x[j,0]
        x[i,0] = (b[i] - sum) / U[i,i]
    return x

if __name__ == '__main__':
    for n in range(3,200):
        A = np.random.randint(0,20,size=[n,n])
        b = np.random.randint(0,20,size=[n,1])
        L = lowerlisation(A)
        U = upperlisation(A)
        #print(b)
        #print(np.dot(L,solve_Lx(L,b)))
        #print(np.dot(U,solve_Ux(U,b)))
        assert b.any() == np.dot(L,solve_Lx(L,b)).any()
        assert b.any() == np.dot(U,solve_Ux(U,b)).any()
        print('n=',n,'passed')