from fenics import *
import matplotlib.pyplot as plt

mesh = IntervalMesh(20, 0.0, 1.0)
V = FunctionSpace(mesh, 'P', 1)

u_D = Expression('0', degree=0)


def boundary(x):
    tol = 1E-10
    return abs(x[0]) < tol or abs(x[0]-1) < tol


bc = DirichletBC(V, u_D, boundary)

u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

u = Function(V)
solve(a == L, u, bc)

plot(u)
plot(mesh)
plt.show()
