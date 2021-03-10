import math
from dolfin import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from mshr import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True, help='The number of tasks')
args = parser.parse_args()
sample_Num = 20
task_number: int = args.task

def solve_stokes(X=10,Y=3,U0=5,phi=1,nx=100):
    '''
    The FEniCS solver used to solve this question for 
    
    Input
    ----------------------
    - (X,Y) the position of the object dropped, predefined to be the center, default to be (10,3)
    - U0 velocity of the river, defaut to be 5
    - phi the diameter of the object, defaut to be 1
    
    Output
    ----------------------
    - u, the velocity field(vector field)
    - p, the pressure field(scalar field)
    - V, the space of u
    - Q, the space of p
    - W, the space V*Q
    - mesh, the mesh of the field
    - boundary_subdomains, the boundary subdomains
    '''
    # Scaled variables
    D = 20; H = 6; mu =8.9*1e-4
    nx = 500; order = 1

    # Create mesh and define function space
    base = Rectangle(Point(0, 0), Point(D, H))
    hole = Circle(Point(X, Y), phi/2)
    mesh = generate_mesh(base - hole, nx)

    # Define the mixed function space
    Element1 = VectorElement('P', mesh.ufl_cell(), order)
    Element2 = FiniteElement( 'P', mesh.ufl_cell(), order)
    W_elem = MixedElement([Element1, Element2])
    V = FunctionSpace(mesh,Element1)
    Q = FunctionSpace(mesh,Element2)
    W = FunctionSpace(mesh, W_elem)

    # Define boundary condition
    tol = 1E-10

    inflow =Expression(('A*4*x[1]*(B-x[1])/B/B','0'),A=U0,B=H,degree=2)
    def left_right_boundary(x, on_boundary):
        return on_boundary and (abs(x[0])<tol or abs(x[0] - D)<tol)
    bc_LR = DirichletBC(W.sub(0), inflow ,left_right_boundary)

    def top_bottom_boundary(x, on_boundary):
        return on_boundary and (abs(x[1])<tol or abs(x[1] - H)<tol)
    bc_TB = DirichletBC(W.sub(0), Constant((0,0)),top_bottom_boundary)

    def circle_boundary(x, on_boundary):
        r = math.sqrt((x[0]-X)*(x[0]-X)+(x[1]-Y)*(x[1]-Y))
        return on_boundary and abs(r-phi/2)<(0.5/nx)
    bc_CR = DirichletBC(W.sub(0), Constant((0,0)),circle_boundary)

    bc = [bc_LR,bc_TB,bc_CR]
    
    # Define boundary subdomaines
    boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_subdomains.set_all(0)
    AutoSubDomain(left_right_boundary).mark(boundary_subdomains, 1)
    AutoSubDomain(top_bottom_boundary).mark(boundary_subdomains, 2)
    AutoSubDomain(circle_boundary).mark(boundary_subdomains, 3)

    # Define variational formula
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    a = -mu*inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
    f = inner(Constant((0, 0)), v)*dx

    # Construction of preconditioner matrix
    b = -mu*inner(grad(u), grad(v))*dx + p*q*dx
    A, bb = assemble_system(a, f, bc)
    P, btmp = assemble_system(b, f, bc)
    solver = KrylovSolver("tfqmr", "amg")
    solver.set_operators(A, P)

    # Solve
    U = Function(W)
    solver.solve(U.vector(), bb)
    u, p = U.split()
    u = project(u,V)
    p = project(p,Q)
    return u,p,V,Q,W,mesh,boundary_subdomains

def plot_field(u,mesh,tit='',labelx='',labely=''):
    '''
    To plot a field(scalar or vector)
    
    Input
    ----------------------
    - u, the field
    - mesh, the mesh of the domaine
    - tit, the title, default to be vide string
    - labelx, the xlabel, default to be vide string
    - labely, the ylabel, default to be vide string
    '''
    plt.figure()
    plot(mesh,alpha=0.05)
    plu=plot(u)
    plt.colorbar(plu,fraction=0.2, pad=0.04)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(tit)

def calculate_force(p,mesh,boundary_subdomains,H=1):
    '''
    To calculate the force applied on the object
    
    Input
    ------------------
    - p, the pressure field
    - mesh, the mesh of the domaine
    - boundary_subdomains
    - H, the height of the object, defaut to be 1
    
    Output
    ------------------
    - force, the force on the object
    '''
    dss = ds(subdomain_data=boundary_subdomains)
    n = FacetNormal(mesh)
    ex  = Constant((1,0))
    f = p*dot(n, ex)* dss(3)
    force = H*assemble(f)
    return force

U0s = np.random.normal(5, 1, sample_Num)
force1 = []
max_velocity1 = []

#for each sample, calculate the corresponding number
k = 1
for U in U0s:
    u,p,V,Q,W,mesh,boundary_subdomains = solve_stokes(U0=U,nx=1000)

    u_mag=sqrt(dot(u,u))
    u_mag=project(u_mag,FunctionSpace(mesh,'P',1))
    u_array=np.array(u_mag.vector())
    force1.append(calculate_force(p,mesh,boundary_subdomains))
    max_velocity1.append(np.max(u_array))
    print(k,end=',')
    k += 1

np.savetxt("./project/results/U0s"+str(task_number)+".txt",U0s)
np.savetxt("./project/results/force_U0s"+str(task_number)+".txt",force1)
np.savetxt("./project/results/max_velocity_U0s"+str(task_number)+".txt",max_velocity1)

Ys = np.random.normal(3, 0.5, sample_Num)
force = []
max_velocity = []

#for each sample, calculate the corresponding number
k = 1
for Y in Ys:
    u,p,V,Q,W,mesh,boundary_subdomains = solve_stokes(Y=Y,nx=1000)

    u_mag=sqrt(dot(u,u))
    u_mag=project(u_mag,FunctionSpace(mesh,'P',1))
    u_array=np.array(u_mag.vector())
    force.append(calculate_force(p,mesh,boundary_subdomains))
    max_velocity.append(np.max(u_array))
    print(k,end=',')
    k += 1

np.savetxt("./project/results/Ys"+str(task_number)+".txt",U0s)
np.savetxt("./project/results/force_Ys"+str(task_number)+".txt",force1)
np.savetxt("./project/results/max_velocity_Ys"+str(task_number)+".txt",max_velocity1)