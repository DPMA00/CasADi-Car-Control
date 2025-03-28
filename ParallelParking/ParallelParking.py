import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Constants, car parameters 
l = 0.256

GG = 9.81 #ms^-2 

# acceleration and steering rate limits 
a_max = 0.3*GG
a_min = -0.1*GG

deltaD_max = np.pi/8
deltaD_min = -np.pi/8

min_clearance = 18/100
min_distance= 0.1308+min_clearance

# reference pose
xpose = 3
ypose = 3
phipose = np.pi/2
deltaPose = 0
velocityPose = 0

################# MPC parameters ###################
N = 100
Ts = 0.05


Q_rx = 10000
Q_ry = 500
Q_rphi = 150
Q_rdelta = 10
Q_rv = 0

Q_r = ca.diagcat(Q_rx,Q_ry,Q_rphi,Q_rdelta,Q_rv)

R_raccel = 3
R_rdelta = 3

R_r = ca.diagcat(R_raccel,R_rdelta)







def shift_step(x0,t,T,u,f):
    fval = f(x0, u[:,0]) # apply the first control to RHS
    st = ca.DM.full(x0 + (T*fval)) #explicit shift to next state
    
    t = t + T #update timestep
    
    u_guess = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1) #trim initial input and duplicate last(best guess)
    )
   
    
    return t, st, u_guess #st provides initial guess for states, controls (multiple shooting)

def DM2Arr(dm):
    return np.array(dm.full())

##### STATES #####

x = ca.SX.sym('x')
y = ca.SX.sym('y')
phi = ca.SX.sym('phi')
delta = ca.SX.sym('delta')
v = ca.SX.sym('v')


states = ca.vertcat(x,
                    y,
                    phi,
                    delta,
                    v)
n = states.numel()




##### CONTROLS ####

accel = ca.SX.sym('a')
deltaDot = ca.SX.sym('deltaDot')

controls = ca.vertcat(accel,deltaDot)
m = controls.numel()



# Dynamics
RHS = ca.vertcat(v*ca.cos(phi),
                 v*ca.sin(phi),
                 v/l * ca.tan(delta),
                 deltaDot,
                 accel)

f = ca.Function('f', [states,controls], [RHS])



X = ca.SX.sym('X', n, N+1) # Predictive states in the obj function :  n x N+1
U = ca.SX.sym('U',m, N) # Decision variables in the obj function : m x N
P = ca.SX.sym('P',n+n) # States and refernce values (include initial states)

obstR = 0.1308
Car1_x = 3
Car1_y = 3+(2*obstR+min_clearance)

Car2_x = 3
Car2_y = 3-(2*obstR+min_clearance)






# discretization (multiple shooting)
g = X[:,0]-P[:n]
obj = 0
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    obj = obj + (st - P[n:]).T @ Q_r @ (st - P[n:]) + con.T @ R_r @ con
    st_next = X[:, k+1]
    k1 = f(st, con)
    k2 = f(st + Ts/2*k1, con)
    k3 = f(st + Ts/2*k2, con)
    k4 = f(st + Ts * k3, con)
    RK4 = st + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = ca.vertcat(g, st_next - RK4)




# add collision constraints
for k in range(N+1):
    EuclDist1 = np.sqrt((Car1_x-X[0,k])**2+(Car1_y-X[1,k])**2)
    EuclDist2 = np.sqrt((Car2_x-X[0,k])**2+(Car2_y-X[1,k])**2)
    g = ca.vertcat(g,EuclDist1,EuclDist2)
   
#NLP params
opt_var = ca.vertcat(X.reshape((-1,1)),
                     U.reshape((-1,1)))

nlp_prob = {
    'f': obj,
    'x': opt_var,
    'g': g,
    'p': P
    }

opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = ca.nlpsol('solver','ipopt',nlp_prob,opts)


lbg = ca.DM.zeros((n*(N+1) + 2*(N+1), 1)) ## with obst
ubg = ca.DM.zeros((n*(N+1) + 2*(N+1), 1)) ## with obst

lbg[n*(N+1):] = 2*obstR+0.02## with obst
ubg[n*(N+1):] = ca.inf ## with obs

# lbg = ca.DM.zeros((n*(N+1), 1)) # no obst
# ubg = ca.DM.zeros((n*(N+1), 1))
 
lb_state_const = ca.DM.zeros((n*(N+1),1))
lb_cont_const = ca.DM.zeros((m*N,1))
ub_state_const = ca.DM.zeros((n*(N+1),1))
ub_cont_const = ca.DM.zeros((m*N,1))

lb_state_const[0: n*(N+1): n] = -ca.inf     # X lower bound
lb_state_const[1: n*(N+1): n] = -ca.inf    # Y lower bound
lb_state_const[2: n*(N+1): n] = -ca.inf     # phi lower bound
lb_state_const[3: n*(N+1): n] = -np.deg2rad(44)    # delta lower bound
lb_state_const[4: n*(N+1): n] = -0.3   

ub_state_const[0: n*(N+1): n] = ca.inf      # X upper bound
ub_state_const[1: n*(N+1): n] = ca.inf      # Y upper bound
ub_state_const[2: n*(N+1): n] = ca.inf      # phi upper bound
ub_state_const[3: n*(N+1): n] = np.deg2rad(44)    # delta upper bound
ub_state_const[4: n*(N+1): n] = 0.6

lb_cont_const[0: m*N: m] = a_min
lb_cont_const[1: m*N: m] = deltaD_min

ub_cont_const[0: m*N: m] = a_max
ub_cont_const[1: m*N: m] = deltaD_max


lbx = ca.vertcat(lb_state_const,lb_cont_const)
ubx = ca.vertcat(ub_state_const,ub_cont_const)

args = {
    'lbg': lbg,  # constraints lower bound
    'ubg': ubg,  # constraints upper bound
    'lbx': lbx,
    'ubx': ubx
}




t0 = 0
state_init = ca.DM([2.6,3.8,np.deg2rad(90),0,0])
x_ref = ca.DM([xpose,ypose,phipose,deltaPose,velocityPose])


t = ca.DM(t0)  # time history
X0 = ca.repmat(state_init, 1, N+1)
u_init = ca.DM.zeros((m,N))

simtime = 15

mpc_iter = 0

cat_states = DM2Arr(X0) # state history
cat_controls = DM2Arr(u_init[:, 0]) # control history (1st one)
times = np.array([[0]])


prediction = [ca.DM.zeros((n,N))]

checkConst = []


if __name__ == "__main__":
    main_loop = time()
    while(mpc_iter < simtime/Ts):
        sol = solver()
        t1 = time()
        args['p'] = ca.vertcat(
                state_init,    # current state
                x_ref   # reference
            )
            # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n*(N+1), 1),
            ca.reshape(u_init, m*N, 1)
        )
    
        sol = solver(               #initial sates,reference states, constraints..
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )
        
        checkConst.append(sol['g'][n*(N+1):])
        
        u = ca.reshape(sol['x'][n * (N + 1):], m, N) #
        X0 = ca.reshape(sol['x'][: n * (N+1)], n, N+1)
        
        prediction.append(X0)
        
        cat_states = np.dstack((cat_states,
            DM2Arr(X0)))
    
        cat_controls = np.vstack((cat_controls,
            DM2Arr(u[:, 0])))
        
        t = np.vstack((
            t,
            t0
        ))
    
        t0, state_init, u_init = shift_step(state_init, t0,Ts,u,f)
    
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )
    

        t2 = time()
        print(mpc_iter)
        print(t2-t1)
        times = np.vstack((times,
            t2-t1))
    
        mpc_iter = mpc_iter + 1
    
        main_loop_time = time()

    
    #### Plotting
    
    xses = []
    yses = []
    phises = []
    deltas = []
    vses = []
    
    pred_x = []
    pred_y = []
    pred_phi = []
    
    for k in range(len(prediction)):
        preds = prediction[k]
        preds = preds.full()
        pred_x.append(preds[0])
        pred_y.append(preds[1])
        
    
    for k in range(cat_states.shape[2]):
        xses.append(cat_states[0,0,k])
        yses.append(cat_states[1,0,k])
        phises.append(cat_states[2,0,k])
        deltas.append(cat_states[3,0,k])
        vses.append(cat_states[4,0,k])
        
    
    # Errors
    e_x = abs(xses[len(xses)-1]-xpose)
    e_y = abs(yses[len(xses)-1]-ypose)
    e_psi = abs(phises[len(xses)-1]-phipose)
    e_delta = abs(deltas[len(xses)-1]-deltaPose)
    e_v = abs(vses[len(xses)-1]-velocityPose)
    
    #Average solving time
    avg_solTime = np.average(times) 
    
    # Controls
    reshapedControls = np.reshape(cat_controls,(-1,2))
    ases = reshapedControls[:,0]
    deltaDs = reshapedControls[:,1]
    

    plt.title('Car Trajectory - Prediction Horizon 5s')
    plt.plot(xses,yses,'black', linestyle='--',label='Trajectory')
    plt.ylabel('y(m)')
    plt.xlabel('x(m)')
    plt.scatter(xpose,ypose,c = 'blue', marker='o',s=100, label='Setpoint')
    plt.scatter(Car1_x,Car1_y,c = 'red', marker='o',s=100, label='Car Obstacle 1')
    plt.scatter(Car2_x,Car2_y,c = 'gold', marker='o',s=100, label='Car Obstale 2')
    plt.text(0.8, 0.7, f'$e_x$: {e_x:.3f} $m$', transform=plt.gca().transAxes)
    plt.text(0.8, 0.65, f'$e_y$: {e_y:.3f} $m$', transform=plt.gca().transAxes)
    plt.text(0.8, 0.6, f'$e_{{\psi}}$: {e_psi:.3f} $rad$', transform=plt.gca().transAxes)
    plt.text(0.8, 0.55, f'$e_{{\delta}}$: {e_delta:.3f} $rad$', transform=plt.gca().transAxes)
    plt.text(0.8, 0.5, f'$e_v$: {e_v:.3f} $\\frac{{m}}{{s}}$', transform=plt.gca().transAxes)
    plt.legend()
    plt.tight_layout()
    plt.xlim((2.5,3.5))
    plt.ylim((2.5,4))
    plt.savefig('ExtendedTrajectory.PNG',dpi=300)
    plt.show()
    
    
    
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].step(t,deltaDs, 'b-')
    axes[0].set_xlabel('time$(s)$')
    axes[0].set_ylabel('$\\omega(\\frac{rad}{s})$')
    axes[0].set_title('Steering Rate')
    
    axes[1].step(t,ases, 'g-')
    axes[1].set_xlabel('time$(s)$')
    axes[1].set_ylabel('$a(\\frac{m}{s^2})$')
    axes[1].set_title('Acceleration')
    plt.tight_layout()
    plt.savefig('ExtendedTrajectory_Controls.PNG',dpi=300)
    plt.show()

    



