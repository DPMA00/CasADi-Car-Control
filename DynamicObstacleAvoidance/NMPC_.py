import numpy as np
import casadi as ca
from time import time


class NMPC:
    def __init__(self, n, m, N, Ts, Q, R):
        """ 
        Initialize MPC parameters
        
        :param n(int): nr states
        :param m(int): nr controls
        :param N(int): prediction horizon
        :param Q(np.array): state penalty matrix --> positive semidefinite (n x n)-matrix
        :param R(np.array): control penalty matrix --> positive semidefinite (n x m) - matrix
        
        """
        self.n = n
        self.m = m
        self.N = N
        self.Ts =  Ts
        self.Q = Q
        self.R = R
        
        self.X = ca.SX.sym('X',n, N+1) # N+1 State predictions
        self.U = ca.SX.sym('U',m, N) # N control predictions
        
        self.nrDynamicObstacles = 2
    
    def Constraints(self, upperXs,lowerXs,upperUs,lowerUs):
        self.upperXs = upperXs
        self.lowerXs = lowerXs
        self.upperUs = upperUs
        self.lowerUs = lowerUs
        
    

        
    def createModel(self, states, controls, RHS):
        """
        This function defines the nonlinear system dynamics symbolically
        
        :param states(ca.SX.sym): symbolic state vector --> e.g. ca.vertcat(ca.SX.sym('x'),ca.SX.sym('y'))
        :param controls(ca.SX.sym): symbolic control vector --> see above
        :param RHS(ca.SX.sym): time derivatives of the state vector: e.g. for x,y:
                                                    ca.vertcat(v*cos(phi),v*sin(phi))
                                                    
        
        Returns:
            - CasADi function that maps current states and controls to the RHS
            
        """

        self.f = ca.Function('f', [states,controls], [RHS])
        
    def sim_step(self, x0, t, Ts, u, f):
        """
        This function simulates the MPC one time step by:
            - Advancing the systems states by one time step by applying the first control input
            - Updates te time for the next MPC iteration
        
        :param x0(np.array): current state --> np.array([x0,y0,phi0])
        :param t(float): current time --> float
        :param T(float): sampling time --> float
        :param u(np.array): control sequence --> m x N np.array
        :param f(ca.Function): nonlinear mapping function 
        
        Returns
            - updated time
            - updated states
        
        """
        
        fval = self.f(x0, u[:,0]) # first control is applied to RHS
        st = ca.DM.full(x0 + (Ts*fval)) #explicit shift to next state
        
        t = t + Ts
       
        return t, st
    
    
    def updateObserve(self, carPos):
        """
        Function updates the states of multiple (or single) obstacles
        
        :param carPos(np.array): requires obstacle x, y, vx, vy information for each obstacle
        
        """
        self.Observ = carPos
        
        
    def advancedPointCtrl(self, nrDynamicObstacles, avoidancePenalty):
        """
        This function sets up the NLP problem for scenarios 
        where the objective is to track or stabilize towards a specific point or target
        
        It incorporates dynamic constraints for the purpose of handling dynamic obstacles
        
        Dynamic object positions are predicted using a *CONSTANT & LINEAR* 2D velocity model                
            
        :param nrDynamicObstacles(int): number of dynamic obstacles
        :param avoidancePenalty(int/float): penalty on the avoidance of dynamic obstacles
        
        """
        # Iinitialize a symbolic parameter vector
        self.P = ca.SX.sym('P', self.n + self.n + 4 * nrDynamicObstacles) 
        
        # Extract obstacle positions and velocities from P
        dummy_x0 = self.P[self.n + self.n::4]  # x positions of all obstacles
        dummy_y0 = self.P[self.n + self.n + 1::4]  # y positions of all obstacles
        dummy_vx = self.P[self.n + self.n + 2::4]  # vx of all obstacles
        dummy_vy = self.P[self.n + self.n + 3::4]  # vy of all obstacles
        
        ### Multiple Shooting ###
        self.g = self.X[:, 0] - self.P[:self.n]  # Initial state constraint
        self.obj = 0  # Initialize objective function
        
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            self.obj += (st - self.P[self.n:2*self.n]).T @ self.Q @ (st - self.P[self.n:2*self.n]) #target tracking penalty
            self.obj += con.T @ self.R @ con
            st_next = self.X[:, k+1]
            k1 = self.f(st, con)
            k2 = self.f(st + self.Ts/2*k1, con)
            k3 = self.f(st + self.Ts/2*k2, con)
            k4 = self.f(st + self.Ts * k3, con)
            RK4 = st + (self.Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.g = ca.vertcat(self.g, st_next - RK4)
        
        # Collision constraints for multiple obstacles
        for k in range(self.N + 1):
            for obs in range(nrDynamicObstacles):  # Obstacles
                # Predict obstacle position at step k
                x_obs_k = dummy_x0[obs] + dummy_vx[obs] * k * self.Ts
                y_obs_k = dummy_y0[obs] + dummy_vy[obs] * k * self.Ts
                
                # Squared distance between MPC car and obstacle
                dist_sq = (x_obs_k - self.X[0, k])**2 + (y_obs_k - self.X[1, k])**2
                
                # Add collision constraint
                self.g = ca.vertcat(self.g, dist_sq)
                
                # Add penalty to objective function
                self.obj += avoidancePenalty * 1/(dist_sq + 1e-5)
        
        ### NLP Parameters ###
        self.opt_var = ca.vertcat(self.X.reshape((-1, 1)), self.U.reshape((-1, 1))) 
        
        self.nlp_prob = {
            'f': self.obj,
            'x': self.opt_var,
            'g': self.g,
            'p': self.P
        }
        
        # Solver options
        self.opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }
        
        # Set up the NLP solver
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp_prob, self.opts)
        
        ### Constraint Bounds ###
        total_constraints = self.n * (self.N + 1) + nrDynamicObstacles * (self.N + 1)
        self.lbg = ca.DM.zeros((total_constraints, 1))
        self.ubg = ca.DM.zeros((total_constraints, 1))
        
        # Dynamics constraints
        self.lbg[:self.n * (self.N + 1)] = 0
        self.ubg[:self.n * (self.N + 1)] = 0
        
        # Collision constraints
        self.lbg[self.n * (self.N + 1):] = 4  # Minimum distance squared (r+r_obstacle)**2
        self.ubg[self.n * (self.N + 1):] = ca.inf
        
        # State and control bounds
        self.lb_state_const = ca.DM.zeros((self.n * (self.N + 1), 1))
        self.ub_state_const = ca.DM.zeros((self.n * (self.N + 1), 1))
        self.lb_cont_const = ca.DM.zeros((self.m * self.N, 1))
        self.ub_cont_const = ca.DM.zeros((self.m * self.N, 1))
        
        # States
        for i in range(self.n):
            self.lb_state_const[i: self.n * (self.N + 1): self.n] = self.lowerXs[i]    
            self.ub_state_const[i: self.n * (self.N + 1): self.n] = self.upperXs[i] 
        
        # Controls
        for i in range(self.m):
            self.lb_cont_const[i: self.m * self.N: self.m] = self.lowerUs[i]  
            self.ub_cont_const[i: self.m * self.N: self.m] = self.upperUs[i]  
    
        self.lbx = ca.vertcat(self.lb_state_const, self.lb_cont_const)
        self.ubx = ca.vertcat(self.ub_state_const, self.ub_cont_const)
    
        self.args = {
            'lbg': self.lbg,  # constraints lower bound
            'ubg': self.ubg,  # constraints upper bound
            'lbx': self.lbx,
            'ubx': self.ubx
        }
        
        
    def stZeroRef(self, state_init, x_ref=None):
        self.state_init = state_init
        self.x_ref = x_ref
        
        # Runtime constants and other information
        self.t0 = 0
        self.mpc_iter = 0
        self.u_init = ca.DM.zeros((self.m,self.N))
        
        self.t = ca.DM(self.t0)  # time history
        self.X0 = ca.repmat(self.state_init, 1, self.N+1)
        self.times = np.array([[0]])
        
    
    
    def solveProblem(self):
        # Update the parameter vector `p` with current state, reference, and obstacle states
        self.args['p'] = ca.vertcat(
            self.state_init,    # Current state
            self.x_ref,         # Reference state
            self.Observ         # Obstacle states
        )
        
        # Warm-start the solver with the previous solution
        self.args['x0'] = ca.vertcat(
            ca.reshape(self.X0, self.n*(self.N+1), 1),  # Warm-started states
            ca.reshape(self.u_init, self.m*self.N, 1)   # Warm-started controls
        )
        
        # Solve the NLP
        sol = self.solver(
            x0=self.args['x0'],  # Initial guess (warm-started)
            lbx=self.args['lbx'],
            ubx=self.args['ubx'],
            lbg=self.args['lbg'],
            ubg=self.args['ubg'],
            p=self.args['p']
        )
        
        # Extract the optimal solution
        u_opt = ca.reshape(sol['x'][self.n * (self.N + 1):], self.m, self.N)  # Optimal controls
        X0_opt = ca.reshape(sol['x'][: self.n * (self.N+1)], self.n, self.N+1)  # Optimal states
        
        self.horizon = X0_opt # predictions
                
        # # Shift the optimal solution for warm-start
        self.X0 = ca.horzcat(
            X0_opt[:, 1:],  # Discard the first state
            ca.reshape(X0_opt[:, -1], -1, 1)  # Duplicate the last state
        )
        self.u_init = ca.horzcat(
            u_opt[:, 1:],  # Discard the first control
            ca.reshape(u_opt[:, -1], -1, 1)  # Duplicate the last control
        )
        
        # Update state and time
        self.t0, self.state_init = self.sim_step(self.state_init, self.t0, self.Ts, u_opt, self.f)
        
        self.mpc_iter += 1
        print(self.mpc_iter)
        
        return self.state_init, u_opt[:, 0]
    
    