from Car import Car
from NMPC_ import NMPC
import numpy as np
import casadi as ca
from LinearKF import LinearKF

def main():
    Q = ca.diagcat(100, 100, 100, 10, 10)
    R = ca.diagcat(1, 1)
    n = 5   # number of states
    m = 2   # number of controls
    N = 100 # prediction horizon
    Ts = 0.1
    
    # Instantiate Car MPC object
    CarMPC = NMPC(n, m, N, Ts, Q, R)
    
    # Define symbolic states
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    phi = ca.SX.sym('phi')
    v = ca.SX.sym('v')
    delta = ca.SX.sym('delta')
    states = ca.vertcat(x, y, phi, v, delta)
    
    # Define symbolic controls
    w = ca.SX.sym('w')
    a = ca.SX.sym('a')
    controls = ca.vertcat(w, a)
    
    # System dynamics (RHS) 
    RHS = ca.vertcat(v * ca.cos(phi),
                     v * ca.sin(phi),
                     v/2 *ca.tan(delta),
                     w,
                     a)
    
    # Initial state & target 
    state0 = np.array([-5, -11., np.deg2rad(50), 0, 0])
    stateRef = np.array([6, 6, 0, 0, 0])
    
    # Initialize the MPC with the provided states, controls and model
    CarMPC.stZeroRef(state0, stateRef)
    CarMPC.createModel(states, controls, RHS)
    
    # Constraints on states & controls
    lbx = [-ca.inf,  -ca.inf, -np.pi/2, -0.5,   np.deg2rad(-44)]
    ubx = [ ca.inf,   ca.inf,  np.pi/2,  0.7,   np.deg2rad(44)]
    lbu = [-np.pi/3, -0.1]
    ubu = [ np.pi/3,  0.7]
    
    CarMPC.Constraints(ubx, lbx, ubu, lbu)
    
    # Set up the controller for point control with dynamic obstacles
    CarMPC.advancedPointCtrl(2, 10000)
    
    
    
    # Dummy Car #1 with sensor
    dummyX, dummyY, dummyPhi = [-5, 10, -0.5]
    dummyCar = Car(dummyX, dummyY, dummyPhi, length=2, width=1, color='blue')
    sensor = LinearKF(dim_x=4, dim_z=2)
    sensor.F = np.array([[1,0, Ts,0],
                         [0,1, 0, Ts],
                         [0,0, 1,  0],
                         [0,0, 0,  1]])
    sensor.B = np.zeros((4,0))
    sensor.Q = np.eye(4)*0.01
    sensor.H = np.array([[1,0,0,0],
                         [0,1,0,0]])
    sensor.R = np.eye(2)*0.5
    sensor.X0 = np.array([dummyCar.state[0],  
                          dummyCar.state[1],  
                          0.2,  # initial vx guess
                          0.1]) # initial vy guess
    sensor.P = np.eye(4)*10.0
    
    
    # Dummy Car #2 with sensor
    dummyX2, dummyY2, dummyPhi2 = [5, -10, 2.3]
    dummyCar2 = Car(dummyX2, dummyY2, dummyPhi2, length=2, width=1, color='green')
    sensor2 = LinearKF(dim_x=4, dim_z=2)
    sensor2.F = np.array([[1,0, Ts,0],
                          [0,1, 0, Ts],
                          [0,0, 1,  0],
                          [0,0, 0,  1]])
    sensor2.B = np.zeros((4,0))
    sensor2.Q = np.eye(4)*0.01
    sensor2.H = np.array([[1,0,0,0],
                          [0,1,0,0]])
    sensor2.R = np.eye(2)*0.5
    sensor2.X0 = np.array([dummyCar2.state[0],  
                           dummyCar2.state[1],  
                           0.2,
                           0.1])
    sensor2.P = np.eye(4)*10.0
    
    
    
    
    # ----------------------- Simulation Loop ------------------------------------- #
    
    total_steps = 600  # 
    
    for step in range(total_steps):
        dummyCar.propagate([0.35, 0], Ts)
        real_x, real_y, real_phi = dummyCar.get_state()
        meas_x = real_x + np.random.normal(0, 0.1)
        meas_y = real_y + np.random.normal(0, 0.1)
        measurement = np.array([meas_x, meas_y])
        sensor.Predict(u=np.array([]))
        x_upd, _ = sensor.Update(measurement)
        
        
        dummyCar2.propagate([0.60, 0], Ts)
        real_x2, real_y2, real_phi2 = dummyCar2.get_state()
        meas_x2 = real_x2 + np.random.normal(0, 0.1)
        meas_y2 = real_y2 + np.random.normal(0, 0.1)
        measurement2 = np.array([meas_x2, meas_y2])
        sensor2.Predict(u=np.array([]))
        x_upd2, _ = sensor2.Update(measurement2)
        
        
        CarMPC.updateObserve(np.hstack((x_upd, x_upd2)))
        
        states_sol, controls_sol = CarMPC.solveProblem()
        
        
        
    
if __name__ == '__main__':
    main()


    
    