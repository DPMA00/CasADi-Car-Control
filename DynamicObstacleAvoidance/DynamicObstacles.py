# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:35:28 2025

@author: diete
"""

from Car import Car
from NMPC import NMPC
import numpy as np
import casadi as ca
import time as time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.animation import FuncAnimation,FFMpegWriter
from LinearKF import LinearKF
import matplotlib.animation as animation

# Set the path to ffmpeg.exe
path = 'C:\\Users\\diete\\OneDrive\\Documents\\ffmpeg\\ffmpeg-2025-01-27-git-959b799c8d-essentials_build\\bin\\ffmpeg.exe'

plt.rcParams['animation.ffmpeg_path'] = path 

Q = ca.diagcat(100,100,100,10,10)
R = ca.diagcat(1,1)
n = 5
m = 2
N = 100
Ts = 0.1
CarMPC = NMPC(n,m,N,Ts,Q,R)


x = ca.SX.sym('x')
y = ca.SX.sym('y')
phi = ca.SX.sym('phi')
v = ca.SX.sym('v')
delta = ca.SX.sym('delta')

states = ca.vertcat(x,
                    y,
                    phi,
                    v,
                    delta)

w = ca.SX.sym('w')
a = ca.SX.sym('a')

controls = ca.vertcat(w,
                      a)

RHS = ca.vertcat(v*ca.cos(phi),
                 v*ca.sin(phi),
                 v/2 * ca.tan(delta),
                 w,
                 a)


state0 = np.array([-5,-11.,np.deg2rad(50),0,0])
stateRef = np.array([6,6,0,0,0])
CarMPC.stZeroRef(state0,stateRef)
CarMPC.createModel(states,controls,RHS)

lbx = [-ca.inf, -ca.inf,-np.pi/2,-0.5,np.deg2rad(-44)]
ubx = [ca.inf, ca.inf,np.pi/2,0.7,np.deg2rad(44)]

lbu = [-np.pi/3,-0.1]
ubu =  [np.pi/3,0.7]
CarMPC.Constraints(ubx, lbx, ubu, lbu)


#CarMPC.advancedPointCtrl(2,10000) #10000 with 1/1e-3
CarMPC.advancedPointCtrl(2,10000)
CtrlCar = Car(state0[0],state0[1],state0[2],length=2,width=1, color='red')




### Dummy 1 ###
dummyX,dummyY, dummyPhi = [-5, 10, -0.5]
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
                      0.2,                  
                      0.1])                 
sensor.P = np.eye(4)*10.0 

### Dummy 2 ###
dummyX2,dummyY2, dummyPhi2 = [5, -10, 2.3]
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



fig, ax = plt.subplots()
def init():
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect("equal")
    return []


def update(frame):
    
    ## Dummy 1 ##
    dummyCar.propagate([0.35, 0], Ts) 
    real_x, real_y, real_phi = dummyCar.get_state()

    meas_x = real_x + np.random.normal(0, 0.1)
    meas_y = real_y + np.random.normal(0, 0.1)
    measurement = np.array([meas_x, meas_y])
    
    
    x_pred, P_pred = sensor.Predict(u=np.array([]))
    x_upd, P_upd = sensor.Update(measurement)
    dummyPatch = dummyCar.get_patch("Dummy Car 1")
    
    ## Dummy 2 ##
    dummyCar2.propagate([0.60, 0], Ts) 
    real_x2, real_y2, real_phi2 = dummyCar2.get_state()

    meas_x2 = real_x2 + np.random.normal(0, 0.1)
    meas_y2 = real_y2 + np.random.normal(0, 0.1)
    measurement2 = np.array([meas_x2, meas_y2])
    
    
    x_pred2, P_pred2 = sensor2.Predict(u=np.array([]))
    x_upd2, P_upd2 = sensor2.Update(measurement2)
    dummyPatch2 = dummyCar2.get_patch("Dummy Car 2")
   

    #CarMPC.updateObserve(x_upd2)
    CarMPC.updateObserve(np.hstack((x_upd,x_upd2)))
    states, controls = CarMPC.solveProblem()
    
    CtrlCar.state = [states[0,0],states[1,0],states[2,0]]
    CtrlCarPatch = CtrlCar.get_patch("MPC Car")
    
    ax.clear()
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect("equal")
    #ax.tight_layout()
    
    ax.scatter(stateRef[0], stateRef[1], marker='x', color='red', label='Target')
    
    
    ax.add_patch(dummyPatch)
    c2 = Circle((real_x, real_y), 1, fill=False, linestyle='--', color='blue')
    ax.add_patch(c2)

    
    ax.scatter(x_upd[0], x_upd[1], color='k', label='KF update D1')
    c3 = Circle((x_upd[0], x_upd[1]), 1, fill=False, linestyle=':', color='k')
    ax.add_patch(c3)
    
    ax.add_patch(dummyPatch2)
    c4 = Circle((real_x2, real_y2), 1, fill=False, linestyle='--', color='blue')
    ax.add_patch(c4)

    
    ax.scatter(x_upd2[0], x_upd2[1], color='k', label='KF update D2')
    c5 = Circle((x_upd2[0], x_upd2[1]), 1, fill=False, linestyle=':', color='k')
    ax.add_patch(c5)

    ax.add_patch(CtrlCarPatch)
    c1 = Circle((states[0,0], states[1,0]), 1, fill=False, linestyle='--', color='red')
    ax.add_patch(c1)
    
    predX = CarMPC.X0.full()[0]
    predY = CarMPC.X0.full()[1]
    
    ax.plot(predX,predY, c='green', lw=0.5, marker='+', markersize=2)
    
    ax.legend(loc='upper left',prop={'size': 4})
    return []

ani = FuncAnimation(fig, update, frames=range(600), init_func=init, blit=False)


# writer = animation.PillowWriter(fps=50,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# ani.save('DynamicObstacleAvoidance.gif', writer=writer,dpi=100)

writer = FFMpegWriter(
    fps=50,  # Frame rate (30 frames per second)
    metadata=dict(artist='Me'),  # Metadata
    bitrate=1800  # Bitrate (1800 kbps)
)

# Save the animation with increased DPI
ani.save("movie15.mp4", writer=writer, dpi=300)  # Increase DPI to 300