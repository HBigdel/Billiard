#Copyright (c) <year>, <copyright holder>
#All rights reserved.

#This source code is licensed under the BSD-style license found in the
#LICENSE file in the root directory of this source tree. 
#Owner: Hadi Bigdely (h.bigdely at marianopolis.edu)
# importing libraries
import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from random import uniform
import math
from sympy import sin, cos, pi
from matplotlib.animation import FuncAnimation

################################################

# give the value of n
n=6
#give the angle of first hit (0 < Alpha < 2*pi)
#Alpha=0.66*pi
num_frames_step=10
#Alpha=1.07637430531796*pi
#Alpha=0.328526392676322*pi
#Alpha=np.arctan(np.sqrt(2))
#Alpha=np.arctan(1/2)
Alpha=(2*pi*uniform(0,360))/360
#How many iretation
times=50
######################################################


# Getting coordinates of the polygon+x-coords plus y-coords.
def coords(n):
    '''
input: n>2 is number of sides of the polygon
output: coordinates, x-coords, y-coords
alpha=the angle of partitioning the circle
    '''
    Cords=[]
    for i in range(n+1):
        x_i=cos((2*i*pi)/n)
        y_i=sin((2*i*pi)/n)
        Cords.append([float(x_i),float(y_i)])
    return np.array(Cords)
print(f"The coordinates are {coords(n)}")
#we add the initial vertex one more time to make the polygon a closed curve
#Extend_cords_of_poly=np.vstack([cords_of_poly(n),[(float(cos(0))),(float(sin(0)))]])



######################################################

#plotting the polygon
#Sketching a red ball in the center of billiard

X=coords(n)[:,0]
Y=coords(n)[:,1]
'''
plt.figure()
ax=plt.axes()
ax.plot(X,Y,color="blue")
ax.scatter(np.array([float(0)]),np.array([float(0)]),color="red",marker="o")
plt.title(f"Polygonal billiard of size {n}")
ax.set_aspect('equal')
plt.show()
'''
plt.figure()
plt.plot(X,Y, color="blue")
plt.scatter([0], [0], color="red", marker="o")
plt.title(f"Polygonal billiard of size {n}")
plt.axis("equal")
plt.show()

#####################################################
#Finding the iniitial side the ball will hit when the moving angle is Alpha
for i in range(n):
    if Alpha>=(2*i*pi)/n and Alpha<(2*(i+1)*pi)/n:
        init_side_angles=[(2*i*pi)/n,(2*(i+1)*pi)/n]
        Ini=coords(n)[i:i+2,:]
init_side_cords=Ini    

print(f"The initial angle, Alpha={Alpha}")
print(f"Initial side angles are {init_side_angles}") 
print(f"Initial side coordinates are {init_side_cords}")


########################################################
# Finding the intersection of the first hit from the center and the side
#<x,y>=<0,0>+t<cos(alpha),sin(alpha)> is the line from origin to the first intersection and <x,y>=<x_1,y_1>+s<x_2-x_1,y_2-y_1> id the first hitted side
#(x_2-x_1)t-cos(alpha)=-x_1
#(y_2-y_1)t-sin(alpa)=-y_1
x_1=init_side_cords[0,0]
y_1=init_side_cords[0,1]
x_2=init_side_cords[1,0]
y_2=init_side_cords[1,1]

A_s=np.array([[-x_1,-float(cos(Alpha))],[-y_1,-float(sin(Alpha))]])
A_t=np.array([[x_2-x_1,-x_1],[y_2-y_1,-y_1]])
A=np.array([[x_2-x_1,-float(cos(Alpha))],[y_2-y_1,-float(sin(Alpha))]])


s_int_fst=np.linalg.det(A_s)/np.linalg.det(A)
t_int_fst=np.linalg.det(A_t)/np.linalg.det(A)

x_int_fst=t_int_fst*float(cos(Alpha))
y_int_fst=t_int_fst*float(sin(Alpha))
int_fst=np.array([x_int_fst,y_int_fst])
x_int_fst_and_orig=[x_int_fst,0]
y_int_fst_and_orig=[y_int_fst,0]




###############################################################
#plotting the polygon and the trajectory of the first hit

plt.figure()
ax=plt.axes()
ax.plot(X,Y,color='blue')
ax.scatter(np.array([0]),np.array([0]),color="red",marker="o")
ax.plot(x_int_fst_and_orig,y_int_fst_and_orig,color="red")
plt.title(f"Polygonal billiard of size {n}")
ax.set_aspect('equal')
plt.show()


###############################################################
#defining two helper fuctions, first projection, second angle of two vectors

def proj_of_v_w(v,w):
    w_norm_sqrd=np.sum(v**2)
    proj_of_v_w=(np.dot(v,w)/w_norm_sqrd)*w
    return proj_of_v_w

def angle(v,w):
    v_u=v/np.sqrt(np.sum(v**2))
    w_u=w/np.sqrt(np.sum(w**2))
    angle=math.acos(np.clip(np.dot(v_u,w_u),-1,1))
    if angle<2e-08:
        angle=0
    return angle
#print(f"the angle between i,j={angle(np.array([1,1]),np.array([1,-1]))}")
########################################################################################


#finding the direction of the second trajectory 
'''
v=vector from the first intersection to the origin
w=the vector of the first hited side

'''
for i in range(n):
    if Alpha==(2*i*pi)/n:
        print("The ball into the pocket ")
        break
    else:
        v=-int_fst
        w=init_side_cords[1]-init_side_cords[0]
        scnd_dir=v-2*proj_of_v_w(v,w)
       
print(f"The direction of the first trajectory which is also the first crossing point is {int_fst}")
print(f"The direction of the second trajectory is {scnd_dir}")

######################################################################
#The following function receives a tip point on one edge, the edge and the direction of tip-leaving trajectory and
#outputs the next the side that includes the tail.
'''
inputs:
initial_point=array,           the initial point on the initial side
initial_side=2d array,      the end points of the side containing the initial point
dir_from_initialp=array,     the direction to leave initial point
outputs:
terminal_side=2d array,      the end points of the side containing the terminal point
'''
def Terminal_side(initial_point,initial_side,dir_from_initialp):
    #finding the angle between initial side and initial direction (\in (0,pi/2))
    initial_side = np.array(initial_side)
    initial_point = np.array(initial_point)
    dir_from_initialp = np.array(dir_from_initialp)
    v=initial_side[1]-initial_side[0] #this is the vector of initial side
    w=dir_from_initialp
    #angle for us is always the large one, meaning > pi/2
    if np.dot(v,w)>=0:
        angle_v_and_w=angle(v,w)
    else:
        v=-v
        angle_v_and_w=angle(v,w)
        
    #determining terminal_side, the side that the initial direction's line will cross
    
    for i in range(n):
        if np.array_equal(coords(n)[i],initial_side[0]) and np.array_equal(coords(n)[i+1],initial_side[1]):
            continue
        if np.array_equal(coords(n)[i],initial_point) and np.array_equal(coords(n)[i+1],initial_point):
            continue

        vector_a=coords(n)[i]-initial_point
        vector_b=coords(n)[i+1]-initial_point
       
        if (angle(vector_a,v)>angle_v_and_w and angle(vector_b,v)>angle_v_and_w) or (angle(vector_a,v)<angle_v_and_w and angle(vector_b,v)<angle_v_and_w):
            continue
        if ((angle(vector_b,v)<angle_v_and_w and angle_v_and_w<angle(vector_a,v)) or (angle(vector_a,v)<angle_v_and_w and angle_v_and_w<angle(vector_b,v))):
            return coords(n)[i:i+2,:]
             
    return None

######################################################################
#The following function receives a tip point on one edge and the direction of tip-leaving trajectory and
#outputs the next the terminal point
'''
inputs:
initial_point=array,           the initial point on the initial side
terminal_side=2d array,      the end points of the side containing the initial point
dir_from_initialp=array,     the direction to leave initial point
outputs:
terminal_point=array,           the terminal point on the terminal side
''' 
def Terminal_point(initial_point,terminal_side,dir_from_initialp): 
    #Now we find the terminal point, the point of crossing the terminal side.
    #<x,y>=initial_point+t*dir_from_initial_point> is the line from initial point with initial direction and <x,y>=<x_1,y_1>+s<x_2-x_1,y_2-y_1> id the terminal side

    x_1=terminal_side[0,0]
    y_1=terminal_side[0,1]
    x_2=terminal_side[1,0]
    y_2=terminal_side[1,1]
    # We use Cramer's rule to find intersection of two parametric lines
    A_s=np.array([[initial_point[0]-x_1,-dir_from_initialp[0]],[initial_point[1]-y_1,-dir_from_initialp[1]]])
    A_t=np.array([[x_2-x_1,initial_point[0]-x_1],[y_2-y_1,initial_point[1]-y_1]])
    A=np.array([[x_2-x_1,-dir_from_initialp[0]],[y_2-y_1,-dir_from_initialp[1]]])


    s_terminalp=np.linalg.det(A_s)/np.linalg.det(A)
    t_terminalp=np.linalg.det(A_t)/np.linalg.det(A)

    x_terminalp=x_1+s_terminalp*(x_2-x_1)
    y_terminalp=y_1+s_terminalp*(y_2-y_1)
    terminal_point=np.array([x_terminalp,y_terminalp])
    #initialp_and_terminalp=np.array([initial_point,terminal_point])
    return terminal_point

  
######################################################################
#the following 
#The following function receives end edge and the direction of tip-leaving trajectory and
#outputs and next directon of the trajectory
'''
inputs:
terminal_side=2d array,      the end points of the side containing the initial point
dir_from_initialp=array,     the direction to leave initial point
outputs:
dir_from_terminalp=array     the direction to leave the terminal point
'''  
def Dir_from_terminalp(terminal_side,dir_from_initialp):
    v=-dir_from_initialp
    w=terminal_side[1]-terminal_side[0]
    dir_from_terminalp=v-2*proj_of_v_w(v,w)   
    return dir_from_terminalp 

######################################################################
#the following 
#The following function receives a tip point on one edge, the edge and the direction of tip-leaving trajectory and
#outputs the next hit point (tail), the side that includes the tail and next directon of the trajectory
'''
inputs:
initial_point=array,           the initial point on the initial side
initial_side=2d array,      the end points of the side containing the initial point
dir_from_initialp=array,     the direction to leave initial point
outputs:
terminal=array,           the terminal point on the terminal side
terminal_side=2d array,      the end points of the side containing the terminal point
dir_from_terminalp=array     the direction to leave the terminal point


'''
def next_point_and_dir(initial_point,initial_side,dir_from_initialp):
    Tside=Terminal_side(initial_point,initial_side,dir_from_initialp)
    Tpoint=Terminal_point(initial_point,Tside,dir_from_initialp)
    D_from_terminalp=Dir_from_terminalp(Tside,dir_from_initialp)
    return Tpoint,Tside,D_from_terminalp

print(f"The second point and side, initial and terminal are {next_point_and_dir(np.array([0,0.8660254]), np.array([[0.5,0.8660254],[-0.5,0.8660254]]), np.array([-1,-1]))}")
print(f"the second poin, side is {next_point_and_dir(int_fst, init_side_cords, scnd_dir)}")
####################################################
#Here using a loop we plot the general case
#Alpha=the angle
#times=number of iteration

initial_point=int_fst
initial_side=init_side_cords
dir_from_initialp=scnd_dir
fig, ax = plt.subplots()
for i in range(times): 
    terminal_point,terminal_side,dir_from_terminalp=next_point_and_dir(initial_point,initial_side,dir_from_initialp)
    if np.all(terminal_side==None):
        bingo_star = plt.Line2D([0.5], [0.5], marker="*", markersize=20, color="gold", markeredgecolor="black")
        ax.add_line(bingo_star)
        print(f"The ball is in a pocket after {i+1}")
        break
    else:
        initialp_and_terminalp=np.array([initial_point,terminal_point])
        X_next_traj=initialp_and_terminalp[:,0]
        Y_next_traj=initialp_and_terminalp[:,1]
        ax.plot(X,Y,color="blue")
        ax.scatter(np.array([0]),np.array([0]),color="red",marker="o")
        ax.plot(x_int_fst_and_orig,y_int_fst_and_orig,color="red")
        ax.plot(X_next_traj,Y_next_traj,color="red")
        print(f"{i+1}th trajectory, initp={initial_point}, termp={terminal_point},initial side={initial_side}, terminal side={terminal_side},dir from terminal={dir_from_terminalp}, dir from initp={dir_from_initialp}")
        initial_point=terminal_point
        initial_side=terminal_side
        dir_from_initialp=dir_from_terminalp
plt.title(f"Polygonal billiard of size {n}")
ax.set_aspect('equal')
plt.show()  
##############################################################
#To animate, we need the coordinates of the end points of all the trajectories
initial_point=int_fst
initial_side=init_side_cords
dir_from_initialp=scnd_dir
coordinates=[(0,0),(int_fst[0],int_fst[1])]

for i in range(times): 
    terminal_point,terminal_side,dir_from_terminalp=next_point_and_dir(initial_point,initial_side,dir_from_initialp)
    coordinates.append((terminal_point[0],terminal_point[1]))
    initial_point=terminal_point
    initial_side=terminal_side
    dir_from_initialp=dir_from_terminalp
coordinates=np.array(coordinates)    
print(f"coordinates={coordinates}")

Set=coordinates
m = len(Set)
#num_frames_step = 20
num_frames = num_frames_step * (m - 1)

fig = plt.figure()
ax = plt.axes()
scatter_plot = ax.scatter([], [], color="green", marker="o",s=100)


def animation_function(frame):
    segment_index = int(frame / num_frames_step)
    segment_progress = (frame % num_frames_step) / num_frames_step
    x_start,y_start=Set[segment_index]
    x_end,y_end=Set[segment_index+1]

    x = x_start + (x_end-x_start) * segment_progress
    y = y_start + (y_end-y_start) * segment_progress
    plt.plot(X,Y, color="blue")
    ax.plot([x_start, x], [y_start, y], "r-", alpha=0.3)
    scatter_plot.set_offsets([[x, y]])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    return scatter_plot,


animation = FuncAnimation(fig, animation_function, frames=num_frames, interval=3, repeat=False)
plt.gca().set_aspect("equal", adjustable="box")
plt.title(f"Polygonal billiard of size {n} with initial angle {Alpha}")
plt.show()