from distutils.command.config import dump_file
from re import U
from statistics import mode
import numpy as np
from google.protobuf import text_format

from compensator import Compensator
from controller import Controller
from model import Model

from matplotlib import pyplot as plt

from conf_proto import fuzzy_control_conf_pb2

def ref_state_generator(model,t):
    x0 = np.sin(t)
    x1 = np.cos(t)
    x2 = -np.sin(t)
    
    Xd = np.array([x0,x1])
    xn_d = x2
    X = model.state()
    return X,Xd,xn_d

## create objects
root = '.'
simulation_conf = fuzzy_control_conf_pb2.SimulationConf()
f = open('./conf/simulation_conf.pb.txt','rb')
text_format.Parse(f.read(),simulation_conf)
f.close()
T = simulation_conf.T
compensator =  Compensator(root,T)
controller = Controller(root)
model = Model(root,T)
## simu run
# ref: xd=sin(t)
cycle = 10000

xd_list = []
t_list = []
x_list = []
u1_list = []
ud_list = []
uc_list = []
u2_list = []
uh_list = []
du_list = []

for i in np.arange(cycle):
    t = i * T
    # ref generate
    X,Xd,xn_d = ref_state_generator(model,t)
    # control update
    u1,u_hat,epslon = controller.compute_cmd(X,Xd,xn_d)
    # compensator
    adaptive_value,phi = compensator.adaptive_update(u_hat,epslon)
    u_c = compensator.compensation_out()
    u2 = u1 + u_c
    # model update
    X1,u_d = model.update(u2)
    d_u = u2 - u_d

    # log
    t_list.append(t)
    x_list.append(list(X1))
    u1_list.append(u1)
    uc_list.append(u_c)
    u2_list.append(u2)
    uh_list.append(u_hat)
    ud_list.append(u_d)
    xd_list.append(list(Xd))
    du_list.append(d_u)

# plot
plt.figure()
plt.subplot(2,2,1)
plt.plot(np.array(x_list)[:,0],np.array(x_list)[:,1],label='x')
plt.plot(np.array(xd_list)[:,0],np.array(xd_list)[:,1],label='xd')
plt.grid()
plt.legend()

plt.subplot(2,2,2)
plt.plot(t_list,u1_list,label='u1')
plt.plot(t_list,uc_list,label='uc')
plt.plot(t_list,u2_list,label='u2')
plt.plot(t_list,uh_list,label='uh')
plt.plot(t_list,ud_list,label='ud')

plt.grid()
plt.legend()

plt.subplot(2,2,3)
plt.plot(np.array(t_list),np.array(x_list)[:,0],label='x')
plt.plot(np.array(t_list),np.array(xd_list)[:,0],label='xd')
plt.plot(np.array(t_list),np.array(x_list)[:,0]-np.array(xd_list)[:,0],label='error')

plt.grid()
plt.legend()

plt.subplot(2,2,4)
plt.plot(np.array(t_list),np.array(x_list)[:,1],label='x_dot')
plt.plot(np.array(t_list),np.array(xd_list)[:,1],label='xd_dot')
plt.plot(np.array(t_list),np.array(x_list)[:,1]-np.array(xd_list)[:,1],label='error')

plt.grid()
plt.legend()


plt.figure()
plt.plot(t_list,du_list,label='du')
plt.plot(t_list,uc_list,label='du_hat')
plt.grid()

plt.show()
