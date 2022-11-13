import numpy as np
from google.protobuf import text_format

from compensator import Compensator
from controller import Controller
from model import Model

from conf_proto import fuzzy_control_conf_pb2
from matplotlib import pyplot as plt

class Simulation(object):
    def __init__(self,root,test_name):
        self.test_name = test_name
        simulation_conf = fuzzy_control_conf_pb2.SimulationConf()
        f = open('./conf/simulation_conf.pb.txt','rb')
        text_format.Parse(f.read(),simulation_conf)
        f.close()
        self.T = simulation_conf.T
        self.cycle = simulation_conf.cycle
        self.compensator =  Compensator(root,self.T)
        self.controller = Controller(root)
        self.model = Model(root,self.T)
        self.init_list()
        return

    def init_list(self):
        self.xd_list = []
        self.t_list = []
        self.x_list = []
        self.u1_list = []
        self.ud_list = []
        self.uc_list = []
        self.u2_list = []
        self.uh_list = []
        self.du_list = []
        return

    def ref_state_generator(self,t):
        x0 = np.sin(t)
        x1 = np.cos(t)
        x2 = -np.sin(t)
        
        Xd = np.array([x0,x1])
        xn_d = x2
        X = self.model.state()
        return X,Xd,xn_d

    def simulate(self):
        for i in np.arange(self.cycle):
            t = i * self.T
            # ref generate
            X,Xd,xn_d = self.ref_state_generator(t)
            # control update
            u1,u_hat,epslon = self.controller.compute_cmd(X,Xd,xn_d)
            # compensator
            adaptive_value,phi = self.compensator.adaptive_update(u_hat,epslon)
            u_c = self.compensator.compensation_out()
            u2 = u1 + u_c
            # model update
            X1,u_d = self.model.update(u2)
            d_u = u2 - u_d

            # log
            self.t_list.append(t)
            self.x_list.append(list(X1))
            self.u1_list.append(u1)
            self.uc_list.append(u_c)
            self.u2_list.append(u2)
            self.uh_list.append(u_hat)
            self.ud_list.append(u_d)
            self.xd_list.append(list(Xd))
            self.du_list.append(d_u)

        return

    def data_plot(self):
        plt.figure(self.test_name)
        plt.subplot(2,3,1)
        plt.plot(np.array(self.x_list)[:,0],np.array(self.x_list)[:,1],label='x')
        plt.plot(np.array(self.xd_list)[:,0],np.array(self.xd_list)[:,1],label='xd')
        plt.grid()
        plt.legend()

        plt.subplot(2,3,2)
        plt.plot(self.t_list,self.u1_list,label='u1')
        plt.plot(self.t_list,self.uc_list,label='uc')
        plt.plot(self.t_list,self.u2_list,label='u2')
        plt.plot(self.t_list,self.uh_list,label='uh')
        plt.plot(self.t_list,self.ud_list,label='ud')
        plt.grid()
        plt.legend()

        plt.subplot(2,3,3)
        plt.plot(np.array(self.t_list),np.array(self.x_list)[:,0],label='x')
        plt.plot(np.array(self.t_list),np.array(self.xd_list)[:,0],label='xd')
        plt.plot(np.array(self.t_list),np.array(self.x_list)[:,0]-np.array(self.xd_list)[:,0],label='error')
        plt.grid()
        plt.legend()

        plt.subplot(2,3,4)
        plt.plot(np.array(self.t_list),np.array(self.x_list)[:,1],label='x_dot')
        plt.plot(np.array(self.t_list),np.array(self.xd_list)[:,1],label='xd_dot')
        plt.plot(np.array(self.t_list),np.array(self.x_list)[:,1]-np.array(self.xd_list)[:,1],label='error')
        plt.grid()
        plt.legend()

        plt.subplot(2,3,5)
        plt.plot(self.t_list,self.du_list,label='du')
        plt.plot(self.t_list,self.uc_list,label='du_hat')
        plt.grid()
        plt.legend()

        plt.show()
        return