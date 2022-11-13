import numpy as np
from google.protobuf import text_format
import os
from matplotlib import pyplot as plt

from conf_proto import fuzzy_control_conf_pb2

class Model(object):
    def __init__(self,path_root,T):
        self.root = path_root
        self.conf_file = self.root + "/conf/model_conf.pb.txt"
        self.load_conf()
        self.T = T
        return

    def set_use_deadzone(self,use_deadzone):
        self.model_conf.use_deadzone = use_deadzone
        return

    def load_conf(self):
        self.model_conf = fuzzy_control_conf_pb2.ModelConf()
        f = open(self.conf_file,'rb')
        text_format.Parse(f.read(),self.model_conf)
        print("model_conf: ",self.model_conf)
        f.close() 

        self.X = np.array(self.model_conf.init_value)
        self.N = self.model_conf.N
        self.b = self.model_conf.b
        self.mu = self.model_conf.mu
        self.m = self.model_conf.deadzone_conf.m
        self.delta_l = self.model_conf.deadzone_conf.delta_l
        self.delta_r = self.model_conf.deadzone_conf.delta_r

        return
    
    def f(self,X_k,u):
        X = np.zeros(2)
        x = X_k[0]
        v = X_k[1]
        X[0] = v
        X[1] = self.mu*(1-x*x)*v-x + self.b * u
        return X
    
    def deadzone(self,u1):
        if not self.model_conf.use_deadzone:
            return u1

        if u1 <= self.delta_l:
            u2 = self.m * (u1 - self.delta_l)
        elif u1 >= self.delta_r:
            u2 = self.m * (u1 - self.delta_r)
        else:
            u2 = 0
        return u2

    def odeRK4(self,u):
        dt = self.T / self.N
        for i in np.arange(self.N):
            K1 = self.f(self.X,u)
            K2 = self.f(self.X+K1*dt/2,u)
            K3 = self.f(self.X+K2*dt/2,u)
            K4 = self.f(self.X+K3*dt,u)
            self.X += dt/6.0*(K1+2.0*K2+2.0*K3+K4)
        return

    def update(self,u):
        u1 = self.deadzone(u)
        self.odeRK4(u1)
        return self.X,u1
    
    def state(self):
        return self.X
    
    def test(self):
        x_list = []
        v_list = []
        for i in np.arange(1000):
            X = self.update(0)
            x_list.append(X[0])
            v_list.append(X[1])
        
        plt.figure()
        plt.plot(x_list,v_list)
        plt.grid()
        return
    
def main(root):
    model = Model(root,0.01)
    model.test()

    plt.show()


# if __name__ == '__main__':
#     main(".")