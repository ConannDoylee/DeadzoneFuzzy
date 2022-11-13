import numpy as np
import math
import os
from google.protobuf import text_format
from scipy.special import comb

from conf_proto import fuzzy_control_conf_pb2

class Controller(object):

    def __init__(self,path_root):
        self.root = path_root
        self.conf_file = self.root + "/conf/controller_conf.pb.txt"
        self.load_conf()
        self.build_hurwitz_poly()
        return
    
    def load_conf(self):
        self.controller_conf = fuzzy_control_conf_pb2.ControllerConf()
        f = open(self.conf_file,'rb')
        text_format.Parse(f.read(),self.controller_conf)
        print("controller_conf: ",self.controller_conf)
        f.close() 
        return
    
    def build_hurwitz_poly(self):
        lambbda = self.controller_conf.lambbda
        n = self.controller_conf.n
        if n < 1:
            print("ERROR: n < 1")
            os._exit(0)
        lambbda_list = [1]
        for i in np.arange(n-1):
            lambbda_list.append(lambbda*lambbda_list[-1])
        lambbda_list.reverse()

        comb_list = []
        for i in np.arange(n):
            c = comb(n-1, i)
            comb_list.append(c)
        comb_list.reverse()

        # c
        lambbda_array = np.array(lambbda_list)
        comb_array = np.array(comb_list)
        self.c = lambbda_array * comb_array
        # c_bar
        self.c_bar = lambbda_array * comb_array
        self.c_bar[-1] = 0
        
        print("lambbda_array: ",lambbda_array)
        print("comb_array: ",comb_array)
        print("self.c: ",self.c)
        print("self.c_bar: ",self.c_bar)
    
    def compute_cmd(self,X,Xd,xn_d):
        u1 = 0
        u_hat = 0
        
        X_van = X - Xd
        epsilon = self.c.transpose().dot(X_van)
        b = self.controller_conf.b
        m = self.controller_conf.m
        # f
        f = self.controller_conf.mu * (1-X[0]*X[0]) * X[1] - X[0]
        # f2 = c_bar'.dot(X_van)
        f2 = self.c_bar.transpose().dot(X_van)
        # f3 = kesi*epsilon = c'.dot(X_van)
        f3 = self.controller_conf.kesi * epsilon
        # u1=(bm)^-1 *(-f + xn_d - xd_nc_bar'.dot(X_van) - kesi*epsilon)        
        u1 = 1.0/(m*b) * (-f + xn_d - f2 - f3)
        # u_hat=(bm)^-1 *(-f + xn_d - xd_nc_bar'.dot(X_van))
        u_hat = 1.0/(m*b) * (-f + xn_d - f2)

        return u1,u_hat,epsilon

    def test(self):
        X = np.array([0,0])
        Xd = np.array([1,0])
        xn_d = 0
        u1,u_hat = self.compute_cmd(X,Xd,xn_d)
        print("u1,u_hat: ",u1,u_hat)

        return
def main(root):
    controller = Controller(root)
    controller.test()

# if __name__ == '__main__':
#     main(".")
