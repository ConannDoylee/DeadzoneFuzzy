import numpy as np
import skfuzzy as fuzz
import os
from google.protobuf import text_format
from matplotlib import pyplot as plt

from conf_proto import fuzzy_control_conf_pb2

class Compensator(object):

    def __init__(self,path_root,T):
        self.root = path_root
        self.conf_file = self.root + "/conf/compensator_conf.pb.txt"
        self.load_conf()
        self.set_up()
        self.load_adaptive_init()
        self.T = T
        return
    
    def set_use_comp(self,use_comp):
        self.compensation_conf.use_compensation_out = use_comp
        return

    def load_adaptive_init(self):
        self.adaptive_value = np.array(self.compensation_conf.init_adaptive_values)
        self.compensation_value = self.compensation_conf.init_compensation_value
        return

    def load_conf(self):
        self.compensation_conf = fuzzy_control_conf_pb2.CompensationConf()
        f = open(self.conf_file,'rb')
        text_format.Parse(f.read(),self.compensation_conf)
        print("compensation_conf: ",self.compensation_conf)
        f.close()
        return
    
    def build_mems(self,type,array,range_array):
        if type == "trimf":
            return fuzz.trimf(range_array,array)
        elif type == "trapmf": 
            return fuzz.trapmf(range_array,array)
        else:
            print("type incorrect: ",type)
            os._exit(0)
                                            
    def set_up(self):
        # Membership functions
        self.mems_dict = {}
        range_start = self.compensation_conf.fuzzy_build_conf.ant_1.range_array[0]
        range_end = self.compensation_conf.fuzzy_build_conf.ant_1.range_array[1]
        range_delta = self.compensation_conf.fuzzy_build_conf.ant_1.range_array[2]
        range_array = np.arange(range_start,range_end,range_delta)
        self.mems_dict['range'] = range_array
        for mf in self.compensation_conf.fuzzy_build_conf.ant_1.mfs:
            self.mems_dict[mf.name] = self.build_mems(mf.type,mf.array,range_array)
        return

    def compute_rule_weight(self,rule,u):
        weight = 0

        certainties = []
        for mf_name in rule.mf_names:
            certainty = fuzz.interp_membership(self.mems_dict['range'],self.mems_dict[mf_name],u)
            certainties.append(certainty)
        
        if not certainties:
            print("rule has some problems.")
            os._exit(0)

        certainties = np.array(certainties)

        if rule.type == "AND":
            weight = np.min(certainties)
        elif rule.type == "or":
            weight = np.max(certainties)
        else:
            print("rule type cannot be found")
            os._exit(0)

        return weight

    def adaptive_update(self,u_hat,epslon):
        rule_weights = []
        for i,rule in zip(np.arange(len(self.compensation_conf.fuzzy_build_conf.rules)),\
                                self.compensation_conf.fuzzy_build_conf.rules):
            rule_weights.append(self.compute_rule_weight(rule,u_hat))
        # PHI
        # print("u_hat,rule_weights: ",u_hat,rule_weights)
        self.PHI = np.array([wi/sum(rule_weights) for wi in rule_weights])
        # D_dot_hat
        phi = self.compensation_conf.phi
        D_dot_hat = -phi*epslon*self.PHI
        # D_hat
        self.adaptive_value += D_dot_hat * self.T
        return self.adaptive_value,self.PHI

    def compensation_out(self):
        if not self.compensation_conf.use_compensation_out:
            return 0
        return self.adaptive_value.transpose().dot(self.PHI)

    def plot_mfs(self):
        plt.figure()
        for key in self.mems_dict:
            if key != 'range':
                plt.plot(self.mems_dict['range'],self.mems_dict[key])
        plt.grid()
        return

def main(root):
    compensator = Compensator(root,0.01)
    compensator.plot_mfs()
    plt.show()


# if __name__ == '__main__':
#     main(".")


