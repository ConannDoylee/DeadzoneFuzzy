from matplotlib import pyplot as plt
import numpy as np

from simulation import Simulation

## deadzone | compensation
test_group = [[True,True,'dead+comp'],
            [True,False,'dead+non-comp']]

simu_dict = {}
for test in test_group:
    use_dead = test[0]
    use_comp = test[1]
    test_name = test[2]
    # simulation
    simu_dict[test_name] = Simulation('.',test_name)
    # config
    simu_dict[test_name].compensator.set_use_comp(use_comp)
    simu_dict[test_name].model.set_use_deadzone(use_dead)
    # simulate
    simu_dict[test_name].simulate()

plt.figure('error')
for key in simu_dict:
    plt.subplot(1,2,1)
    plt.plot(np.array(simu_dict[key].t_list),np.array(simu_dict[key].x_list)[:,0]-np.array(simu_dict[key].xd_list)[:,0],label=key+'x')
    plt.grid()
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(np.array(simu_dict[key].t_list),np.array(simu_dict[key].x_list)[:,1]-np.array(simu_dict[key].xd_list)[:,1],label=key+'v')
    plt.grid()
    plt.legend()

plt.show()