import numpy as np

# NETWORK CONFIGURATION
# office working time [min]
# every system in the network works in the same hours
working_time = 8 * 60

# number of requesters per class
requester_num = [250, 60, 150, 60, 100]

# entry probabilities for every system (i) for all classes (r) [i x r == 9 x 5]
# for now every class at the beginning is only allowed to enter 1st system (entry ticket)
p_0_ir = [[1., 1., 1., 1., 1.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.]]

# transition matrices [i x i == 9x9]
# Class 1:
p_1 = [[.0, .4, .3, .0, .0, .0, .0, .0, .3],
       [.0, .0, 1., .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, 1., .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0]]
# Class 2:
p_2 = [[.0, .4, .3, .0, .0, .0, .0, .0, .3],
       [.0, .0, 1., .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, 1., .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0]]
# Class 3:
p_3 = [[.0, .4, .3, .0, .0, .0, .0, .0, .3],
       [.0, .0, 1., .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, 1., .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0]]
# Class 4:
p_4 = [[.0, .0, .0, .0, .0, .0, 1., .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0]]
# Class 5:
p_5 = [[.0, .0, .0, .0, .0, .0, .0, 1., .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0],
       [.0, .0, .0, .0, .0, .0, .0, .0, .0]]

""" SYSTEMS:
1. Entry ticket		M/M/inf
2. Application		M/M/inf
3. Ticket office	M/M/3
4. O1				M/M/3
5. O2				M/M/1
6. O3				M/M/2
7. O4				M/M/1
8. O5				M/M/1
9. O6				M/M/2
"""
system_types = [3, 3, 1, 1, 1, 1, 1, 1, 1]

# service times per system [min]
# do not depend on class
service_times = [1 / 4, 1 / 2, 2, 5, 5, 3, 2, 2, 3]

# number of channels for every system [m_i]
channels_num = [np.inf, np.inf, 3, 3, 1, 2, 1, 1, 2]

network_states = [(0, 0, 0, 0, 0, 0, 0, 0, 0),
                  (5, 5, 3, 2, 0, 1, 5, 6, 2),
                  (3, 3, 2, 2, 1, 1, 0, 0, 2),
                  (5, 4, 3, 2, 1, 0, 1, 2, 3)]

# waiting costs of classes entries in each system
# only systems with finite number of channels matter (3-9)
C_ir = [[6, 5, 5, 0, 0],
        [4, 0, 0, 0, 0],
        [0, 3, 0, 0, 0],
        [0, 0, 4, 0, 0],
        [0, 0, 0, 5, 0],
        [0, 0, 0, 0, 4],
        [6, 5, 5, 0, 0]]

# costs of unoccupied channels in every system
# related to existence of channel not being used
C_i = [5, 5, 3, 3, 2, 2, 5]

# CLONALG CONFIGURATION
# minimum and maximum number of available channels for every system
min_num = 1
max_num = 10
population_size = 300
selection_size = 30
clone_rate = 10
mutation_rate = 8
iterations = 100

"""
NON-EDITABLE PART BELOW
this is just the preparation of whole configuration for further computations
"""
requester_num = np.array(requester_num)
p_0_ir = np.array(p_0_ir)
p_1 = np.array(p_1)
p_2 = np.array(p_2)
p_3 = np.array(p_3)
p_4 = np.array(p_4)
p_5 = np.array(p_5)
p_r = [p_1, p_2, p_3, p_4, p_5]
system_types = np.array(system_types)
service_times = np.array(service_times).reshape(-1, 1)
channels_num = np.array(channels_num).reshape(-1, 1)
C_ir = np.array(C_ir)
C_i = np.array(C_i)
