import numpy as np


working_time = 8*60	# office working time [min] (every system in the network works in the same hours)
requester_num = np.array([100, 20, 50, 20, 40]) # number of requesters per class

# entry probabilities for every system (i) for all classes (r) [i x r == 9 x 5]
# for now every class at the beginning is only allowed to enter 1st system (entry ticket)
p_0_ir = np.array([[1., 1., 1., 1., 1.],
				   [0., 0., 0., 0., 0.],
				   [0., 0., 0., 0., 0.],
				   [0., 0., 0., 0., 0.],
				   [0., 0., 0., 0., 0.],
				   [0., 0., 0., 0., 0.],
				   [0., 0., 0., 0., 0.],
				   [0., 0., 0., 0., 0.],
				   [0., 0., 0., 0., 0.]])

# transition matrices [i x i == 9x9]
# Class 1:
p_1 = np.array([[.0, .4, .3, .0, .0, .0, .0, .0, .3],
				[.0, .0, 1., .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, 1., .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0]])
# Class 2:
p_2 = np.array([[.0, .4, .3, .0, .0, .0, .0, .0, .3],
				[.0, .0, 1., .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, 1., .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0]])
# Class 3:
p_3 = np.array([[.0, .4, .3, .0, .0, .0, .0, .0, .3],
				[.0, .0, 1., .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, 1., .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0]])
# Class 4:
p_4 = np.array([[.0, .0, .0, .0, .0, .0, 1., .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0]])
# Class 5:
p_5 = np.array([[.0, .0, .0, .0, .0, .0, .0, 1., .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0],
				[.0, .0, .0, .0, .0, .0, .0, .0, .0]])

p_r = [p_1, p_2, p_3, p_4, p_5]

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
system_types = np.array([3, 3, 1, 1, 1, 1, 1, 1, 1])
service_times = np.array([1/6, .5, 2, 5, 5, 3, 2, 2, 2]).reshape(-1,1) # service times per system [min] (do not depend on class)
channels_num = np.array([np.inf, np.inf, 3, 3, 1, 2, 1, 1, 2]).reshape(-1,1) # m_i

network_states = [(0, 0, 0, 0, 0, 0, 0, 0, 0),
				  (5, 5, 3, 2, 0, 1, 5, 6, 2),
				  (3, 3, 2, 2, 1, 1, 0, 0 ,2),
				  (5, 4, 3, 2, 1, 0, 1, 2, 3)]
