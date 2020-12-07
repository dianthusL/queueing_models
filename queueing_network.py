import numpy as np
from scipy import linalg


working_time = 8*60 # office working time (every system in the network works in the same hours)
requester_num = np.array([100, 20, 50, 20, 20, 40]) # number of requesters per class

lambdas = requester_num / working_time # entry lambdas for every class

# entry probabilities for every system (i) for all classes (r) [i x r == 9 x 6]
# for now every class at the beginning is allowed to enter only 1st system (entry ticket)
p_0_ir = np.array([[1.,1.,1.,1.,1.,1.],
				   [0.,0.,0.,0.,0.,0.],
				   [0.,0.,0.,0.,0.,0.],
				   [0.,0.,0.,0.,0.,0.],
				   [0.,0.,0.,0.,0.,0.],
				   [0.,0.,0.,0.,0.,0.],
				   [0.,0.,0.,0.,0.,0.],
				   [0.,0.,0.,0.,0.,0.],
				   [0.,0.,0.,0.,0.,0.]])

lambda_0_ir = lambdas * p_0_ir
print(lambda_0_ir)

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

# Transition matrices [i x i == 9x9]
# Class 1:
p_1 = np.array([[0.,.5,.5,0.,0.,0.,0.,0.,0.],
				[0.,0.,1.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,1.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.]])
# Class 2:
p_2 = np.array([[0.,.5,.5,0.,0.,0.,0.,0.,0.],
				[0.,0.,1.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,1.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.]])
# Class 3:
p_3 = np.array([[0.,.5,.5,0.,0.,0.,0.,0.,0.],
				[0.,0.,1.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,1.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.]])
# Class 4:
p_4 = np.array([[0.,0.,0.,0.,0.,0.,1.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.]])
# Class 5:
p_5 = np.array([[0.,0.,0.,0.,0.,0.,0.,1.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.]])
# Class 6:
p_6 = np.array([[0.,0.,0.,0.,0.,0.,0.,0.,1.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.]])

p_r = [p_1, p_2, p_3, p_4, p_5, p_6]
lambda_ir = []

for i, p in enumerate(p_r):
	np.fill_diagonal(p, -1)
	lambda_ir.append(linalg.solve(p.T, -lambda_0_ir[:, i]))

lambda_ir = np.array(lambda_ir).T
print("Throughput of each class in every system:")
print(lambda_ir)


# relative service intensity rho_ir [i x r == 9 x 6]
service_times = np.array([[1/6,.5,2,5,5,3,2,2,2]]) # service times per system [minutes] (do not depend on class)
channels_num = np.array([[np.inf,np.inf,3,3,1,2,1,1,2]]) # m_i

# for M/M/c:	rho_ir = lambda_ir / (m_i * mu_ir)
# for M/M/inf	rho_ir = lambda_ir / mu_ir ??
channels_num[channels_num == np.inf] = 1
mu_ir = 1. / service_times
rho_ir = lambda_ir / (channels_num.T * mu_ir.T)

print("Relative service intensity of each class in every system:")
print(rho_ir)
