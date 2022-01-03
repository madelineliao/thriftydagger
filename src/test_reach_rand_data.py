import matplotlib.pyplot as plt
import pickle

data_path = './data/Reach2D/oracle_random_start.pkl'

data = pickle.load(open(data_path, 'rb'))

obs_x = []
obs_y = []

for demo in data[:100]:
    obs = demo['obs']
    obs_x += [o[0] for o in obs]
    obs_y += [o[1] for o in obs]
    
plt.scatter(obs_x, obs_y)
plt.show()