from time import sleep
from gridworld import GridWorld

gw = GridWorld( rows=3, 
                cols=5, 
                pits=[(2,0), (2,1), (2,2), (2,3), (2,4)], 
                goals=[(1,4)], 
                live_reward=-0.01
            )

# Fixed exploration: 50%
# gw.q_learning(gamma=0.9, epsilon_0=0.5)

# Decreasing exploration (up to 100 )
# gw.q_learning(gamma=0.9, episodes=100)

# Decreasing exploration with utilities plot
gw.q_learning(gamma=0.9, episodes=100, plot=True)