from gridworld import GridWorld

gw = GridWorld(3, 4, [(1,1)], [(1,3)], [(0,3)], -0.1)
gw.r_draw_q_values()
gw.solve_bellman(gamma=0.9)
# gw.wait_esc_key()

