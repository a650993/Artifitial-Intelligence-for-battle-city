# Artifitial Intelligence Battle City
Tntroduction
--------

Battle City is a multi-directional shooter video game for
the Family Computer produced and published in 1985 by
Namco.  
It is the first game that allows 2 players to dual at
the same time.  
The player controls a tank and the goal is
to eliminate a fixed number of the enemy tanks and prevent
our castle form destroyed by enemy.  
We want to set an ai
agent to let ai play the game by itself to see whether the
performance is well.But how do we implement a way?  
In this project, we use various route-finding algorithms to help
ai decides forwarding directions more wisely.


requirement of environment to run 
--------
Python 3  
pip install py.game   

way to test different algorithm
--------
Algorithm Options: BFS, DFS, UCS, a_star (default: UCS)  
change the following code in ai.py file:  
path_direction = self.#function name#(player_rect, enemy_rect, 6)  
path_direction = self.#function name#(player_rect, default_pos_rect, 6)    

way to test different heuristic function of A*
--------
Heuristic function Options: manhattan_distance, euclidean_distance, chebyshev_distance (default: manhattan_distance)  
change the following code in ai.py file:  
def heuristic(self, a, b):  
    return self.#function name#(a, b)  


