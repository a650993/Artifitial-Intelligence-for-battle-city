# Artifitial Intelligence Battle City
Introduction
--------
![image](https://github.com/a650993/Artifitial-Intelligence-for-battle-city/blob/main/battle%20city.png)  
Battle City is a multi-directional shooter video game for the Family Computer produced and published in 1985 by Namco.  
It is the first game that allows 2 players to dual at the same time.  
The player controls a tank and the goal is to eliminate a fixed number of the enemy tanks and prevent our castle form destroyed by enemy.  
We want to set an ai agent to let ai play the game by itself to see whether the performance is well.  
But how do we implement a way?  
In this project, we use various route-finding algorithms to help ai decides forwarding directions more wisely.  


requirement of environment to run 
------
Python 3  
`pip install pygame` 

Execution
--------
Run the following command to perform ai playing Battle City:      
`python tanks.py` 


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
    
## Contributions

|         |林秉          |林哲安  |張嘉文  |
| ------------- |:-------------:| :-----:|:-----:|
| Idea and discussion      |  33.3%     | 33.3% | 33.3% |
| Report      | 35%      |    35% |   30%|
| Data Collection   | 40%   | 30% | 30% |
| Coding  | 25%      |    60% |15% |
| Video Recording       | 35%    | 50% | 15% |
| Result discussion and improvement | 25%      | 60% | 15% |

