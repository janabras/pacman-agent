# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint




#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        
        actions = game_state.getLegalActions(self.index) # Here we just get legal actions from game_state

        return random.choice(actions) # And here we pick among this legal actions randomly

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def chooseAction(self, game_state):
        
        self.noFoodTime = 0
        
        RemainingFood = len(self.get_food(game_state).as_list())  # We get the remain food as a list
        FoodCarrying = game_state.get_agent_state(self.index).num_carrying  # Here we declare the food our offensive agent is carrying in this game state

        if self.get_previous_observation() is not None:        # With this conditions we say that if offensive agent eat food in the last state, we set the previous food our agent was carrying to 1, else 0.
            preFoodCarrying= self.get_previous_observation().get_agent_state(self.index).num_carrying
        else: 
            preFoodCarrying = 0    

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)] # Here we get the indexs of our opponents agents
        Chase_ghosts = [i for i in enemies if not i.is_pacman and i.get_position() != None and i.scaredTimer is 0] # Here we get the rival defensive ghost index
        if preFoodCarrying == FoodCarrying and len(Chase_ghosts) == 0:  # If the food that our agent was carrying in the last state is equal to the food is carrying now and the rival defensive ghost has no index
            self.noFoodTime += 1                                        # We continue playing
        else:                                                           # If not we finish
            self.noFoodTime = 0
    
    def Check(self, game_state):
        self.powerMode = False
        self.survivalMode = False
        self.survivalPoint = self.start
        State = game_state.get_agent_state(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]

        MinGhostScare = min(enemy.scaredTimer for enemy in enemies)  # Here we look for the minimum scaredTimer between the two enemy agents
        if MinGhostScare < 15:  # If this minimum scaredTimer is less than 15, we return False
            self.powerMode = False
        else:                   # If both enemy agents are scared and the scared time is more than 15 we return True 
            self.powerMode = True
            self.survivalMode = False   
            self.survivalPoint = self.start


    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def chooseAction(self, game_state):
        self.detectDest = []
        self.initFood = self.get_food_you_are_defending(game_state).as_list()  # We get the initial food
        actions = game_state.getLegalActions(self.index) 
        values = [self.evaluate(game_state, i) for i in actions]

        MaxValue = max(values)  # Here we have the max value from the values we calculated previously with the function evaluate
        BestActions = [i for i, z in zip(actions, values) if z == MaxValue] # We select the best actions when the value is max so to have the best ones
        CurrentEnemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]  # Here we have the indexs of the opponents agents
        CurrentInvader = [enemy for enemy in CurrentEnemies if enemy.is_pacman and enemy.get_position() != None] # Here we know the index of the Attacking agent of the rival
        food_list = game_state.get_food_you_are_defending(game_state).as_list()  # This is the food we are defending as a list
        location = game_state.get_agent_state(self.index).get_position() # Location of our defensive agent

        if len(self.initFood) - len(food_list) > 0:  # If the length of the initial food minus the one that we are defending is positive
            AteFood = list(set(self.initFood).difference(set(food_list)))  # Here we get the food that it has been eaten as a list
            self.initFood = food_list
            self.chaseDest = AteFood
        StateAgent = game_state.get_agent_state(self.index)



    def AStar(self, game_state, destin):
        from util import PriorityQueue
        visited = []
        movements = {}
        costs = {}
        first_cost = 0

        FirstAgentState = game_state.get_agent_state(self.index)
        FirstLocation = FirstAgentState.get_position()
        movement[FirstLocation] = []
        costs[FirstLocation] = 0
        visited.append(FirstLocation)

        PriorityQueue = PriorityQueue()
        PriorityQueue.push(game_state, first_cost)

        while not PriorityQueue.isEmpty():    # While the queue is not empty 
            CurrentState = PriorityQueue.pop()  # We pop the currentState
            CurrentLocation = CurrentState.get_agent_state(self.index).get_position()  # The current location of the current state

        if CurrentLocation == destin:   # If the location is the same as the destination we just return the movements that the state has done
            return movements[CurrentLocation]

        actions = CurrentState.getLegalActions(self.index)

        for i in actions:
            SuccesorState = self.get_successor(CurrentState, i)  # We get the successor of the current state
            SuccesorLocation = SuccesorState.get_agent_state(self.index).get_position() # We get the position of the successor
            UpdatedCost = 1 # We put the new cost at 1
            NextCost = costs[CurrentLocation] + UpdatedCost # And we calculate the next one as the cost to get the current location plus the cost it has to get to the successor

        if SuccesorLocation not in visited or NextCost < costs[SuccesorLocation]:  # If the successor is not in visited nodes or the NextCost is less than the cost of the Successor location
            
            visited.append(SuccesorLocation)  # We append the successor state to visited nodes
            movements[SuccesorLocation] = []  
            costs[SuccesorLocation] = NextCost  # We update the cost of the successor to next cost
            heuristic = self.Heuristic(SuccesorLocation, destin)  # Then, we apply the heuristic function we have created after
            priority = NextCost + heuristic  # Then, we update the priority by using the heuristic 
            PriorityQueue.push(SuccesorState, priority) # And, we push the Successor and this priority to the queue

    def Heuristic(self, location, destin):  # The heuristic we have applied is just optimising the distance to the final destination by using the manhattanDistance from util
        from util import manhattanDistance
        distance = manhattanDistance(location, destin)
        return distance


    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
