import random
import math

BOT_NAME = "MagicalBot"  

class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""
    def __init__(self, sd=None):
        if sd is None:
            self.st = None
        else:
            random.seed(sd)
            self.st = random.getstate()

    def get_move(self, state):
        if self.st is not None:
            random.setstate(self.st)
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move."""
    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""

    def get_move(self, state):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player()
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            util = self.minimax(state)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state

    def minimax(self, state):
        """Determine the minimax utility value of the given state.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the exact minimax utility value of the state
        """
 
        if state.is_full():
            return state.utility()
        if state.next_player() == 1:
            utility = self.max_value(state)
        else:
            utility = self.min_value(state)
        return utility

    def min_value(self, state):
        if state.is_full():
            return state.utility()
        v = math.inf
        succs = state.successors()
        for move in succs:
            v = min(v, self.max_value(move[1]))
        return v
    
    def max_value(self, state):
        if state.is_full():
            return state.utility()      
        v = -math.inf
        succs = state.successors()
        for move in succs:
            v = max(v, self.min_value(move[1]))
        return v


class MinimaxHeuristicAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def min_value(self, state, depth):
        if state.is_full() or depth == self.depth_limit:
            return self.evaluation(state)
        v = math.inf
        succs = state.successors()
        for move in succs:
            v = min(v, self.max_value(move[1], depth + 1))
        return v
    
    def max_value(self, state, depth):
        if state.is_full() or depth == self.depth_limit:
            return self.evaluation(state)      
        v = -math.inf
        succs = state.successors()
        for move in succs:
            v = max(v, self.min_value(move[1],depth + 1))
        return v

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state.

        The depth data member (set in the constructor) determines the maximum depth of the game 
        tree that gets explored before estimating the state utilities using the evaluation() 
        function.  If depth is 0, no traversal is performed, and minimax returns the results of 
        a call to evaluation().  If depth is None, the entire game tree is traversed.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        if self.depth_limit == 0:
            return self.evaluation(state)

        if self.depth_limit == None:
            return MinimaxAgent.minimax(state)

        if state.is_full():
            return state.utility()

        if state.next_player() == 1:
            utility = self.max_value(state, 1)
        else:
            utility = self.min_value(state, 1)
        return utility


    def streaks_helper(self, lst):  
        """Get the lengths of all the streaks of the same element in a sequence."""
        rets = []  # list of (element, length) tuples
        prev = lst[0]
        curr_len = 1
        for curr in lst[1:]:
            if curr == prev:
                curr_len += 1
            else:
                rets.append((prev, curr_len))
                prev = curr
                curr_len = 1
        rets.append((prev, curr_len))
        return rets
    
    def helper(self, streaks, state):
        sump1 = 0
        sump2 = 0
        for i in range(0, len(streaks)):
            if i < len(streaks)-1:
                if streaks[i][1] >= 2 and streaks[i+1][0] == 0:
                    if streaks[i][0] == 1 and state.next_player() == 1:
                        sump1 += 1
                    elif streaks[i][0] == -1 and state.next_player() == -1:
                        sump2 += 1
            if i > 0:
                if streaks[i][1] >= 2 and streaks[i-1][0] == 0:
                    if streaks[i][0] == 1 and state.next_player() == 1:
                        sump1 += 1
                    elif streaks[i][0] == -1 and state.next_player() == -1:
                        sump2 += 1
            if i != 0 and i != len(streaks)-1:
                if streaks[i-1][0] == streaks[i+1][0] and streaks[i][0] == 0:
                    if streaks[i-1][0] == 1 and state.next_player() == 1:
                        sump1 += 1
                    elif streaks[i-1][0] == -1 and state.next_player() == -1:
                        sump2 += 1
        return sump1, sump2



    def evaluation(self, state):

        """Estimate the utility value of the game state based on features.

        N.B.: This method must run in constant time for all states!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heuristic estimate of the utility value of the state
        """
        sump1 = 0
        sump2 = 0
        cols = state.get_cols()
        for i in range(0, len(cols)):
            streaks = self.streaks_helper(cols[i])
            for i in range(0,len(streaks)-1):
                if (streaks[i][1] >= 2) and (streaks[i][0] == 1) and (streaks[i+1][0] == 0) and (state.next_player() == 1):
                    sump1 += 1
                elif (streaks[i][1] >= 2) and (streaks[i][0] == -1) and (streaks[i+1][0] == 0) and (state.next_player() == -1):
                    sump2 += 1
        
        rows = state.get_rows()
        for i in range(0, len(rows)):
            streaks = self.streaks_helper(rows[i])
            sump1 += self.helper(streaks, state)[0]
            sump2 += self.helper(streaks, state)[1]
                           
        diags = state.get_diags()
        diags = [x for x in diags if len(x)>=3]
        for i in range(0, len(diags)):
            streaks = self.streaks_helper(diags[i])
            sump1 += self.helper(streaks, state)[0]
            sump2 += self.helper(streaks, state)[1]
        return sump1 - sump2


class MinimaxPruneAgent(MinimaxAgent):
    """Smarter computer agent that uses minimax with alpha-beta pruning to select the best move."""

    def minimax(self, state):
        """Determine the minimax utility value the given state using alpha-beta pruning.

        The value should be equal to the one determined by MinimaxAgent.minimax(), but the 
        algorithm should do less work.  You can check this by inspecting the value of the class 
        variable GameState.state_count, which keeps track of how many GameState objects have been 
        created over time.  This agent does not use a depth limit like MinimaxHeuristicAgent.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors().  That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to to column 1.

        Args: 
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """

        if state.is_full():
            return state.utility()
        if state.next_player() == 1:
            return self.max_value(state,-math.inf,math.inf)
        else:
            return self.min_value(state,-math.inf,math.inf)
        
    def min_value(self, state,alpha,beta):
        if state.is_full():
            return state.utility()      
        v = math.inf
        succs = state.successors()
        for move in succs:
            val = self.max_value(move[1],alpha,beta)
            v = min(val,v)
            beta = min(v,beta)
            if beta <= alpha:
                break
        return v
    
    def max_value(self, state,alpha,beta):
        if state.is_full():
            return state.utility()      
        v = -math.inf
        succs = state.successors()
        for move in succs:
            val = self.min_value(move[1],alpha,beta)
            v = max(val,v)
            alpha = max(v,alpha)
            if beta <= alpha:
                break
        return v


# N.B.: The following class is provided for convenience only; you do not need to implement it!

class OtherMinimaxHeuristicAgent(MinimaxAgent):
    """Alternative heursitic agent used for testing."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state."""

        return 26  # Change this line, unless you have something better to do.

