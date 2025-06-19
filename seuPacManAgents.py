# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState
from multiAgents import MultiAgentSearchAgent


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        """
        Implementação do algoritmo Minimax para o Pacman.
        Retorna a melhor ação calculada pelo algoritmo.
        """
        
        def minimax(agentIndex=0, depth=0, state=gameState):
            # Verifica se o jogo acabou ou atingiu a profundidade máxima
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            # Calcula o próximo agente e se é o turno do Pacman (MAX)
            nextAgent = agentIndex + 1
            if nextAgent >= state.getNumAgents():
                nextAgent = 0  # Volta para o Pacman
                depth += 1    # Aumenta a profundidade apenas após todos os agentes
            
            isPacman = (agentIndex == 0)  # Pacman é o agente 0 (MAX)
            
            # Inicializa valores extremos dependendo se é MAX ou MIN
            bestValue = -float('inf') if isPacman else float('inf')
            bestAction = None
            
            # Para cada ação possível do agente atual
            for action in state.getLegalActions(agentIndex):
                # Gera o próximo estado
                nextState = state.generateSuccessor(agentIndex, action)
                
                # Chama minimax recursivamente para o próximo agente
                value = minimax(nextAgent, depth, nextState)
                
                # Atualiza o melhor valor e ação
                if isPacman:
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                else:
                    if value < bestValue:
                        bestValue = value
            
            # Se for o Pacman (chamada inicial), retorna a melhor ação
            if agentIndex == 0 and depth == 0:
                return bestAction
            # Caso contrário, retorna apenas o melhor valor
            return bestValue
        
        # Inicia o algoritmo a partir do Pacman (agente 0) na profundidade 0
        return minimax()


def betterEvaluationFunction(currentGameState: GameState):
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    # Calcula a distância de Manhattan para a comida mais próxima
    foodDistances = [manhattanDistance(pos, f) for f in food]
    if len(foodDistances) > 0:
        minFoodDistance = min(foodDistances)
    else:
        minFoodDistance = 0

    # Distância para o fantasma mais próximo
    ghostDistances = [manhattanDistance(pos, ghost.getPosition()) for ghost in ghostStates]
    minGhostDistance = min(ghostDistances)

    # Aumenta a pontuação se o fantasma estiver assustado, mas penaliza se estiver muito perto
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    if min(scaredTimes) > 0:
        minGhostDistance = 0  # Ignora fantasmas assustados

    return currentGameState.getScore() - (1.5 / (minFoodDistance + 1)) + (2 / (minGhostDistance + 1))

# Abbreviation
better = betterEvaluationFunction
