"""Fly Evolution Engine — modular components.

Etapa 1: Separar a evolucao em componentes independentes
  (GENOMA, INDIVIDUO, POPULACAO, SELECAO, REPRODUCAO, AVALIACAO, HALL DA FAMA, EXPERIMENTO).

Etapa 2: Tres experimentos cientificos
  (Genoma apenas, Lamarckismo artificial, Evolucao + aprendizado).

Etapa 3: Estilos de jogo e teste de generalizacao
  (7 estilos de adversarios, avaliacao contra generalizacao).
"""

from evolution.genome import Genome
from evolution.individual import Individual, FitnessProfile, MatchRecord, InheritanceMode
from evolution.population import Population
from evolution.hall_of_fame import HallOfFame
from evolution.evaluation import Evaluator, WeightedEvaluationResult
from evolution.experiment import ExperimentRunner, ExperimentResult, StatisticalComparison, ExperimentType
from evolution.reproduction import breed, create_random_population
from evolution.opponent_test import OpponentTest, evaluate_generalization, GeneralizationResult, StyleTestResult

__all__ = [
    "Genome",
    "Individual",
    "FitnessProfile",
    "MatchRecord",
    "InheritanceMode",
    "Population",
    "HallOfFame",
    "Evaluator",
    "WeightedEvaluationResult",
    "ExperimentRunner",
    "ExperimentResult",
    "StatisticalComparison",
    "ExperimentType",
    "breed",
    "create_random_population",
    "OpponentTest",
    "evaluate_generalization",
    "GeneralizationResult",
    "StyleTestResult",
]
