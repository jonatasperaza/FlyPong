"""Fly Evolution Engine — modular components.

Etapa 1: Separar a evolução em componentes independentes
  (GENOMA, INDIVIDUO, POPULAÇÃO, SELEÇÃO, REPRODUÇÃO, AVALIAÇÃO, HALL DA FAMA, EXPERIMENTO).

Etapa 2: Três experimentos científicos
  (Genoma apenas, Lamarckismo artificial, Evolução + aprendizado).
"""

from evolution.genome import Genome
from evolution.individual import Individual, FitnessProfile, MatchRecord, InheritanceMode
from evolution.population import Population
from evolution.hall_of_fame import HallOfFame
from evolution.evaluation import Evaluator, WeightedEvaluationResult
from evolution.experiment import ExperimentRunner, ExperimentResult, StatisticalComparison, ExperimentType
from evolution.reproduction import breed, create_random_population

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
]
