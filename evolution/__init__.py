"""Fly Evolution Engine — modular components.

Etapa 1 do roadmap Super Copa das Moscas: separar a evolução em
componentes independentes (GENOMA, INDIVÍDUO, POPULAÇÃO, SELEÇÃO,
REPRODUÇÃO, AVALIAÇÃO, HALL DA FAMA, EXPERIMENTO).
"""

from evolution.genome import Genome
from evolution.individual import Individual
from evolution.population import Population
from evolution.hall_of_fame import HallOfFame
from evolution.evaluation import Evaluator
from evolution.experiment import Experiment

__all__ = ["Genome", "Individual", "Population", "HallOfFame", "Evaluator", "Experiment"]
