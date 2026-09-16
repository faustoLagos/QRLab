from environments.BaseAviary import BaseAviary
from environments.BaseRLAviary import BaseRLAviary
from environments.inspection_sim2real import InspectionSim2Real
from environments.inspection_s2r_random_velocities import InspectionS2RRandomVelocities
from environments.s2r_globalObs import S2R_GlobalObs
from environments.sim2real_test import Sim2RealTest
from environments.monte_carlo_env import MonteCarloEnv
from environments.stage_1_env import Stage1Env
from environments.stage_2_env import Stage2Env
from environments.stage_3_env import Stage3Env
from environments.stage_4_env import Stage4Env
from environments.stage_5_env import Stage5Env
from environments.stage_6_env import Stage6Env
from environments.stage_6_acrobatic import Stage6Acrobatic

environment_map = {
    'InspectionSim2Real': InspectionSim2Real,
    'InspectionS2RRandomVelocities': InspectionS2RRandomVelocities,
    'Sim2RealTest': Sim2RealTest,
    'S2R_GlobalObs': S2R_GlobalObs,
    'MonteCarloEnv': MonteCarloEnv,
    'Stage1Env': Stage1Env,
    'Stage2Env': Stage2Env,
    'Stage3Env': Stage3Env,
    'Stage4Env': Stage4Env,
    'Stage5Env': Stage5Env,
    'Stage6Env': Stage6Env,
    'Stage6Acrobatic': Stage6Acrobatic
}
