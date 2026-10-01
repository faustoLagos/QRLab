from environments.BaseAviary import BaseAviary
from environments.BaseRLAviary import BaseRLAviary
from environments.icra27env import ICRA27Env
from environments.test_env import TestEnv

environment_map = {
    'ICRA27Env': ICRA27Env,
    'TestEnv': TestEnv
}
