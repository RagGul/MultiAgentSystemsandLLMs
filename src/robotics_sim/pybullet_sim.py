import pybullet as p, pybullet_data, time
from src.config import Config

class PyBulletSim:
    def __init__(self):
        mode = p.GUI if Config.GUI else p.DIRECT
        p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        self.plane = p.loadURDF("plane.urdf")
        # spawn dummy boxes as placeholders for robots
        self.bodies = [p.loadURDF("r2d2.urdf", [i*1.5,0,0.1]) for i in range(5)]

    def step(self):
        p.stepSimulation()
        if Config.GUI:
            time.sleep(Config.SIM_DT)

    def shutdown(self):
        p.disconnect()
