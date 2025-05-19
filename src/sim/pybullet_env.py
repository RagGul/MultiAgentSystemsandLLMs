import time, atexit
import pybullet as p
import pybullet_data

class PyBulletEnv:
    """Spins up a Bullet world with a plane and lets you spawn cylinders."""
    def __init__(self, gui: bool = True):
        self.physics = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("plane.urdf")
        self.robots = {}                      # agent_id → bodyUniqueId
        atexit.register(p.disconnect)

    def spawn_robot(self, agent_id: str, pos, radius=0.1, height=0.05):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                                  rgbaColor=[0.2, 0.5, 0.9, 1.0])
        body = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis, basePosition=pos)
        self.robots[agent_id] = body

    def set_robot_colour(self, agent_id: str, rgba):
        bid = self.robots.get(agent_id)
        if bid is not None:
            p.changeVisualShape(bid, -1, rgbaColor=rgba)

    def step(self, dt=1/60):
        p.stepSimulation()
        time.sleep(dt)
