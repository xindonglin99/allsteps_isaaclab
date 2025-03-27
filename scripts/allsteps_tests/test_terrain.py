import numpy as np

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi

class Allsteps():

  num_steps = 20
  step_radius = 0.25
  rendered_step_count = 3
  init_step_separation = 0.75

  def __init__(self):
    self.dist_range = np.array([0.65, 1.25])
    self.pitch_range = np.array([-30, +30])  # degrees
    self.yaw_range = np.array([-20, 20])
    self.tilt_range = np.array([-15, 15])
    self.step_param_dim = 5

    self.curriculum = 0
    self.max_curriculum = 9

    self.terrain = self.generate_foot_steps()

  def generate_foot_steps(self):
    
    self.curriculum = min(self.curriculum, self.max_curriculum)
    ratio = self.curriculum / self.max_curriculum

    # {self.max_curriculum + 1} levels in total
    dist_upper = np.linspace(*self.dist_range, self.max_curriculum + 1)
    dist_range = np.array([self.dist_range[0], dist_upper[self.curriculum]])
    yaw_range = self.yaw_range * 0.7 * DEG2RAD
    pitch_range = self.pitch_range * ratio * DEG2RAD + np.pi / 2
    tilt_range = self.tilt_range * ratio * DEG2RAD

    N = self.num_steps
    dr = np.random.uniform(*dist_range, size=N)
    dphi = np.random.uniform(*yaw_range, size=N)
    dtheta = np.random.uniform(*pitch_range, size=N)
    x_tilt = np.random.uniform(*tilt_range, size=N)
    y_tilt = np.random.uniform(*tilt_range, size=N)

    # make first step below feet
    dr[0] = 0.0
    dphi[0] = 0.0
    dtheta[0] = np.pi / 2

    dr[1:3] = self.init_step_separation
    dphi[1:3] = 0.0
    dtheta[1:3] = np.pi / 2

    x_tilt[0:3] = 0
    y_tilt[0:3] = 0

    dphi = np.cumsum(dphi)

    dx = dr * np.sin(dtheta) * np.cos(dphi)
    dy = dr * np.sin(dtheta) * np.sin(dphi)
    dz = dr * np.cos(dtheta)

    # Fix overlapping steps
    dx_max = np.maximum(np.abs(dx[2:]), self.step_radius * 2.5)
    dx[2:] = np.sign(dx[2:]) * np.minimum(dx_max, self.dist_range[1])

    x = np.cumsum(dx)
    y = np.cumsum(dy)
    z = np.cumsum(dz)

    return np.stack((x, y, z, dphi, x_tilt, y_tilt), axis=1)
  
if __name__ == "__main__":
  allsteps = Allsteps()
  print(allsteps.terrain)