'''
DEFINES ALL THE INTERSECTION MATH FOR SHAPES
ALSO DEFINES SOME CALCULATION ATTRIBUTES, e.g epsilon and max distance
ALSO DEFINES SOME VECTOR MATH FUNCTIONS THAT ARE MORE EFFECIANT THAN USING NUMPY
'''

import numpy as np
import time
import classes as rt
epsilon = 0.0000001  # intersection round off prevention
root_epsilon = 0.0001
draw_distance = 400
class InterTest:
  def __init__(self, attrdict=None, v=None, nv=None, p=None, q=None):
    self.attrdict = attrdict
    self.v = v
    self.nv = nv
    self.p = p
    self.q = q

def point(o, v):
  return None

def sphere(s, v, internal=False):
  global root_epsilon
  global epsilon
  # im not sure if some of the changes i made are typical
  # but it removes some mulitplys, and changes some to adds
  # Its speedier, however it becomes really hard to follow from
  # the base math equations. Im 90% sure its right though.
  d = v.nv
  o_to_c = v.p - s.w_pos
  b = dot(d, o_to_c)
  c = norm_sq(o_to_c) - s.attr_dict["r"]**2
  w = (b*b - c)
  w = np.sqrt(w) if w > 0 else 0
  if w >= root_epsilon:
    t1 = (w - b)
    t2 = (-b - w)
    n1 = o_to_c + t1 * d
    if abs(t1 - t2) > epsilon:
      n2 = o_to_c + t2*d
      return [[t1, n1/norm(n1),  n1 + s.w_pos], [t2, n2/norm(n2),  n2 + s.w_pos]]
    else:
      return [[t1, n1 / norm(n1), n1 + s.w_pos]]
  return [[draw_distance, None, None]]

def triangle(o, v):
  global epsilon
  points = o.attr_dict["vertices_points"]
  d = v.nv

  e1 = points[1] - points[0]
  e2 = points[2] - points[0]
  t = v.p - points[0]

  p = cross(d, e2)
  q = cross(t, e1)
  pe1 = dot(p, e1)
  if -epsilon < pe1 < epsilon:
    return [[draw_distance, None, None]]
  else:
    pe1_inv = 1/pe1
    u = pe1_inv * dot(p, t)
    h = pe1_inv * dot(q, d)  # v
    w = pe1_inv * dot(q, e2)
    if u < epsilon or h < epsilon or u+h > 1 or w < epsilon:
      return [[draw_distance, None, None]]
  n = cross(e2, e1)
  return [[w, n/norm(n), v.p + d*w]]

def plane(o, v):
  x_1 = triangle(o.attr_dict["left"], v)
  return x_1 if x_1[0][0] != draw_distance else triangle(o.attr_dict["right"], v)

def pyramid(o, v):
  for i in o.attr_dict["faces"]:
    x = triangle(i, v)
    if x is not None:
      return x
  return None

def dot(x, y):
  return x[0]*y[0] + x[1]*y[1] + x[2]*y[2]
def norm_sq(x):
  return x[0]**2 + x[1]**2 + x[2]**2
def norm(x):
  return (x[0]**2 + x[1]**2 + x[2]**2)**.5
def cross(x, y):
  return x[1]*y[2] - x[2]*y[1],  x[2]*y[0] - x[0]*y[2], x[0]*y[1] - x[1]*y[0]
def distance(x, y):
  return norm(vector(x, y))
def vector(y, x):
  return [x[0]-y[0], x[1]-y[1], x[2]-y[2]]
def equals(x, y, tol=0):
  if x is None or y is None:
    return False
  return abs(x[0]-y[0]) <= tol and abs(x[1] - y[1]) <= tol and abs(x[2] - y[2]) <= tol
def is_zero(x, tol=epsilon):
  return abs(x[0]) <= tol and abs(x[1]) <= tol and abs(x[2]) <= tol

def pow2(x):
  return x[0]*x[0], x[1]*x[1], x[2]*x[2]

def log2(x):
  global epsilon
  return np.log2(max(x[0], epsilon)), np.log2(max(x[1], epsilon)), np.log2(max(x[2], epsilon))
if __name__ == '__main__':
  a = rt.Vector((-3, -13, -4)), rt.Vector((-19, 65, 2)), rt.Vector((-43.5, 48, 2)), rt.Vector((22.5, 4, -4))
  v_1 = rt.Vector(np.array((0, 0, 0)),np.array( (4.69, 1.26, 4.01)))
  o_1 = rt.Object(rt.Color((.179, .01, .031)), "Left floor", "Triangle", {"vertices": [a[0], a[1], a[2]],	"xyz": [-11, 26, -1]})
  o_2 = rt.Object(rt.Color((.114, .8, .107)), "Close Sphere", "Sphere", {"xyz": [4.69, 1.26, 4.05], "r": 4.25})
  tries = 100000
  s_t_1 = time.time()
  print(cross(a[0].v, a[1].v), np.cross(a[0].v, a[1].v))

  for i in range(tries):
    #triangle(o_1, v_1)
    #sphere(o_2, v_1)
    x33 = cross(a[0].v, a[1].v)
  s_t_2 = time.time()
  for i in range(tries):
    x33 = np.cross(a[0].v, a[1].v)
  print(f"MEE: {(s_t_2-s_t_1)/tries} NUMPY: {(time.time()-s_t_1)/tries}")
