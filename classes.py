'''
classes.py
This is All the classes used for the Ray Tracer.
For the most part the classful structure is followed.
Some things deviated over time for speed/simplicity. e.g. The color classes is larger not used, colors are
just np.arrays.
Classes:
  Vector : defines a vector, point, or ray in Cartesian space
  World : holds all the values needed to build a scene.
  Reflection : The reflection class defines BRDF from their parameters
  Model : stores the shape defintions that can exist in the world, as well as thier intersection functions
  Texture : container for all the maps associated with the surface ( and material) of objects
  Object :  contains all the attributes associated with an object in a scene.
  Camera : describes the camera, and holds the pixel matrix.
  Light : a light object
  Image : creates an image file from pixels. can save raw values, or give tone reproduciton params
'''


import intersects as inter
import numpy as np
import heapq as hp
from copy import deepcopy
''''
Vector - defines a vector, point, or ray in Cartesian space
  @params:
    v :  the (x,y,z) vector
    p :  the origin point
    q :  the end point
    l2 :  the distance
    nv :  the direction vector
  @methods:
    rotate : rotate a vector around a given point
    add : add two vectors
    negate : return a vector in the opposite direction
    reflect : return a ray_vector in the perfectly reflected direction around a normal, option to return halfway vector
    refract : return a ray_vector refracted based on given indexs and normal
    
'''
class Vector:
  def __init__(self, p1=np.array((0, 0, 0), dtype="float64"), p2=None, ray_origin=False):
    if p2 is not None:
      self.v = np.array(p2-p1) if ray_origin is False else p1  # non normalized vector
      self.p = p1 if ray_origin is False else p2  # origin of ray if not at world/camera origin
      self.q = p2  # from p to q, unless Ray then origin
    else:
      self.v = np.array(p1)  # non normalized vector
      self.p = np.array(p1)  # point
      self.q = np.array((0, 0, 0))   # origin of ray when at world/camera origin
    self.l2 = inter.norm(self.v)  # Euclidean distance of vector
    self.nv = self.v/self.l2 if self.l2 != 0 else np.array((0, 0, 0), dtype="float64")  # direction vector

  def rotate(self, ax=np.array((0, 0, 0), dtype="float64"), origin=np.array((0, 0, 0), dtype="float64"), t="deg"):
    if t == "deg":
      a = []
      for i in ax:
        a.append(np.radians(i))
    cx = np.cos(a[0])
    sx = np.sin(a[0])
    cy = np.cos(a[1])
    sy = np.sin(a[1])
    cz = np.cos(a[2])
    sz = np.sin(a[2])
    x_m = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    y_m = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    z_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    # ORDER MATTERS IF components of vector are 0 -- need to generalize
    return Vector(np.dot(z_m, np.dot(y_m, np.dot(x_m, (self.p - np.array(origin))))) + np.array(origin))


  def add(self, v, sub=0, p=0):
    if p == 1:
      if sub == 1:
        return Vector(v.p, self.p)
      else:
        return Vector(v.negate().p, self.p)
    else:
      return Vector(self.nv)

  def negate(self):
    return Vector(self.q, self.p)

  def reflect(self, norm, hw=False, origin=None):
    a = inter.dot(self.nv, norm)*norm
    ro = True if origin is not None else False
    return Vector(self.nv - a/2, origin, ray_origin=ro) if hw else Vector(self.nv - 2*a, origin, ray_origin=ro)

  # n2 is inner, n1 is outer i.e. ray starts in n1 **like the textbooks**
  def refract(self, ip, sn, n2, n1, inner=False): # backwards numbering is intentional
    # do the snell
    snell_ratio = n1/n2

    entrance_angle = inter.dot(self.nv, sn) if inner else inter.dot(self.nv, -sn)
    a = 1 - (snell_ratio**2 * (1 - entrance_angle**2))
    if a < inter.epsilon:
      return self.reflect(sn)
    b = self.nv - (entrance_angle * sn)
    #temp = Vector(snell_ratio * b + (sn * (a**.5)), ip, ray_origin=True)
    #print("refract", snell_ratio, self.p, snell_ratio * b + sn * a**.5)
    #print(f"\t{self.nv} {inter.norm(self.nv)} {entrance_angle}")
    #print(f"\t{temp.nv} {inter.norm(temp.nv)} {inter.dot(temp.nv, sn)}")

    return Vector(snell_ratio * b + sn * a**.5, ip, ray_origin=True)

  def __str__(self):
    return f"{self.v} Normalized: {self.nv} L2: {self.l2} Origin{self.p}"


'''
  This class holds all the values needed to build a scene.
    @params:
      color : the abient color value (sky)
      ambiance : the intestity of the background color
      camera : the camera that view the world ( could be list, but is just one atm)
      shape_dict : a dictonary of shapes that exist in the world. Used to validate objects before being added
      lights : a list of the lights that exist in the scene
      objects : a list of objects that exist in the scene
      world_refraction : the index of refraction of the "air"

    @methods:
      add_object : used to add an object to the worlds object list. Handles validation of shape, and give unique ID
      spawn : the ray tracer
'''
class World:
  def __init__(self, c=(0, 0, 0), o=(), s=(), ambiance=100000, bny=None, cam=None):
    self.color = np.array(c, dtype="float64")  # this will not be affected by phong, but will be reduced by tone normalization need to scale
    self.ambiance = ambiance  # power of "the sun"
    self.color *= ambiance
    self.objects = []
    self.camera = cam
    self.next_id = 1
    self.shape_dict = dict(s)
    self.lights = []
    self.world_refraction = 1
    for i in o:
      self.add_object(i, is_light=type(i).__name__ == "Light")
    # this should produce same numbers for same scene
    # use utc if you want actual random seed
    self.rng = np.random.default_rng(len(self.objects))

  """
  -- Add an object to the world.
  -- Objects must have a Model that exists in the world's shape_dict, and have the appropriate attributes defined.
  -- If the object can exist it is given a model object (that has its intersect function defined), and assigned
     a unique ID.
  """
  def add_object(self, o, is_light=False):
    if o.model_type in self.shape_dict:
      x = self.shape_dict[o.model_type]
    else:
      return "That Model is not defined in this world"
    for i in x.attributes:
      if i not in o.attr_dict:
        return "Missing Attributes, cant create object in this world"
    o.model = x
    o.attr_list = x.attributes
    o.o_ID = self.next_id
    self.next_id += 1
    if is_light:
      self.lights += [o]
    else:
      self.objects += [o]
    return "Object created", o.o_ID

#  increasing rd detects shadows on points not in view of camera
#  and currently would draw them on the wrong side of the object

  def spawn(self, v, d=0, MAX_DEPTH=8, ignore=(), ray_type="camera", outer_refraction_index=None,):
    if outer_refraction_index is None:
      outer_refraction_index = self.world_refraction
    save_refl_color = np.array((0, 0, 0), dtype="float64")   # used for recursion
    save_trans_color = np.array((0, 0, 0), dtype="float64")  # used for recursion
    if d > MAX_DEPTH:
      if ray_type == "shadow":
        return np.array((0, 0, 0), dtype="float64"), 1, 1
      return np.array((0, 0, 0), dtype="float64"), inter.draw_distance, save_refl_color, save_trans_color
    v_n = v.negate()
    # Find closest intersection
    temp_intersects = []
    search_space = self.objects
    for o in search_space:
      x = o.model.ray_intersect_func(o, v)
      for i in x:
        if i[0] > inter.epsilon:
          # give ths garbo heapq api some help
          hp.heappush(temp_intersects, (i[0], self.rng.random(), [i[0], i[1], o, i[2]]))
    if len(temp_intersects) != 0:
      closest = hp.heappop(temp_intersects)[2]
      object_distance = closest[0]
    else:
      object_distance = inter.draw_distance
    if object_distance >= inter.draw_distance:  # if no intersection
      if ray_type == "shadow":
          return np.array((0, 0, 0), dtype="float64"), 1, 1
      else:
        return self.color, object_distance, save_refl_color, save_trans_color
    else:  # if intersection
      object_point = closest[3]
      # need to give texture the already rotated, object centered point
      texture_point = Vector(object_point - closest[2].w_pos).rotate(-1*closest[2].orientation).v
      object_color = closest[2].texture.color_map(texture_point)
      o_ref = closest[2].reflection
      surface_norm = closest[1] if closest[2].texture.norm_map is None else closest[2].texture.norm_map(object_point)
      reflection_coef = closest[2].texture.reflection_map
      # Transmission value is percent of non reflected light to transmit
      transmission_coef = closest[2].texture.transmission_map
      transmission_coef_corrected = transmission_coef
      #  if transmission + reflection == 1 do not scale, let brdf give exactly 0
      # allow greater than one for emission
      if transmission_coef >= 1:
        transmission_coef -= reflection_coef
      # if there is transmisison log scale it around .5
      # idk if this is actualy right but it didnt look right when .9 transmission was still very bright
      # also i didnt test for reflection > 0. so idk how that looks
      # I think afunctino that is a bit slower in the middle but still just as fast at the edge would be better
      # or maybe just an input map to keep input of .4 - .6 closer to .5
      elif transmission_coef > 0:
        transmission_coef_corrected = (1-reflection_coef) * (1 + np.e**(-20 * (closest[2].texture.transmission_map - .5))) ** -1

      refraction_index = closest[2].texture.refraction_map
      # for each light get local illumination
      d_s_color = 0
      d_s_distance = 1
      if ray_type != "shadow" and (1-transmission_coef-reflection_coef) > inter.epsilon:  # Ray is not a shadow ray
        for light in self.lights:
          # spawn a shadow ray
          s = Vector(object_point, light.w_pos)
          r = s.negate().reflect(surface_norm, hw=o_ref.blin)
          s_factor = self.spawn(s, d=d+4, ignore=[], ray_type="shadow")
          #  Get the Local illumination
          intensity = object_color * light.attr_dict['intensity'] * (1-transmission_coef_corrected-reflection_coef)
          if o_ref.model == "phong":
            # diffuse component
            d_s_color = object_color * o_ref.diffuse*abs(inter.dot(s.nv, surface_norm))
            # specular component
            d_s_color += light.color * (abs(inter.dot(r.nv, v_n.nv))**o_ref.exponent)*o_ref.specular
          elif o_ref.model == "strauss":
            #  Qd
            d_s_color = object_color * o_ref.diffuse*inter.dot(surface_norm, s.nv)
            #  Qs
            d_s_color += light.color * pow(-(np.dot(r.nv, v.nv)) + 0j, o_ref.exponent).real*o_ref.specular
          # shadow color does not go to camera ?
          d_s_color = (d_s_color + (s_factor[0]*reflection_coef/s_factor[1]**2)) * intensity * s_factor[2]
          d_s_distance = s.l2**2
        # Add in the Global
      trans_color = 0
      refl_color = 0
      r_distance = 1
      t_distance = 1
      if reflection_coef > inter.epsilon:
        refl_values = self.spawn(v.reflect(surface_norm, origin=object_point), d=d+1, ignore=[], ray_type="reflection")
        refl_color = reflection_coef * refl_values[0]
        save_refl_color = refl_color + refl_values[2]
        r_distance = refl_values[1]
      if transmission_coef > inter.epsilon:
        if (ray_type == "transmission" and closest[2] in [x[0] for x in ignore]):
          trans_values = self.spawn(v.refract(object_point, surface_norm, self.world_refraction, outer_refraction_index, inner=True),
                                   d=d + 1, outer_refraction_index=self.world_refraction,
                                   ignore=[(closest[2], closest[3])], ray_type="transmission")
        else:
          trans_values = self.spawn(v.refract(object_point, surface_norm, refraction_index, outer_refraction_index),
                                   d=d+1, outer_refraction_index=refraction_index,
                                   ignore=[(closest[2], closest[3])], ray_type="transmission")
        trans_color = trans_values[0]*transmission_coef_corrected
        save_trans_color = trans_color + trans_values[3]
        t_distance = trans_values[1]

    # the amount of photons that reach any given point from a source follows the inverse squaare law
    # as such the sum of photons that reach a point from multiple sources follows the inverse square law
    # that means we must divide the total number of photons by the distance they travelled. However not all
    # photons traveld the same distance, so we cant simply divide them all by the same number
    # we also cant use the average, becasue 1 photon infiinty away would negate a million photons 1 away
    # so we use the weighted average, of each indivual source divided by its distance traveld as the distance
    # to pass up the reccursion chain.
    d_weights = np.array([np.log2(np.nansum(trans_color)+inter.epsilon) - 2*np.log2(t_distance+inter.epsilon),
                          np.log2(np.nansum(refl_color)+inter.epsilon) - 2 * np.log2(r_distance+inter.epsilon),
                          np.log2(np.nansum(d_s_color)+inter.epsilon) - 2*np.log2(d_s_distance+inter.epsilon)])
    d_weights = np.array([i + min(d_weights) for i in d_weights])
    w_avg = (d_weights/(np.nansum(d_weights)+inter.epsilon))
    d_travel = (t_distance**2, r_distance**2, d_s_distance**2) * w_avg

    r_color = np.array((trans_color, refl_color, d_s_color))

    if ray_type == "shadow":
      return r_color[0] + r_color[1] + r_color[2],\
             np.nansum(d_travel) ** .5 + object_distance,\
             transmission_coef + reflection_coef

    return r_color[0] + r_color[1] + r_color[2], np.nansum(d_travel)**.5 + object_distance, save_refl_color, save_trans_color


  def __str__(self):
    x = ""
    for i in self.objects:
      x += f"{i}\n"
    return x


'''
  The reflection class defines BRDF from their parameters
  I think initally this was goign to be more robust but it stopped at phong ans strauss
  
  @params: pretty much follow their respective models.
'''
class Reflection:
  def __init__(self, a=0, k=.6, s=None, e=5, blin=False, model="phong", t=0.0, m=0.0):
    self.model = model
    if self.model == "phong":
      self.ambient = a
      self.specular = s if s is not None else .3
      self.diffuse = k
      self.exponent = e
      self.blin = blin
    elif self.model == "strauss":
      self.blin = False
      self.s = s if s is not None else 0  # smoothness
      self.t = t  # transparency
      self.m = m  # metalness
      self.rd = (1-self.s**3)*(1-t)
      self.d = (1-m*self.s)
      self.diffuse = self.d * self.rd
      self.exponent = 3/(1-self.s)  # h
      self.rn = (1-t) - self.rd
      self.specular = min(1.0, self.rn + (self.rn + .1))    # rj k=.1 cant do attenuation atm/ too lazy. j=1
  # No idea
  def execute_model(self):
    return 0


'''
The Model class stores the shapes that can exist in the world, as well as thier intersection functions
 This is used by the World class 
'''
class Model:  # aka Shape
  def __init__(self, a=(), i_ntr=None, n=None):
    self.n = n
    self.attributes = list(a)  # list of attributes that an object must define to use intersection
    self.ray_intersect_func = i_ntr  # intersection function for a model


'''
  Texturs : container for all the maps associated with the surface ( and material) of objects
  desined with the intent that the values would be functions passed in. some are currently only constants
'''
class Texture:
  def __init__(self, cm=np.array((0, 0, 0), dtype="float64"), nm=None, rm=0, tm=0, rfm=1):
    self.color_map = cm  # function that takes position on object and returns rgb
    self.norm_map = nm  # function that takes position on object and returns normal vector
    #  At the moment the 3 below are just constants, not functions
    self.reflection_map = rm  # function that takes position on object and returns reflection coefficient
    self.transmission_map = tm  # function that takes position on object and returns transmission coefficient
    self.refraction_map = rfm  # function that takes position on object and returns index of refraction


'''
  Object 
   contains all the attributes associated with an object in a scene. Stored as a list by the World
   @params:
     o_ID : unique id assinged by the world
     name : readable name of object (non-unique)
     w_pos : cartesian position in the world
     model_type: which Model class the object belongs too, and will be validated against
     attr_dict : a dictionary that stores the attributes of an object, as defined by its Model
                  e.g. radius, vertices, faces
     model : the model object assigned by the World after validation
     attr_list : the list of attributes assigned by the World after validation
     reflection : the Reflection object that describes the shader to use
     texture : the Texture object that contains mappings
'''
class Object:
  def __init__(self, n="None", m="Point", m_a=None, refl=None, text=None):
    if m_a is None:
        m_a = {"xyz": [0, 0, 0], "up": [0, 1, 0]}
    if refl is None:
      refl  = {}
    if text is None:
      text = Texture(None)
    self.o_ID = "000000"
    self.name = n
    self.w_pos = np.array(m_a["xyz"], dtype="float64")  # world position
    self.orientation = np.array(m_a["up"])
    self.model_type = m
    self.attr_dict = m_a
    self.model = None
    self.attr_list = None
    self.reflection = Reflection(**refl)
    self.texture = text
    # depricated machine learnign thigns
    self.ml_params = {'ML_REFLECTION': 0, 'ML_TRANSMISSION': 0, 'ML_DIFFUSE': 1, 'ML_SPECULAR': 1}

  def __repr__(self):
    return self.name
  def __str__(self):
    return f"{self.name}\n\tObject Id: {self.o_ID:05d}\n\tworld position: {self.w_pos}\n\t"
  def __lt__(self, other):
    return self
'''
@method
  render : renders the scene based on the world position of the camera and the worlds objects
'''
class Camera(Object):
  # @param la - the world position you want the viewport to be centered on
  # @param res - (width, height) tuple number of pixels
  def __init__(self, n="None", m_a=None, f=1.0, res=(100, 100), vp_w=100, vp_h=100, la=(0, 0, 0), up=(0, 0, 1), sampling=1):
    Object.__init__(self, n=n, m_a=m_a)
    self.c_ID = "000000"
    self.f = f  # focal Length
    self.res = np.array(res, dtype="int32")
    self.pw = (vp_w/self.res[0])  # pixel width
    self.ph = (vp_h/self.res[1])  # pixel height
    self.vp = [[0 for i in range(self.res[0])]for i in range(self.res[1])]  # viewport/film plane
    self.vp_ml = [[0 for i in range(self.res[0])]for i in range(self.res[1])]
    self.la = Vector(la, self.w_pos)  # look at vector
    self.up = Vector(up)  # up vector
    self.sampling = sampling

  def render(self, w):
    # use component to create vector projection scaled  to focal length.
    vp_d = self.f * self.la.nv
    # do this projection to all ray vectors via the pixel height deltas
    dx = np.cross(self.up.v, vp_d)
    if dx[0] == 0 and dx[1] == 0 and dx[2] == 0:
      dx = np.array((1,1,1))
    else:
      dx = dx/inter.norm(dx)
    dy = np.cross(vp_d, dx)
    if dy[0] == 0 and dy[1] == 0 and dy[2] == 0:
      dy = np.array((1,1,1))
    else:
      dy = dy/inter.norm(dy)


    dx, dy = (dx*self.pw, dy*self.ph)

    res = self.res  # (w,h)
    row_c = res[0] // 2  # row center pixel
    vp_d = self.w_pos - vp_d + dy*(res[1]//2)   # move vp_d to top of film plane
    if res[1] % 2 == 0:
      vp_d = vp_d - (dy/2)  # move down half a pixel if even num pixels
    if res[0] % 2 == 0:
      vp_d = vp_d - (dx/2)
      off_index = 0
    else:
      off_index = 1
    for i in range(res[1]):  # for each row
      self.render_loop(off_index, vp_d, i, row_c, dx, dy, w)
      vp_d = vp_d - dy

  def render_loop(self, off_index, vp_d, i, row_c, dx, dy, w):
    if off_index == 1:  # odd number of pixels do center
      c_sam = w.spawn(Vector(self.w_pos, vp_d))
      self.vp[i][row_c] = c_sam[0] / c_sam[1]**2
    rc = vp_d
    lc = vp_d
    previous_color_r = [0, 0, 0]
    previous_color_l = [0, 0, 0]
    for j in range(1, row_c + 1):  # for each pixel
      rc = rc + dx
      lc = lc - dx
      center_sample_r = w.spawn(Vector(self.w_pos, rc))  # (np.array((0,0,0),dtype='float32'))
      center_sample_l = w.spawn(Vector(self.w_pos, lc))
      center_sample_r = center_sample_r[0] / center_sample_r[1]**2 # (np.array((0,0,0),dtype='float32'))
      center_sample_l = center_sample_l[0] / center_sample_l[1]**2
      #print(center_sample_r, previous_color_r)
      # only super sample if the previous pixel is not close to current pixel
      if self.sampling == 1 or inter.equals(center_sample_r, previous_color_r, tol=.3):
        self.vp[i][row_c + j - (1 - off_index)] = center_sample_r
        previous_color_r = center_sample_r
      else:
        super_ray_r, samples_r = self.recurse_sample_rays(rc, dx, dy, w, center_sample_r, 1, self.sampling)
        previous_color_r = super_ray_r / samples_r
        self.vp[i][row_c + j - (1 - off_index)] = previous_color_r

      if self.sampling == 1 or np.allclose(center_sample_l, previous_color_l, atol=.3):
        self.vp[i][row_c - j] = center_sample_l
        previous_color_l = center_sample_l
      else:
        super_ray_l, samples_l = self.recurse_sample_rays(lc, dx, dy, w, center_sample_l, 1, self.sampling)
        previous_color_l = super_ray_l / samples_l
        self.vp[i][row_c - j] = previous_color_l

  def recurse_sample_rays(self, c, dx, dy, w, sr, d, max):
    added_samples = 1
    return_sample_ray = [sr[0], sr[1], sr[2]]
    offsets = self.get_offsets(c, dx, dy)
    for s_s_i in range(4):
      super_ray_sample = w.spawn(Vector(self.w_pos, offsets[s_s_i]))
      super_ray_sample = super_ray_sample[0] / super_ray_sample[1]**2

      if d < max and not inter.equals(super_ray_sample, sr, tol=.01):
        temp_sample = self.recurse_sample_rays(offsets[s_s_i], dx/2, dy/2, w, super_ray_sample, d+1, max)
        return_sample_ray += temp_sample[0]
        added_samples += temp_sample[1]
      else:
        return_sample_ray += super_ray_sample
        added_samples += 1

    return return_sample_ray, added_samples

  def get_offsets(self, center, dx, dy):
    off_x = dx / 4
    off_y = dy / 4
    return center + off_x + off_y, center + off_y - off_x, center + off_x - off_y, center - off_y - off_x

  def print_film(self):
    str1 = "[\n"
    for i in self.vp:
      str1 += "\t["
      for j in i:
        str1 += f"{j}, "
      str1 += "]\n"
    str1 += "]"
    print(str1)

  def __str__(self):
    return Object.__str__(self) + f"\n\tLook at: {self.la}\n\tUp: {self.up}\n\tFocal Length: {self.f}\n\t"
class Light(Object):
  def __init__(self, c=(1, 1, 1), n="None", m_a=None):
    Object.__init__(self, n=n, m_a=m_a)
    self.l_ID = "000000"
    self.color = np.array(c, dtype='float64')
'''
@method : normalize_tone : does the tone reproduction stuff
'''
class Image:
  def __init__(self, save_name='test', save=False, save_raw=False, cam=None,
               encoding="PNG", tr_method="ward", MAX_DISPLAY_ILLUMINANCE=1, p=(), gamma=2.2):
    self.args = p
    self.pixels = deepcopy(cam.vp) if cam is not None else deepcopy(self.args['pixels'])
    self.pixels_ml = deepcopy(cam.vp_ml) if cam is not None else None

    self.num_pixels = len(self.pixels) * len(self.pixels[0])
    self.tone_method = tr_method

    self.ld_max = MAX_DISPLAY_ILLUMINANCE
    self.illuminance_constants = (.27, .67, .06)
    self.gamma = gamma
    if tr_method != "NONE":
      self.normalize_tone(args=self.args)

    if save:
      with open(f'{save_name}', 'w') as fn:
        wr = ""
        for i in self.pixels:
          for j in i:
            for k in j:
              wr += f"{k} "
        fn.write(f"{len(self.pixels)} {len(self.pixels[0])} {len(self.pixels[0][0])}\n{wr}")

    #else:
    if encoding == "PNG":
      self.rgb_matrix = self.color_to_rgb("byte")
      self.image = self.create_png(save_name)

  def normalize_tone(self, args=()):
    sf = 1
    l_min = float('inf')
    l_max = float('-inf')
    # calculate log average luminance
    accum = 0
    for row in range(len(self.pixels)):
      for col in range(len(self.pixels[0])):
        t = inter.dot(self.illuminance_constants, self.pixels[row][col] )
        l_max = t if t > l_max else l_max
        l_min = t if t < l_min else l_min
        accum += np.log2(max(t, inter.epsilon))
    accum /= self.num_pixels
    l_avg = 2**accum
    if self.tone_method == "uniform_scale":
      sf /= self.ld_max
    elif self.tone_method == "ward":
      sf = ((1.219 + (self.ld_max/2)**.4)/(1.219 + l_avg**.4))**2.5
    elif self.tone_method == "reinhard":
      sf = args['a']/l_avg
    elif self.tone_method == "log_adaptive":
      l_max /= l_avg
      sf = self.ld_max * .001 / np.log10(l_max + 1)
    np_pixels = np.array(self.pixels)


    if self.tone_method == "reinhard":
      np_pixels = self.ld_max / (1 / (np_pixels * sf) + 1)
    elif self.tone_method == 'log_adaptive':
      l_w = np_pixels / l_avg
      np_pixels = sf
      np_pixels *= np.log(l_w + 1)
      b = np.log(args['b'])/np.log(.5)
      np_pixels /= np.log(2+(8*((l_w/l_max)**b)))
    else:
      np_pixels *= sf

    self.pixels = np_pixels**(1/self.gamma)

  def color_to_rgb(self, num_type):
    x = []
    for row in self.pixels:
      temp = []
      for color in row:
        if num_type == "byte":
          temp += [int(max(0, min(255, np.floor(f * 255.0)))) for f in color]
        else:
          temp += color
      x.append(temp)
    return x

  def create_png(self, fn):
    import png
    f = open(fn+".png", 'wb')
    png.Writer(len(self.pixels[0]), len(self.pixels), greyscale=False).write(f, self.rgb_matrix)
    return f

if __name__ == '__main__':
 pass
