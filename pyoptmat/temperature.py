"""
  Temperature interpolation formula of various kinds
"""

import torch
import torch.nn as nn

class TemperatureParameter(nn.Module):
  """
    Superclass of all temperature-dependent parameters

    This class takes care of scaling the end result, if required

    Args:
      scaling (optional):   how to scale the temperature-dependent values,
                            defaults to no scaling
  """
  def __init__(self, *args, scaling = lambda x: x, **kwargs):
    super().__init__(*args, **kwargs)
    self.scaling = scaling

  def forward(self, T):
    """
      Return the actual parameter value

      Args:
        T:      current temperature
    """
    return self.scaling(self.value(T))

class ConstantParameter(TemperatureParameter):
  """
    A parameter that is constant with temperature

    Args:
      pvalue:       the constant parameter value
  """
  def __init__(self, pvalue, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.pvalue = pvalue

  def value(self, T):
    """
      Pretty simple, just return the value!

      Args:
        T:          current batch temperatures
    """
    return self.pvalue

  @property
  def shape(self):
    return self.pvalue.shape

class KMRateSensitivityScaling(TemperatureParameter):
  """
    Parameter that scales as:

      $\frac{-\mu b^3}{kTA}

    where $\mu$ further depends on temperature

    Args:
      A:        Kocks-Mecking slope parameter, sets shape
      mu:       scalar, temperature-dependent shear modulus
      b:        scalar, Burgers vector
      k:        scalar, Boltzmann constant
  """
  def __init__(self, A, mu, b, k, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.A = A
    self.mu = mu
    self.b = b
    self.k = k

  def value(self, T):
    """
      Actual temperature-dependent value

      Args:
        T:      current temperatures
    """
    return -self.mu(T) * self.b**3.0 / (self.k * T * self.A)

  @property
  def shape(self):
    return self.A.shape

class KMViscosityScaling(TemperatureParameter):
  """
    Parameter that varies as

      $\exp{B} \mu \dot{\varepsilon}_0^{-1/n}

    where $B$ is the Kocks-Mecking intercept parameter and the
    rest are defined in the `KMRateSensitivityScaling` object.
    
    $n$ is the rate sensitivity, again given by the `KMRateSensitivityScaling`
    object

    Args:
      A:        Kocks-Mecking slope parameter
      B:        Kocks-Mecking intercept parameter, sets shape, must be
                same shape as A
      mu:       scalar, temperature-dependent shear modulus
      eps0:     scalar, reference strain rate
      b:        scalar, Burger's vector
      k:        scalar, Boltzmann constant
  """
  def __init__(self, A, B, mu, eps0, b, k, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.A = A
    self.B = B
    self.mu = mu
    self.eps0 = eps0
    self.b = b
    self.k = k

    self.n = KMRateSensitivityScaling(self.A, self.mu, self.b, self.k)

  def value(self, T):
    """
      Actual temperature-dependent value

      Args:
        T:      current temperatures
    """
    n = self.n(T)
    return torch.exp(self.B) * self.mu(T) * self.eps0**(-1.0/n)

  @property
  def shape(self):
    return self.B.shape

class PolynomialScaling(TemperatureParameter):
  """
    Mimics np.polyval using Horner's method to evaluate a polynomial

    Args:
      coefs:        polynomial coefficients in the numpy convention (highest order 1st)
  """
  def __init__(self, coefs, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.coefs = coefs

  def value(self, T):
    res = torch.zeros_like(T) + self.coefs[0]
    for c in self.coefs[1:]:
      res *= T
      res += c

    return res

  @property
  def shape(self):
    return self.coefs[0].shape

class Inter1dScaling(TemperatureParameter):
  """
    Mimics np.piecewise linear function

  """

  def __init__(self, x_new, x, y, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.x_new = x_new
    self.x = x
    self.y = y
  
  def sel(self, name, ind):
    return torch.gather(name, 1, ind)
    
  def value(self, T):
    eps = torch.finfo(self.y.dtype).eps
    v = {}
    y_new = torch.zeros(self.x_new.size())
    ind = y_new.long()  
    
    torch.searchsorted(self.x.contiguous(),
            self.x_new.contiguous(), out=ind)    
    
    ind -= 1
    
    ind = torch.clamp(ind, 0, self.x.shape[1] - 1 - 1)
    
    v['slopes'] = (
       (self.y[:, 1:]-self.y[:, :-1])
       /
       (eps + (self.x[:, 1:]-self.x[:, :-1]))
      )
 
    return self.sel(self.y, ind) + self.sel(v['slopes'], ind)*(self.x_new - self.sel(self.x, ind))  

  @property
  def shape(self):
    return self.x_new.shape
  
  
  
class ProposedChabocheJumpWidthScaling(TemperatureParameter):

  def __init__(self, b, X, g, k, tau, eps_dot, gamma_dot, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.b = b
    self.X = X
    self.g = g
    self.k = k
    self.tau = tau
    self.eps_dot = eps_dot
    self.gamma_dot = gamma_dot


  def value(self, T):
    """
      Actual temperature-dependent value

      Args:
        T:      current temperatures
    """
    L_b = self.X*self.b*(1-self.k*T[...,None]/(self.tau*self.b**3)*torch.log(self.eps_dot/self.gamma_dot))/self.g

    return L_b
    
  @property
  def shape(self):
    return (self.X.shape)

class ProposedChabocheScaling(TemperatureParameter):

  def __init__(self, mu, b, X, g, k, tau, eps_dot, gamma_dot, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.mu = mu
    self.b = b
    self.X = X
    self.g = g
    self.k = k
    self.tau = tau
    self.eps_dot = eps_dot
    self.gamma_dot = gamma_dot

    
    self.L_b = ProposedChabocheJumpWidthScaling(self.b, self.X, 
            self.g, self.k, self.tau, self.eps_dot, self.gamma_dot)

  def value(self, T):
    """
      Actual temperature-dependent value

      Args:
        T:      current temperatures
    """    
    L_b = self.L_b(T)
    C = self.mu(T)[...,None]*self.b/L_b
    return C
    
  @property
  def shape(self):
    return (self.X.shape)  
