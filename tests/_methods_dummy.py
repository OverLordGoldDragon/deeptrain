import random
import numpy as np
from inspect import getsource
import functools


def fn1(a, b=5):
  print("wrong animal", a, b)

class Dog():
  meow = fn1
  def __init__(self):
    self.__name__ = 'd'

  def bark(self):
    print("WOOF")

  def __str__(self):
    return "<function"

  def __repr__(self):
    return "<function"


d = Dog()

barker = d.bark
mewoer = d.meow
A = 5
arr = np.random.randn(100, 100)

def __getattr__(name):
  return getattr(random, name, None)

magic = __getattr__
duplicate = fn1
_builtin = str
lambd = lambda: 1

def not_lambda():
    return 1

functools.wraps(lambda: 1)(not_lambda).__name__
