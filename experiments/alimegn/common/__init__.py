'''shared machinery for the aLIMEgn experiments'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

from . import paths, store, degrade, weights, register

register.register()   # every script that imports `common` gets this study's registries
