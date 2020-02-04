##### COLOUR PALETTES

def dark_f(x,factor):
# this function takes a coordinate x and makes it darker by multiplying it by factor
# so a lineal gradient across the new dark colour coming from white passes through the original color
  return (1-factor+factor*x)


# X go from white (1,1,1) to orange (0.97,0.58,0.12)
cdict_yol_X_dict = {'red':   ((0.0, 1.0, 1.0),
                   (1.0, 0.97, 0.97)),

         'green': ((0.0,1.0, 1.0),
                   (1.0, 0.58, 0.58)),

         'blue':  ((0.0, 1.0, 1.0),
                   (1.0, 0.12, 0.12))
        }

# X go from white (1,1,1) to orange (0.97,0.58,0.12)
dfac = 1.4
cdict_yol_X_dark_dict = {'red':   ((0.0, 1.0, 1.0),
                   (1.0, dark_f(0.97,dfac), dark_f(0.97,dfac))),

         'green': ((0.0,1.0, 1.0),
                   (1.0, dark_f(0.58,dfac), dark_f(0.58,dfac))),

         'blue':  ((0.0, 1.0, 1.0),
                   (1.0, dark_f(0.12,dfac), dark_f(0.12,dfac)))
        }

# Y go from white (1,1,1) to green (0.22,0.71,0.29)
cdict_yol_Y_dict = {'red':   ((0.0, 1.0, 1.0),
                   (1.0, 0.22, 0.22)),

         'green': ((0.0,1.0, 1.0),
                   (1.0, 0.71, 0.71)),

         'blue':  ((0.0, 1.0, 1.0),
                   (1.0, 0.29, 0.29))
        }

# Y go from white (1,1,1) to green (0.22,0.71,0.29)
dfac = 1.4
cdict_yol_Y_dark_dict = {'red':   ((0.0, 1.0, 1.0),
                   (1.0, dark_f(0.22,dfac), dark_f(0.22,dfac))),

         'green': ((0.0,1.0, 1.0),
                   (1.0, dark_f(0.71,dfac), dark_f(0.71,dfac))),

         'blue':  ((0.0, 1.0, 1.0),
                   (1.0, dark_f(0.29,dfac), dark_f(0.29,dfac)))
        }

# Z go from white (1,1,1) to green (0.82,0.30,0.85)
cdict_yol_Z_dict = {'red':   ((0.0, 1.0, 1.0),
                   (1.0, 0.82, 0.82)),

         'green': ((0.0,1.0, 1.0),
                   (1.0, 0.30, 0.30)),

         'blue':  ((0.0, 1.0, 1.0),
                   (1.0, 0.85, 0.85))
        }

# Z go from white (1,1,1) to green (0.82,0.30,0.85)
dfac = 1.4
cdict_yol_Z_dark_dict = {'red':   ((0.0, 1.0, 1.0),
                   (1.0, dark_f(0.82,dfac), dark_f(0.82,dfac))),

         'green': ((0.0,1.0, 1.0),
                   (1.0, dark_f(0.30,dfac), dark_f(0.30,dfac))),

         'blue':  ((0.0, 1.0, 1.0),
                   (1.0, dark_f(0.85,dfac), dark_f(0.85,dfac)))
        }

# RedGreen from Red (1,0,0) to green (0,1,0):
cdict_RedtoGreen=  {'red':   ((0.0, 1.0, 1.0),
                   (1.0,0.0,0.0)),

         'green': ((0.0,0.0, 0.0),
                   (1.0,1.0,1.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0,0.0))
        }

# Alpha to Blue
cdict_AlphatoBlue=  {'red': 
                  ((0.0, 0.0, 0.0),
                   (1.0,0.0,0.0)),

         'green': ((0.0,0.0, 0.0),
                   (1.0,0.0,0.0)),

         'blue':  ((0.0, 1.0, 1.0),
                   (1.0, 1.0,1.0)),

         'alpha':  ((0.0, 0.0, 0.0),
                   (1.0, 1.0,1.0)
                   )
        }


# MIXING COLORS 

def colormix(color1,color2):

  return (np.array(color1)+np.array(color2))*0.5


