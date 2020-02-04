import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import matplotlib.colorbar as colorbar
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

import seaborn as sns


def lattice(N=10, geometry = "square", periodic = False, method = "construction", Ny = 4):

    '''
    function to create a lattice object "lat" defined as a dictionary with numerical keys describing
    each cell. Each cell itself is a dictionary containing position and indexes of neighbouring cells

    N --- is the number of rows and columns (NxN lattice)
    Ny --- Aspect ratio is 1 if Ny is None, otherwise Ny is the number of cells in y-axis
    distance between cells is set to 1
    geometry --- the geometry of the lattice and the neighbours and can be:
        "square": square lattice with 4 neighbours per square
        "squarediag": square lattice with 8 neighbours (it consider diagonals)
        "hexagonal": hexagonal lattive with 6 neighbours (use an even number for N)
    periodic -- sets toroidal periiodic conditions in the tissue
    method --- calls to the two different methods to build the network that are:
        "distance" : determins the neighbours as the elments under a given distance
        "construction" : The nearest neighbours are given by the topology
    '''

    if (N%2 == 1) and (geometry == "hexagonal"):
        print("Hexagonal geometry requires an even dimension N")
    elif (method == "distance"):
        return lattice_distance(N=N, geometry = geometry, periodic = periodic)
    elif (method == "construction"):
        return lattice_construction(N=N, geometry = geometry, periodic = periodic, Ny = Ny)
    else:
        "Wrong lattice method, lattice not created"    


def lattice_distance(N=10, geometry = "square", periodic = False):  
    '''
    construction of the lattice given an interaction distance. Defult at the moment
    is chosing nearest neighbours
    '''
    print("Creating lattice...")
    lat = {}

    idx = 0
    if geometry == 'square' or geometry == 'squarediag' :
        Lx = N # Length of the tissue in the x dimension
        Ly = N # Length of the tissue in the y dimension
        for row in range(N):
            for col in range(N):
                lat[idx] = {
                    "mpos": (row,col), # matrix position
                    "xpos": 1.0*col,
                    "ypos": 1.0*row,
                    "neighbours": []  # Neighbours filled later
                }
                idx = idx + 1
    elif geometry == 'hexagonal':
        Lx = N # Length of the tissue in the x dimension
        Ly = N * np.sqrt(3)/2.0 # Length of the tissue in the y dimension
        for row in range(N):
            for col in range(N):
                lat[idx] = {
                    "mpos": (row,col), # matrix position
                    "xpos": col + 0.5 * (row % 2), # alternate rows are interpersed in x-axis 
                    "ypos": row * np.sqrt(3)/2.0, # factor comes from hexagonal packing
                    "neighbours": []  # Neighbours filled later
                }
                idx = idx + 1
    # looking for neighbours, the implementation compares distances between pairs of cells
    # it is not the most efficient but it is very flexible for any kind of lattice
    # squared distance that dictates the radius at which consider neighbours are:
    dist2geo = {'square':1.1,'hexagonal':1.1,'squarediag':2.1}        
    for cella in lat:  # we will look for neighbour of cella
        for cellb in lat:  # we will look for all possibilities cellb
                distx = abs(lat[cella]["xpos"]-lat[cellb]["xpos"])
                disty = abs(lat[cella]["ypos"]-lat[cellb]["ypos"])
                if periodic is True: # correct distances if toroidal boundary conditions are considered
                    distx = min(distx,Lx-distx) 
                    disty = min(disty,Ly-disty)
                dist2 = distx*distx + disty*disty # total Euclidean distance
                if ((dist2 < dist2geo[geometry]) and (cella != cellb)):
                    lat[cella]["neighbours"].append(cellb) # add to the list of neighbours
    print("Done!")
    return lat

def lattice_construction(N=10, geometry = "square", periodic = False, Ny = None):
    '''
    Construction of the lattice by assigning the neighbours at the same time
    that the network is created. This requires knowledge of the cell distribution
    at creation time (true for regular lattices)
    '''

    print("Creating lattice..."),
    print(geometry)
    lat = {}
    idx = 0
    if geometry == 'square':
        Nx = N
        Lx = Nx # Length of the tissue in the x dimension
        if Ny == None:
            Ny = N
        Ly = Ny # Length of the tissue in the y dimension
        for row in range(Ny):
            for col in range(Nx):
                lat[idx] = {
                    "mpos": (row,col), # matrix position
                    "xpos": 1.0*col,
                    "ypos": 1.0*row,
                    "neighbours": []
                    }
                if (idx%Nx) != Nx-1: # If it is not the last cell in the row
                    lat[idx]["neighbours"].append(idx+1) # Add neighbour to the right
                elif periodic is True:
                    lat[idx]["neighbours"].append(idx+1-Nx) # Add neighbour to the right (cyclic)
                
                if (idx%Nx) != 0: # If it is not the first cell in the row
                    lat[idx]["neighbours"].append(idx-1) # Add neighbour to the left
                elif periodic is True:
                    lat[idx]["neighbours"].append(idx-1+Nx) # Add neighbour to the left (cyclic)

                if (idx//Nx) != Ny-1: # It it is not the last row
                    lat[idx]["neighbours"].append(idx+Nx) # Add neighbour at the top
                elif periodic is True:
                    lat[idx]["neighbours"].append(idx%Nx) # Add neighbour at the top (cyclic)

                if (idx//Nx) != 0: # It it is not the first row
                    lat[idx]["neighbours"].append(idx-Nx) # Add neighbour at the bottom
                elif periodic is True:
                    lat[idx]["neighbours"].append(Nx*(Ny-1)+idx) # Add neighbour at the bottm (cyclic)                    
    
                idx = idx + 1 # Next cell

    elif geometry == 'squaredig':
        print("Geometry squaredig still does not accept lattice_construction method")

    elif geometry == 'hexagonal':
        print("Number of cells:"+str(N)+'x'+str(Ny))
        Lx = N # Length of the tissue in the x dimension
        if Ny:
            Ly = Ny * np.sqrt(3)/2.0 # Length of the tissue in the y dimension
        else:
            Ly = N
            Ny = N
        for row in range(Ny):
            for col in range(N):
                lat[idx] = {
                    "mpos": (row,col), # matrix position
                    "xpos": col + 0.5 * (row % 2), # alternate rows are interpersed in x-axis 
                    "ypos": row * np.sqrt(3)/2.0, # factor comes from hexagonal packing
                    "neighbours": []  # Neighbours filled later
                }
                # Computing of neighbours, it is not the most elegant, but is effective
                # another way could be done by defining periodicity on an array and call that array
                if (idx%N) != N-1: # If it is not the last cell in the row
                    lat[idx]["neighbours"].append(idx+1) # Add neighbour to the right
                elif periodic is True:
                    lat[idx]["neighbours"].append(idx+1-N) # Add neighbour to the right (ciclic)
                
                if (idx%N) != 0: # If it is not the first cell in the row
                    lat[idx]["neighbours"].append(idx-1) # Add neighbour to the left
                elif periodic is True:
                    lat[idx]["neighbours"].append(idx-1+N) # Add neighbour to the left (ciclic)

                if ((idx//N) % 2) == 0 : # It it is an even row
                    # Adding top neighbours for even rows
                    lat[idx]["neighbours"].append(idx+N) # Add neighbour to top right
                    if (idx%N) != 0: # If it is not the first cell in the row
                        lat[idx]["neighbours"].append(idx+N-1) # Add neighbour to the top left
                    elif periodic is True:
                        lat[idx]["neighbours"].append(idx+2*N-1) # Add neighbour to the top left
                    # Adding bottom nieghbours to even rows
                    if (idx//N) != 0 : # If it is not the first row
                        lat[idx]["neighbours"].append(idx-N) # Add neighbour to bottom right
                        if (idx%N) != 0: # If it is not the first cell in the row
                            lat[idx]["neighbours"].append(idx-N-1) # Add neighbour to the bottom left
                        elif periodic is True:
                            lat[idx]["neighbours"].append(idx-1) # Add neighbour to the bottom left (cyclic)
                    elif periodic is True:
                        if (idx%N) != 0 : # If it is not the first element on the first row
                                lat[idx]["neighbours"].append(Ny*(N-1)+idx) # Add neighbour to the bottom right (cyclic)
                                lat[idx]["neighbours"].append(Ny*(N-1)+idx-1) # Add neighbour to the bottom left (cyclic)
                        else: # If it is the first element of the first row
                                lat[idx]["neighbours"].append(Ny*(N-1)) # Add neighbour to the bottom right (cyclic)
                                lat[idx]["neighbours"].append(Ny*N-1) # Add neighbour to the bottom left (cyclic)
                
                else: # It is an odd row
                    # Adding bottom nieghbours to even rows
                    lat[idx]["neighbours"].append(idx-N) # Add neighbour to bottom left
                    if (idx%N) != N-1: # If it is not the last cell in the row
                        lat[idx]["neighbours"].append(idx-N+1) # Add neighbour to the bottom right
                    elif periodic is True:
                        lat[idx]["neighbours"].append(idx-2*N+1) # Add neighbour to the bottom left (cyclic)
                    # Adding top neighbours for odd rows
                    if (idx//N) != N-1 : # If it is not the last row
                        lat[idx]["neighbours"].append(idx+N) # Add neighbour to top left
                        if (idx%N) != N-1: # If it is not the last cell in the row
                            lat[idx]["neighbours"].append(idx+N+1) # Add neighbour to the top right
                        elif periodic is True:
                            lat[idx]["neighbours"].append(idx+1) # Add neighbour to the top left
                    if periodic is True:
                        if (idx%N) != N-1: # If it is the last row but not the last element of the row
                            lat[idx]["neighbours"].append(idx%N) # Add neighbour to the top left (cyclic)    
                            lat[idx]["neighbours"].append(idx%N + 1) # Add neighbour to the top right (cyclic)
                        else: # If it is the last element of the last row
                            lat[idx]["neighbours"].append(N-1) # Add neighbour to the top left        
                            lat[idx]["neighbours"].append(0) # Add neighbour to the top left        

                idx = idx + 1 # Next cell
    print("Done!")
    return lat


def diffuse(lat, M, D, deg, dt, neighbourdist=1):
    ''' Diffusion of a spatial concentration M along a lattice lat.
    Using a finite differences algorithm.
    Note that this is not exactly the same as an extracellular diffusion

    M ---  vector with the values of the Diffusive substance
    lat --- lattice to use for the diffusion (neighbour and position info)
    D --- diffusion coefficient
    dt --- integration timestep
    deg ---  the degradation rate
    neighbourdist --- is the distance between neighbours
         it can be set to "check" to look at the distance information in the the lattice
    '''

    N = np.zeros_like(M)  
    if neighbourdist != "check": # This will be the typical scenario
        for idx in range(len(M)):
            N[idx] = (sum(M[lat[idx]["neighbours"]])-len(lat[idx]["neighbours"])*M[idx])/(neighbourdist*neighbourdist)
    elif neighbourdist == "check":
        for idx in M:
            for neighbour in lat[idx]["neighbours"]:
                distx = abs(lat[cella]["xpos"]-lat[cellb]["xpos"])
                disty = abs(lat[cella]["ypos"]-lat[cellb]["ypos"])
                if periodic is True: # correct distances if toroidal boundary conditions are considered
                    distx = min(distx,Lx-distx) 
                    disty = min(disty,Ly-disty)
                dist2 = distx*distx + disty*disty # total Euclidean distance
                N[idx] = (M[neighbour]-M[idx])/dist2
    else:
        print("Wrong argument for neighbourdist in diffuse")

    return M + N * D * dt - deg * M * dt 

def setleft(lat, M, value):
    '''
    Set all the values of cocentration matrix M at the left boundary to a certain value.
    Useful to keep concentration constant for sinks and sources. 
    '''
    for cell in lat:
        if lat[cell]["mpos"][1] == 0:
            M[cell] = value
    return M

def setright(lat, M, value, lastcol = 'auto'):
    ''' identical to setleft but at the right column of the array'''
    # set all the values of M at the left boundary to a certain value. Useful to keep concentration constant. 
    if lastcol == 'auto': # find automatically the right boundary
        cols = [cell["mpos"][1] for cell in lat]
        lastcol = max(cols) 

    for cell in lat:
        if lat[cell]["mpos"][1] == lastcol:
            M[cell] = value
    return M


###############################################
############### PLOTTING FUNCTIONS
##############################################

def printlattice(lattice, M=[], geometry = "square", save = False, limcolor=[0,1], title = "None", titleoffset = 0, show = True, cmap = 'viridis'):
    '''' plot a concentration matrix M on a lattice '''

    positions = [(lattice[idx]["xpos"],lattice[idx]["ypos"]) for idx in lattice] # positions of the cells
    Nsides = {"square": 4, "squarediag": 4, "hexagonal": 6} # number of sides of each cell
    orientation = {"square": np.pi/4, "squarediag": np.pi/4, "hexagonal": 0} # rotation of each cell

    fig, ax = plt.subplots(1) # creation of blank figure
    sns.set_style("ticks") # seaborn styling (just for the look)
    sns.set_context("talk") 
    sns.despine() # no mirror axis
    ax.set_aspect('equal')  # aspect ratio of the figure to keep proportions of polygons
    newblack = sns.xkcd_rgb["charcoal"] # I like this black insetad of pure black

    patch_list = [] # this list will contain all the polygonal shapes
    for idx in lattice:  
        patch_list.append(
            mpatches.RegularPolygon( # add a regular polygon
                    xy=positions[idx], # at a certain position
                    numVertices=Nsides[geometry], # with certain number of sides
                    radius=0.5/np.cos(np.pi/Nsides[geometry]), # with certain radius
                    orientation=orientation[geometry], # and a certain rotation
                    edgecolor=newblack,  # and borders of color
            )
        )
    pc = collections.PatchCollection(patch_list, match_original=True) # create a collection with the list
    pc.set_clim(limcolor) # set the min and max values for the color scale 
    pc.set(array=M, cmap=cmap) # set a color for each polygon based on the given array M
    ax.add_collection(pc) # add the collection to the plotting axis
    if title != "None":
        ax.text(np.sqrt(len(lattice))/2.0,np.sqrt(len(lattice))+titleoffset-1,title)
        #print np.sqrt(len(lattice))/2.0,np.sqrt(len(lattice))
    
    maxx = max([lattice[idx]["xpos"] for idx in lattice]) # maximum x position of the polygons
    maxy = max([lattice[idx]["ypos"] for idx in lattice]) # maximum y position of the polygons
    ax.axis([-1, maxx+1, -1, maxy+1]) # set axis range for the plot
    
    cbb=fig.colorbar(pc) # create colorbar legend

    if save is not False: # if the argument save is set to a string e.g. "figure.pdf"
        plt.savefig(save) # it creates an output file 

    if show:
        plt.show()
    else:
        plt.close(fig)     

    return ax

def printlattices(lattice, M=[], geometry = "square", save = False, limcolors=[[0,1]], title = "None", titleoffset = 0, show = True,
        cmaps = ['viridis'],axisline=True, labels=None, cols = 1, timestamp = None):
    ''' 
    Plot an array of lattices for a list of concentration matrices M
    '''
    # cols is the number of columns to organize the plots

    positions = [(lattice[idx]["xpos"],lattice[idx]["ypos"]) for idx in lattice] # positions of the cells
    Nsides = {"square": 4, "squarediag": 4, "hexagonal": 6} # number of sides of each cell
    orientation = {"square": np.pi/4, "squarediag": np.pi/4, "hexagonal": 0} # rotation of each cell

    figxlen = 10
    figylen = 6
    fig = plt.figure(figsize = (figxlen,figylen)) # creation of figure

    gs1 = gridspec.GridSpec((len(M)-1)//cols+1,cols, # gridspec for the matrices of diffusion
        width_ratios=[1]*cols)
    gs1.update(left = 0.05, right = figylen/figxlen-0.05, top = 0.95, bottom = 0.05, # location of the gridspec in the figure
        wspace = 0.01)
    axes = [plt.subplot(gs1[i//cols,i%cols]) for i in range(len(M))]

    gs2 = gridspec.GridSpec(1,len(M), # gridspec for the colorbars
        width_ratios=[1]*len(M))
    gs2.update(left = figylen/figxlen, right = 0.90, top = 0.85, bottom = 0.25,
        wspace = 3.0) # location of the gridspec in the figure
    caxes = [plt.subplot(gs2[0,i]) for i in range(len(M))]   

    sns.set_style("ticks") # seaborn styling (just for the look)
    sns.set_context("talk") 
    sns.despine() # no mirror axis
    #ax.set_aspect('equal')  # aspect ratio of the figure to keep proportions of polygons
    newblack = sns.xkcd_rgb["almost black"] # I like this black insetad of pure black

    for iax, ax in enumerate(axes):
        if (axisline is False):
            axes[iax].set_frame_on(False)
        ax.set_aspect('equal')  # aspect ratio of the axis to keep proportions of polygons
        patch_list = [] #  list will contain all the polygonal shapes
        for idx in lattice:  
            patch_list.append(
                mpatches.RegularPolygon( # add a regular polygon
                    xy = positions[idx], # at a certain position
                    numVertices = Nsides[geometry], # with certain number of sides
                    radius = 0.5/np.cos(np.pi/Nsides[geometry]), # with certain radius
                    orientation = orientation[geometry], # and a certain rotation
                    edgecolor = newblack,  # and borders of color
                    linewidth = 2.0 
                )
            )
        pc = collections.PatchCollection(patch_list, match_original=True) # create a collection with the list
        if len(cmaps)>1:
            cmap = cmaps[iax]
        else:
            cmap = cmaps[0]
        pc.set(array=M[iax], cmap=cmap) # set a color for each polygon based on the given array M
        pc.set_clim(limcolors[iax]) # set the min and max values for the color scale 
        ax.add_collection(pc) # add the collection to the plotting axis
        if (title != "None" and iax==0):
            ax.text(np.sqrt(len(lattice))/2.0,np.sqrt(len(lattice))/4.0+titleoffset-1,title)
            #print np.sqrt(len(lattice))/2.0,np.sqrt(len(lattice))
        # if iax == 0:    
        ax.set_title(labels[iax])
        ax.get_yaxis().set_visible(False)
        maxx = max([lattice[idx]["xpos"] for idx in lattice]) # maximum x position of the polygons
        maxy = max([lattice[idx]["ypos"] for idx in lattice]) # maximum y position of the polygons
        ax.axis([-1, maxx+1, -1, maxy+1]) # set axis range for the plot
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks([])
        #colorbarticks = range(int(np.floor(limcolors[iax][0])),int(np.ceil(limcolors[iax][1])+1))
        #colorlabelticks = ['$10^{{ {} }}$'.format(t)  for t in colorbarticks]
        cbb = fig.colorbar(pc, cax = caxes[iax],aspect = 100) # create colorbar legend
        #cbb.set_ticks(colorbarticks)
        #cbb.set_ticklabels(colorlabelticks)
        #cbb.outline.set_visible(False)
        if labels:
            caxes[iax].text(0,-0.1,labels[iax], transform=caxes[iax].transAxes, size =15)

    if timestamp:

        caxes[1].text(0,-0.25,'time = {0:.2f} h'.format(float(timestamp)),transform=caxes[1].transAxes)

    plt.tight_layout()



    if save is not False: # if the argument save is set to a string e.g. "figure.pdf"
        plt.savefig(save,dpi=200) # it creates an output file 

    if show:
        plt.show()
    else:
        plt.close(fig)     

    return ax

def printlattices_comp(lattice, M=[], geometry = "square", save = False, limcolors=[[0,1]], title = "None", titleoffset = 0, show = True,
        cmaps = ['viridis'],axisline=True, labels=None, cols = 1, timestamp = None):

    # cols is the number of columns to organize the plots

    positions = [(lattice[idx]["xpos"],lattice[idx]["ypos"]) for idx in lattice] # positions of the cells
    Nsides = {"square": 4, "squarediag": 4, "hexagonal": 6} # number of sides of each cell
    orientation = {"square": np.pi/4, "squarediag": np.pi/4, "hexagonal": 0} # rotation of each cell

    # N_cbars is the number of colorbars
    # N_cbars = [icmap for icmap,x in enumerate(cmaps) if x!='custom']

    figxlen = 12
    figylen = 5
    fig = plt.figure(figsize = (figxlen,figylen)) # creation of figure

    gs1 = gridspec.GridSpec((len(M)-1)//cols,cols, # gridspec for the matrices of diffusion
        width_ratios=[1]*cols)
    gs1.update(left = 0.05, right = figylen/figxlen-0.05, top = 0.95, bottom = 0.05, # location of the gridspec in the figure
        wspace = 0.01)
    axes = [plt.subplot(gs1[i//cols,i%cols]) for i in range(len(M)-1)]
    # print('axes',axes)

    gs2 = gridspec.GridSpec(1,1, # gridspec for the matrices of diffusion
        width_ratios=[1])
    gs2.update(left = figylen/figxlen-0.04, right = 2*figylen/figxlen, top = 0.95, bottom = 0.05, # location of the gridspec in the figure
        wspace = 0.01)
    mixaxes = plt.subplot(gs2[0,0])
    # print('mixaxes',mixaxes)

    diffaxes = axes
    diffaxes.append(mixaxes) # this is a list containing all the axes with a diffusion array
    # print('diffaxes',diffaxes)


    gs3 = gridspec.GridSpec(1,2, # gridspec for the colorbars
        width_ratios=[1]*2)
    gs3.update(left = 2*figylen/figxlen, right = 0.92, top = 0.85, bottom = 0.25,
        wspace = 4.0) # location of the gridspec in the figure
    caxes = [plt.subplot(gs3[0,i]) for i in range(2)] # axes for the colorbars
   
    sns.set_style("ticks") # seaborn styling (just for the look)
    sns.set_context("talk") 
    sns.despine() # no mirror axis
    #ax.set_aspect('equal')  # aspect ratio of the figure to keep proportions of polygons
    newblack = sns.xkcd_rgb["almost black"] # I like this black insetad of pure black

    for iax, ax in enumerate(diffaxes):
        if (axisline is False):
            axes[iax].set_frame_on(False)
        ax.set_aspect(1.0)  # aspect ratio of the figure to keep proportions of polygons
        patch_list = [] #  list will contain all the polygonal shapes
        if iax==(len(M)-1): # bigger plot, bigger lines
            customlinewidth = 3.5
        else:
            customlinewidth = 2.5
        for idx in lattice:  
            patch_list.append(
                mpatches.RegularPolygon( # add a regular polygon
                    xy = positions[idx], # at a certain position
                    numVertices = Nsides[geometry], # with certain number of sides
                    radius = 0.5/np.cos(np.pi/Nsides[geometry]), # with certain radius
                    orientation = orientation[geometry], # and a certain rotation
                    edgecolor = newblack,  # and borders of color
                    linewidth = customlinewidth
                )
            )
        pc = collections.PatchCollection(patch_list, match_original=True) # create a collection with the list
        if len(cmaps)>1:
            cmap = cmaps[iax]
        else:
            cmap = cmaps[0]
        if cmap == 'custom': # if 'custom' then the array M contains the RGBA tuples, otherwhise is a cmap coordinate
            pc.set_facecolor(M[iax]) # set a color for each polygon based on the given array M
        else:
            pc.set(array=M[iax], cmap=cmap) # set a color for each polygon based on the given array M
            pc.set_clim(limcolors[iax]) # set the min and max values for the color scale 
        ax.add_collection(pc) # add the collection to the plotting axis
        if (title != "None" and iax==0):
            ax.text(np.sqrt(len(lattice))/2.0,np.sqrt(len(lattice))/4.0+titleoffset-1,title)
            #print np.sqrt(len(lattice))/2.0,np.sqrt(len(lattice))
        # if iax == 0:    
        ax.set_title(labels[iax])
        ax.get_yaxis().set_visible(False)
        maxx = max([lattice[idx]["xpos"] for idx in lattice]) # maximum x position of the polygons
        maxy = max([lattice[idx]["ypos"] for idx in lattice]) # maximum y position of the polygons
        ax.axis([-1, maxx+1, -1, maxy+1]) # set axis range for the plot
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks([])
        #colorbarticks = range(int(np.floor(limcolors[iax][0])),int(np.ceil(limcolors[iax][1])+1))
        #colorlabelticks = ['$10^{{ {} }}$'.format(t)  for t in colorbarticks]
        if cmap!= 'custom':
            cbb = fig.colorbar(pc, cax = caxes[iax]) # create colorbar legend
        #cbb.set_ticks(colorbarticks)
        #cbb.set_ticklabels(colorlabelticks)
        #cbb.outline.set_visible(False)
            if labels:
                caxes[iax].text(0,-0.1,labels[iax], transform=caxes[iax].transAxes, size =15)

    if timestamp:

        caxes[0].text(0,-0.25,'time = {0:.2f} h'.format(float(timestamp)),transform=caxes[0].transAxes)

    plt.tight_layout()



    if save is not False: # if the argument save is set to a string e.g. "figure.pdf"
        plt.savefig(save,dpi=200) # it creates an output file 

    if show:
        plt.show()
    else:
        plt.close(fig)     

    return ax

##################################################
############################  MOVIE OF LATTICES
###################################################


def initlatticemovie(lattice):
    global figg, axg, caxg, newblack, movieframes
    figg = plt.figure() # creation of blank figure
    axg = figg.add_subplot(121) # main axes
    caxg = figg.add_subplot(121) # colorbar axes
    sns.set_style("ticks") # seaborn styling (just for the look)
    sns.set_context("talk") 
    sns.despine() # no mirror axis
    axg.set_aspect('equal')  # aspect ratio of the figure to keep proportions of polygons
    newblack = sns.xkcd_rgb["charcoal"] # I like this black insetad of pure black

    maxx = max([lattice[idx]["xpos"] for idx in lattice]) # maximum x position of the polygons
    maxy = max([lattice[idx]["ypos"] for idx in lattice]) # maximum y position of the polygons
    axg.axis([-1, maxx+1, -1, maxy+1]) # set axis range for the plot

    movieframes = [] # this list will contain the info of every frame

def addlatticeframe(lattice, M=[], geometry = "square", save = False, limcolor=[0,1], cmap = 'viridis'):

    global pc

    positions = [(lattice[idx]["xpos"],lattice[idx]["ypos"]) for idx in lattice] # positions of the cells
    Nsides = {"square": 4, "squarediag": 4, "hexagonal": 6} # number of sides of each cell
    orientation = {"square": np.pi/4, "squarediag": np.pi/4, "hexagonal": 0} # rotation of each cell

    patch_list = [] # this list will contain all the polygonal shapes
    for idx in lattice:  
        patch_list.append(
            mpatches.RegularPolygon( # add a regular polygon
                    xy=positions[idx], # at a certain position
                    numVertices=Nsides[geometry], # with certain number of sides
                    radius=0.5/np.cos(np.pi/Nsides[geometry]), # with certain radius
                    orientation=orientation[geometry], # and a certain rotation
                    edgecolor=newblack  # and borders of color
            )
        )
    pc = collections.PatchCollection(patch_list, match_original=True) # create a collection with the list
    pc.set(array=M, cmap=cmap) # set a color for each polygon based on the given array M
    pc.set_clim(limcolor) # set the min and max values for the color scale 
    frame = axg.add_collection(pc) # add the collection to the plotting axis and save the artist in frame

    movieframes.append((frame,))
    print("frames: ", len(movieframes))

    if save is not False: # if the argument save is set to a string e.g. "figure.pdf"
        plt.savefig(save) # it creates an output file 
    

    return frame


def makelatticemovie():

    global pc

    cbb=figg.colorbar(pc) # create colorbar legend
    animation.ArtistAnimation(figg, movieframes, interval=1, blit=False, repeat_delay=30)
    anim.save('latticemoving.mp4')

#    plt.show()









