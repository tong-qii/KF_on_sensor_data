import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from filterpy.kalman import predict
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import math
import quaternion
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


import sys
sys.path.insert(1, './Kalman-and-Bayesian-Filters-in-Python/kf_book')
import book_plots as book_plots
from mkf_internal import plot_track
from filterpy.common import Saver


# clean the data and get the final dataframe with no repeated counters
def getDFS(df_raw):
    df_53 = df_raw.iloc[:, 0:13]
    df_53.columns = ['sensor',"DATA","x","y","z","normal1","normal2","normal3","transverse1","transverse2",
                 "transverse3","status","cost_func"]
    # timer 
    counter = df_raw.iloc[:,-1]
    counter.name = 'sw_counter'
    df_get = pd.concat([df_53, counter], axis=1)
     
    # get channels and get the first unique counter values
    sensors=df_get['sensor'].unique().tolist()
    test = df_get.groupby(['sensor'],as_index = False)
    test = list(test)
    dfs = list()
    tlen = list()
    for i in range(0,len(sensors)): 
        dd = test[i][1].drop_duplicates('sw_counter', keep="last")
        dd.reset_index(drop=True, inplace=True)       
        t = len(dd['sw_counter'])
        tlen.append(t)
        dfs.append(dd)
    
    return dfs, tlen

# clean the data and keep all the records even with repeated records
def getDFS_withrpt(df_raw):
    df_53 = df_raw.iloc[:, 0:13]
    df_53.columns = ['sensor',"DATA","x","y","z","normal1","normal2","normal3","transverse1","transverse2",
                 "transverse3","status","cost_func"]
    # timer 
    counter = df_raw.iloc[:,-1]
    counter.name = 'sw_counter'
    df_get =pd.concat([df_53, counter], axis=1)
     
    # get channels and get the first unique counter values
    sensors=df_get['sensor'].unique().tolist()
    test = df_get.groupby(['sensor'],as_index = False)
    test = list(test)
    dfs = list()
    tlen = list()
    for i in range(0,len(sensors)): 
        dd = pd.DataFrame(test[i][1])
        dd.reset_index(drop=True, inplace=True)
        t = len(dd['sw_counter'])
        tlen.append(t)
        dfs.append(dd)
    
    return dfs, tlen

# plot x y z postion data
def plotXYZ(dfs):
    fig,axs = plt.subplots(len(dfs),1,figsize=(8,12))
    for i in range(0, len(dfs)): 
        axs[i].set_title("x,y,z of Channel: {}".format(dfs[i].iloc[1,0]))
        axs[i].plot(dfs[i].index.values,dfs[i]['x'],label = 'x')
        axs[i].plot(dfs[i].index.values,dfs[i]['y'],label = 'y')
        axs[i].plot(dfs[i].index.values,dfs[i]['z'],label = 'z')
        axs[i].legend()
        
    fig.subplots_adjust(hspace=0.8)    
    plt.show()

# plot cost functions
def plotCF(dfs):
    fig,axs = plt.subplots(len(dfs),1,figsize=(8,12))
    for i in range(0, len(dfs)): 
        axs[i].set_title("Cost function of Channel: {}".format(dfs[i].iloc[1,0]))
        axs[i].plot(dfs[i].index.values,dfs[i]['cost_func'],label = 'cost function')
        axs[i].legend()
        
    fig.subplots_adjust(hspace=1)    
    plt.show()
    
# plot xyz in 3D plot
def plot_xyz_in3D(filename,df):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection ='3d')
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title("3D potitions of ["+filename+"]")
    # Data for three-dimensional scattered points
    zdata = df['z']
    xdata = df['x']
    ydata = df['y']
    c = np.arange(0, len(ydata), 1, dtype=int)
    ax.scatter(xdata, ydata, zdata, c =c, cmap='Blues')


# plot normal and transverse unitvectors into the same 3D plot   
def plot_normalstransverse(filename, normals,transverses):
    soa = normals.to_numpy()
    soa2 = transverses.to_numpy()
    start = np.zeros_like(soa)
    norm = np.hstack((start,soa))
    tran = np.hstack((start,soa2))
    X, Y, Z, U, V, W = zip(*norm)
    A,B,C,D,E,F = zip(*tran)

    cs = plt.cm.Reds(np.arange(0, len(U), 1, dtype=int))
    css = plt.cm.Greens(np.arange(0, len(D), 1, dtype=int))

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(-1, 0, 0, 3, 0, 0, color='#aaaaaa',linestyle='dashed')
    ax.quiver(0, -1, 0, 0,3, 0, color='#aaaaaa',linestyle='dashed')
    ax.quiver(0, 0, -1, 0, 0, 3, color='#aaaaaa',linestyle='dashed')
    ax.quiver(X, Y, Z, U, V, W, color = cs)
    ax.quiver(A,B,C,D,E,F,color=css)
    ax.set_xlim([-1,1.1])
    ax.set_ylim([-1,1.1])
    ax.set_zlim([-1,1.1])
    
    plt.title("Unit vectors of ["+filename+"]")
    plt.show()

# to plot polar coordinates
def plotPolar(index, color, title, maxTheta, phi, theta): 
    axs = plt.subplot(1, 3, index, projection='polar') 
    plt.plot(phi, theta, linewidth=0.5, color=color) 
    plt.grid (True)
    axs.set_rmax(maxTheta)
    axs.set_title(title)
    axs.set_rlabel_position(-22.5) # Move radial labels away from plotted line 
    axs.set_rticks ([2 , 4, 6, 8]) # Less radial ticks


# plot polar orientation axe with dataframe list for all sensors
def plotPolarOrientationAxe(filename, dflist):
    xaxis = [1., 0., 0.] 
    yaxis = [0., 1., 0.] 
    zaxis = [0., 0., 1.] 
    axs = []
    
    for c in range(0, len(dflist)):
        fig , axes = plt.subplots (nrows=1, ncols=3)
        #fig.suptitle("Polar Orientation Axes ["+filename+"]", x=0.1, ha="left")
        vn0 = dflist[c][['normal1','normal2','normal3']].iloc[0,:]
        vt0 = dflist[c][['transverse1','transverse2','transverse3']].iloc[0,:]
        vx0 = np.cross(vn0, vt0)
        
        # Reference axes at t=0
        cosx = np.array([np.dot(xaxis, vn0),np.dot(xaxis, vt0), np.dot(xaxis, vx0) ])
        cosy = np.array([np.dot(yaxis, vn0),np.dot(yaxis, vt0), np.dot(yaxis, vx0) ])
        cosz = np.array([np.dot(zaxis, vn0),np.dot(zaxis , vt0), np.dot(zaxis , vx0) ])
        thetax = np.empty( (len(dflist[c]), 1) ) 
        phix = np.empty( (len(dflist[c]), 1) ) 
        thetay = np.empty( (len(dflist[c]), 1) ) 
        phiy = np.empty( (len(dflist[c]), 1) ) 
        thetaz = np.empty( (len(dflist[c]) , 1) ) 
        phiz = np.empty( (len(dflist[c]), 1) )
        
        
        for t in range(0, len(dflist[c])):
            vnt = np.array(dflist[c][['normal1','normal2','normal3']].iloc[t,:])
            vtt = np.array(dflist[c][['transverse1','transverse2','transverse3']].iloc[t,:])
            vxt = np.cross(vnt, vtt)
            newAxis = cosx[0]* vnt + cosx[1]* vtt + cosx[2]* vxt 
            x, y, z = newAxis[0] , newAxis[1] , newAxis[2]
            ryz = math.sqrt(y*y + z*z)
            thetax[t] = math.atan(ryz/x) * 180./math.pi 
            phix[t] = math.atan(z/y)
            newAxis = cosy [0]* vnt + cosy [1]* vtt + cosy [2]* vxt 
            x, y, z = newAxis[0] , newAxis[1] , newAxis[2]
            rxz = math.sqrt(x*x + z*z)
            thetay[t] = math.atan(rxz/y) * 180./math.pi 
            phiy[t] = math.atan(z/x)
            newAxis = cosz [0]* vnt + cosz [1]* vtt + cosz [2]* vxt 
            x, y, z = newAxis[0] , newAxis[1] , newAxis[2]
            rxy = math.sqrt(x*x + y*y)
            thetaz [ t ] = math. atan(rxy/z) * 180./math.pi 
            phiz[t] = math.atan(y/x)
        
        maxTheta = 8
        title = dflist[c].iloc[1,0] + 'x'
        plotPolar(1, 'red', title , maxTheta, phix, thetax) 
        title = dflist[c].iloc[1,0] + 'y'
        plotPolar(2, 'green', title , maxTheta, phiy, thetay) 
        title = dflist[c].iloc[1,0] + 'z'
        plotPolar(3, 'blue', title , maxTheta, phiz , thetaz)
        plt.tight_layout() 
        plt.show()
 

# calulate quaternion from positions, the two list should keep 1 time lag
def get_quaternion_fromposition(lst1,lst2):
    qvec = []

    for i,coord1 in enumerate(lst1):
        M=np.matrix(np.outer(coord1,lst2[i]))
        
        N11=float(M[0][:,0]+M[1][:,1]+M[2][:,2])
        N22=float(M[0][:,0]-M[1][:,1]-M[2][:,2])
        N33=float(-M[0][:,0]+M[1][:,1]-M[2][:,2])
        N44=float(-M[0][:,0]-M[1][:,1]+M[2][:,2])
        N12=float(M[1][:,2]-M[2][:,1])
        N13=float(M[2][:,0]-M[0][:,2])
        N14=float(M[0][:,1]-M[1][:,0])
        N21=float(N12)
        N23=float(M[0][:,1]+M[1][:,0])
        N24=float(M[2][:,0]+M[0][:,2])
        N31=float(N13)
        N32=float(N23)
        N34=float(M[1][:,2]+M[2][:,1])
        N41=float(N14)
        N42=float(N24)
        N43=float(N34)

        N=np.matrix([[N11,N12,N13,N14],\
                  [N21,N22,N23,N24],\
                  [N31,N32,N33,N34],\
                  [N41,N42,N43,N44]])


        values,vectors=np.linalg.eig(N)
        w=list(values)
        mw=max(w)
        quat= vectors[:,w.index(mw)]
        quat=np.array(quat).reshape(-1,)
        quat = quaternion.as_quat_array(quat)
        #qvec[i] = np.array([quat[0],quat[1],quat[2],quat[3]])
        qvec.append(quat)
    
    return qvec


# plot only orientation_vectors, the input should be quaternion arrays
def plot_orientation_vectors(qvec): 
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1,1.1])
    ax.set_ylim([-1,1.1])
    ax.set_zlim([-1,1.1])
    ax.quiver(-1, 0, 0, 3, 0, 0, color='#aaaaaa',linestyle='dashed')
    ax.quiver(0, -1, 0, 0,3, 0, color='#aaaaaa',linestyle='dashed')
    ax.quiver(0, 0, -1, 0, 0, 3, color='#aaaaaa',linestyle='dashed')
    
    cs = plt.cm.Reds(np.arange(0, len(qvec), 1, dtype=int))
    css = plt.cm.Greens(np.arange(0, len(qvec), 1, dtype=int))
    csss = plt.cm.Blues(np.arange(0, len(qvec), 1, dtype=int))
    
    #cs = plt.cm.jet(np.linspace(0,1,len(qvec)))
    
    slicedCM = plt.cm.Reds(np.linspace(0, 1, len(qvec))) 

    
    yaxis = np.array([0. , 1. , 0.]) 
    zaxis = np.array([0. , 0. , 1.])
    
    for i in range(0, len(qvec)):
        yaxis = rotate(qvec[ i ] , yaxis) 
        zaxis = rotate(qvec[ i ], zaxis) 
        xaxis = np.cross(yaxis, zaxis)
        ax.quiver(0., 0., 0., xaxis[0], xaxis[1], xaxis[2], linewidth=0.5,
                   arrow_length_ratio=0.1, color = cs[i])
        ax.quiver(0., 0., 0., yaxis[0], yaxis[1], yaxis[2], linewidth=0.5,
                   arrow_length_ratio=0.1, color=css[i])
        ax.quiver(0., 0., 0., zaxis[0], zaxis[1], zaxis[2], linewidth=0.5,
                   arrow_length_ratio =0.1, color = csss[i])
    fig.show()
    

# plot the points with light to dark colors of orientation vectors, the input should be quaternion arrays
def plot_vector_trajectory(qvec):
    quats = quaternion.as_float_array(qvec)
    # df_quats = pd.DataFrame(df_quats,columns = ['qw','qx','qy','qz'])
    
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection ='3d')
    plt.xlabel("X axis label")
    plt.ylabel("Y axis label")

    ax.quiver(-1, 0, 0, 3, 0, 0, color='#aaaaaa',linestyle='dashed')
    ax.quiver(0, -1, 0, 0,3, 0, color='#aaaaaa',linestyle='dashed')
    ax.quiver(0, 0, -1, 0, 0, 3, color='#aaaaaa',linestyle='dashed')
    
    # Data for three-dimensional scattered points
    yaxis = np.array([0. , 1. , 0.]) 
    zaxis = np.array([0. , 0. , 1.])
    smallAngle = 0.02
    # myq = np.quaternion(math.cos(smallAngle), math.sin(smallAngle), 0., 0.)
    xs=[]
    ys=[]
    zs=[]
    for i in range(0, len(qvec)):
        yaxis = rotate(qvec[ i ] , yaxis) 
        zaxis = rotate(qvec[ i ] , zaxis) 
        xaxis = np.cross(yaxis, zaxis)
        xs.append(xaxis)
        ys.append(yaxis)
        zs.append(zaxis)
    
    xs=np.vstack(xs)
    ys=np.vstack(ys)
    zs=np.vstack(zs)
        
    c = np.arange(0, len(qvec), 1, dtype=int)
    ax.scatter(xs[:,0], xs[:,1], xs[:,2], c =c, cmap='Reds')
    ax.scatter(ys[:,0], ys[:,1], ys[:,2], c =c, cmap='Greens')
    ax.scatter(zs[:,0], zs[:,1], zs[:,2], c =c, cmap='Blues')
    #ax.plot(xs[:,0], xs[:,1], xs[:,2], color = 'red')
    #ax.plot(ys[:,0], ys[:,1], ys[:,2], color = 'green')
    #ax.plot(zs[:,0], zs[:,1], zs[:,2], color = 'blue')
    

# plot only one directon unit vectors into 3D space
# the input should be either normal or transverse unit vectors
def plot_unitvector_points(normals):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection ='3d')
    plt.xlabel("X axis label")
    plt.ylabel("Y axis label")

    xdata = normals.iloc[:,0]
    ydata = normals.iloc[:,1]
    zdata = normals.iloc[:,2]
    
#     xdata2 = transverse.iloc[:,0]
#     ydata2 = transverse.iloc[:,1]
#     zdata2 = transverse.iloc[:,2]
    
    c = np.arange(0, len(normals), 1, dtype=int)
#    cs = np.arange(0, len(transverse), 1, dtype=int)
    ax.scatter(xdata, ydata, zdata, c =c, cmap='Greens')
#    ax.scatter(xdata2, ydata2, zdata2, c =cs, cmap='Reds')


# rotate vec 1 to vec 2
def get_rotation_matrix(vec2, vec1=np.array([1, 0, 0])):
    """get rotation matrix between two vectors using scipy"""
    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))
    r = R.align_vectors(vec2, vec1)
    return r[0].as_matrix()


# get quaternions using the base vector method 
# this is only one line of transformation from unitvector to quaternion
def get_quaternions_from_unitvectors(nvt, tvt):
    ndt=np.array([0,0,1])
    tdt=np.array([0,1,0])
    # calculate rotation matrix m1 that rotates ndt to nvt
    m1=get_rotation_matrix(vec1 = ndt,vec2= nvt)
    # rotate tdt by m1
    vec1=m1@tdt
    # calc m2, the rotation between vec1 and tvt
    m2 = get_rotation_matrix(vec1 = vec1, vec2 = tvt)
    # combine the two rotations to one rotation matrix  
    M=m2@m1
    qs = Quaternion(matrix = M)
    quats = qs.normalised.elements.tolist()
    quats = quaternion.as_quat_array(quats)
    return quats


# get all quaternions from all unitvectors
# return quaternion arrays qVec
def get_allquaternions_from_unitvectors(normals,transverses):
    unitto_quaternions = []
    for i in range(0,len(normals)):
        nvt = normals.to_numpy()[i]
        tvt = transverses.to_numpy()[i]
        quats = get_quaternions_from_unitvectors(nvt,tvt)
        unitto_quaternions.append(quats)
    return unitto_quaternions   



# compute rotation series from all positions
# newts is a series of time indicator
# allpos is xyz in numpy array
def computeRotationSeriesFromAllPositions(newts, allpos): 
    # Implements approach by Horn 1987
    vec4 = np.zeros( (len(allpos), 4) )
    vec4[0][0] = 1.
    qvec = quaternion.as_quat_array(vec4)
    for t in range(1, len(newts)):
        S = np.zeros( (3, 3) )
        for i in range(0, 3):
            for j in range(0, 3):
                S[i][j] += allpos[t][i]*allpos[t-1][j]

        M=np.zeros( (4, 4) ) 
        M[0][0] = ( S[0][0] + S[1][1] + S[2][2])
        M[1][1] = ( S[0][0] - S[1][1] - S[2][2])
        M[2][2] = (-S[0][0] + S[1][1] - S[2][2])
        M[3][3] = (-S[0][0] - S[1][1] + S[2][2])

        M[0][1] =M[1][0] = ( S[1][2] - S[2][1])
        M[0][2] =M[2][0] = ( S[2][0] - S[0][2])
        M[0][3] =M[3][0] = ( S[0][1] - S[0][1])

        M[1][2] =M[2][1] = ( S[0][1] + S[0][1])
        M[1][3] =M[3][1] = ( S[2][0] + S[0][2])

        M[2][3] =M[3][2] = ( S[1][2] + S[2][1] )

        eigenVals , eigenVecs = np.linalg.eig(M)
        reigenVals = np.real(eigenVals) 
        reigenVecs = np.real(eigenVecs)

        maxIndex = np.where(reigenVals == np.amax(reigenVals)) 
        maxEigen = reigenVecs[:,maxIndex[0][0]]

        qvec[t] = np.quaternion(maxEigen[0],maxEigen[1],maxEigen[2],maxEigen[3])

    return qvec


def rotate(q, v, sign=1.):
    qxyz = np.array([q.x, q.y, q.z])
    # Sign of last term determines direction of rotation . Sign must be +/= 1. 
    return np.array(v + 2.*np.cross(qxyz, np.cross(qxyz, v)) + 2.*math.copysign(1.,sign)*q.w*np.cross(qxyz, v))

# Plot reference axes in 3D projections
def plot3DAxes(ax, lim):
    xm, ym, zm = np.array([[-lim,0,0],[0,-lim,0],[0,0,-lim]]) 
    xp, yp, zp = np.array([[2*lim,0,0],[0,2*lim,0],[0,0,2*lim]])
    ax.quiver(xm,ym,zm,xp,yp,zp,arrow_length_ratio=0.1,color='gray',
              linestyle='solid', linewidth=1.)

    
# Plot quaternion axes in 3D
def plotQuaternionSeries (fname , qVec) :
    # Resulting matplotlib window will need to be stretched on the monitor to be useful .
    fig = plt.figure( figsize =(8,8) , constrained_layout=True)
    fig.suptitle('Quaternion Series ['+fname+']', x=.1, ha='left')
    lim = 2*math.pi
    axs = fig.add_subplot (121 , projection='3d' ,
                           xlim=(-lim , lim ) , ylim=(-lim , lim ) , 
                           zlim=(-lim , lim ) ) 
    
    plot3DAxes ( axs , .95* lim )
    
    for i in range(0, len(qVec)):
        rotVec , angle = quaternionToRotationAxis(qVec[ i ])
        axs.quiver(0., 0., 0., rotVec[0], rotVec[1], rotVec[2], linewidth=0.5, arrow_length_ratio=0.1, color='red')
    
    lim = 1.05
    axs = fig.add_subplot (122 , projection='3d' ,
                           xlim=(-lim , lim ) , ylim=(-lim , lim ) , 
                           zlim=(-lim , lim ) ) 
    plot3DAxes ( axs , 1. )
    yaxis = np.array([0. , 1. , 0.]) 
    zaxis = np.array([0. , 0. , 1.])
    smallAngle = 0.02
    # myq = np.quaternion(math.cos(smallAngle), math.sin(smallAngle), 0., 0.)
    for i in range(0, len(qVec)):
        yaxis = rotate(qVec[ i ] , yaxis) 
        zaxis = rotate(qVec[ i ] , zaxis) 
        xaxis = np.cross(yaxis, zaxis)
        axs.quiver(0., 0., 0., xaxis[0], xaxis[1], xaxis[2], linewidth=0.5,
                   arrow_length_ratio=0.1, color='red')
        axs.quiver(0., 0., 0., yaxis[0], yaxis[1], yaxis[2], linewidth=0.5,
                    arrow_length_ratio=0.1, color='green')
        axs.quiver(0., 0., 0., zaxis[0], zaxis[1], zaxis[2], linewidth=0.5,
                    arrow_length_ratio =0.1, color='blue')


def quaternionToRotationAxis (q) :
    rotVec = quaternion.as_rotation_vector(q) 
    angle = np.linalg.norm(rotVec , axis=-1)
    # rotVec = normalized(rotVec)
    return rotVec , angle


# get quat dataframe from quat array
def get_quatDataFrame(quat_array):
    ttt = quaternion.as_float_array(quat_array)
    df_quats = pd.DataFrame(ttt)
    df_quats.columns = ['qw','qx','qy','qz']
    df_quats= df_quats.astype(float)
    return df_quats

# calculate angular velocity from quaternions
# input is dataframe of quaternions and correct time interval between each records
def angular_velocity(quats, t):
    from scipy.interpolate import InterpolatedUnivariateSpline as spline
    R = quats.to_numpy()
    Rdot = np.empty_like(R)
    for i in range(4):
        Rdot[:, i] = spline(t, R[:, i]).derivative()(t)
    R = quaternion.from_float_array(R)
    Rdot = quaternion.from_float_array(Rdot)
    return quaternion.as_float_array(2*Rdot/R)[:, 1:]



# plot angular velocities
# input is the result from function angular_velocity
def plot_angular_velocity(angvelocity):
    df_angvs = pd.DataFrame(angvelocity)
    df_angvs.columns = ['w1','w2','w3']

    plt.figure(figsize=(10,5))
    plt.plot(df_angvs.index.values[1:-2],df_angvs['w1'][1:-2],label = 'w1')
    plt.plot(df_angvs.index.values[1:-2],df_angvs['w2'][1:-2],label = 'w2',alpha = 0.5)
    plt.plot(df_angvs.index.values[1:-2],df_angvs['w3'][1:-2],label = 'w3')
    plt.legend(loc="best")
    plt.title("Angular velocities with t = 250 milliseconds ")



# calculate euler from quaternion for one record
def quaternion_to_euler(w, x, y, z): 
        t0 =2 * (w * x + y * z)
        t1 = 1 - 2 * (x * x + y * y)
        phi = math.atan2(t0, t1)
        t2 = 2 * (w * y - z * x)
        t2 = 1 if t2 > 1 else t2
        t2 = -1 if t2 < -1 else t2
        theta = math.asin(t2)         
        t3 = 2 * (w * z + x * y)
        t4 = 1 - 2 * (y * y + z * z)
        psi = math.atan2(t3, t4) 
        return phi, theta, psi


# get all eulers from all quaterions 
# input is dataframe of quats
def get_eulers_from_allquaterions(df_quats):
    eulers = []
    for i in range(0,len(df_quats)):
        euler = quaternion_to_euler(df_quats['qw'][i],df_quats['qx'][i],df_quats['qy'][i],df_quats['qz'][i])
        eulers.append(euler)
    return eulers


# plot eulers
# input is the result from function get_eulers_from_allquaterions
def plot_eulers(eulers):
    df_eulers = pd.DataFrame(eulers)
    df_eulers.columns = ['phi','theta','psi']
    plt.figure(figsize=(10,5))
    plt.plot(df_eulers.index.values,df_eulers['phi'],label = 'phi')
    plt.plot(df_eulers.index.values,df_eulers['theta'],label = 'theta')
    plt.plot(df_eulers.index.values,df_eulers['psi'],label = 'psi')
    plt.legend(loc="best")
    plt.title("Change of Euler Angles")


# plot single sensor xyz data and cost function
# input is the dataframe of a single sensor
def plot_singlesensor_xyzCF(df_deepA):
    fig, (ax1, ax2) = plt.subplots(2,figsize=(10,8))
    ax1.plot(df_deepA.index.values,df_deepA['x'],label = 'x')
    ax1.plot(df_deepA.index.values,df_deepA['y'],label = 'y')
    ax1.plot(df_deepA.index.values,df_deepA['z'],label = 'z')
    ax1.legend()
    ax1.set_title("x,y,z for deep breathing")
    ax2.plot(df_deepA.index.values, df_deepA["cost_func"],label="cost function")
    ax2.set_title("cost_function")
    plt.subplots_adjust(hspace=0.3)
    #plt.savefig("deepbreath.png")














































