import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import scipy.stats as stats
from scipy import optimize, signal
from scipy.ndimage import gaussian_filter
import math
import cv2 as cv
from skimage.morphology import skeletonize, thin
from bwmorph import *
from scipy import interpolate
from interparce import interparc
from scipy.interpolate import CubicSpline
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
import matplotlib.animation as animation

np.random.seed(38)

def dy(distance, m):
    return m*dx(distance, m)

def dx(distance, m):
    return math.sqrt(distance**2/(m**2+1))

def render(data):
    ## windowing (window and level adjustment)
    wincenter=7500.0
    winwidth=8100.0
    min = wincenter - (winwidth/2)
    max = wincenter + (winwidth/2)

    fig, ax = plt.subplots(1,2)
    ax1, ax2 = ax.ravel()
    ax1.imshow(data, cmap='gray')
    ax1.set_title('raw data')
    ax2.imshow(data, cmap='gray', vmin=min, vmax=max)
    ax2.set_title('windowing view')
    plt.plot()
    plt.show()

def getPeak(f_, component=1, means_array=None):
    f=f_.reshape(-1,1)
    g = mixture.GaussianMixture(n_components=component,covariance_type='spherical', reg_covar=0.00001, means_init=means_array)
    g.fit(f)
    weights = g.weights_
    means = g.means_
    covars = g.covariances_
    # print(means)
    params = []
    for (m,c,w) in zip(g.means_, g.covariances_, g.weights_):
        params.append((m[0],c,w))
    s = sorted(params, key = lambda x: x[0])
    f_axis = f.ravel()
    f_axis.sort()
    for i in range(component):
        if(means[i] == s[0][0]):
            continue
    f_axis = f_.ravel()
    f_axis.sort()
    # for peak in s:
    #     plt.plot(f_axis,peak[2]*stats.norm.pdf(f_axis,peak[0],np.sqrt(peak[1])).ravel())
    # plt.show()
    return s[-2]

def getDist(x, y, m, mask):
    if(m==0):
        print('zero', x, y)
    tmp = np.zeros(mask.shape)
    xs = []
    ys = []
    for d in range(150):
        x_ = x+dx(d,m)
        y_ = y + dy(d,m)
        tmp[int(np.round(y_)),int(np.round(x_))] = 1
        xs.append(x_)
        ys.append(y_)
        x_ = x+-dx(d,m)
        y_ = y - dy(d,m)
        tmp[int(np.round(y_)),int(np.round(x_))] = 1
        xs.append(x_)
        ys.append(y_)
    intersect = np.bitwise_and(np.where(mask>0, True, False), np.where(tmp>0, True, False))
    points = np.where(endpoints(intersect))
    return [[b,a] for a, b in zip(points[0], points[1])]
    points = np.where(intersect>0)
    plt.plot(points[1], points[0])

def g(x, A, mu, sigma):
    return A / (sigma * math.sqrt(2 * math.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))

def cost(parameters):
    a, b, c = parameters
    # y has been calculated in previous snippet
    return np.sum(np.power(g(x, a, b, c) - y_hist, 2)) / len(x)
    
data = np.load('cbct.npy')
coronal_mip = np.max(data, axis=1)

plt.imshow(coronal_mip, cmap='gray')
plt.title('coronal MIP')
plt.show()
render(coronal_mip)

plt.hist(coronal_mip.ravel(), bins=256, density=True)
f = coronal_mip.ravel().astype(np.float)
peak = getPeak(f,3)
T = peak[0] + 3*np.sqrt(peak[1])
f_axis = f.ravel()
f_axis.sort()
plt.plot(f_axis,peak[2]*stats.norm.pdf(f_axis,peak[0],np.sqrt(peak[1])).ravel(), c='cyan')
plt.plot([T]*2, [0,0.0005])
plt.show()

coronal_mask = np.where(coronal_mip > T, 1, 0)
plt.imshow(coronal_mask, cmap='gray')
plt.show()

y_hist = np.sum(coronal_mask, axis=1)
x = range(coronal_mask.shape[0])
plt.plot(x, y_hist)
result = optimize.minimize(cost, [0, 300, 1])
plt.plot(x, g(x, result.x[0], result.x[1], result.x[2]))
plt.show()
w = 3 * result.x[2]
a_s = int(np.ceil(result.x[1]-1.8*w))
a_e = int(np.floor(result.x[1]+0.7*w))
plt.imshow(coronal_mask, cmap='gray')
plt.plot(range(coronal_mask.shape[1]), coronal_mask.shape[1] * [a_s], 'r')
plt.plot(range(coronal_mask.shape[1]), coronal_mask.shape[1] * [a_e], 'r')
plt.plot(range(coronal_mask.shape[1]), coronal_mask.shape[1] * [result.x[1]], 'b')
plt.show()

slices = data[a_s:a_e]
planes = []
fig = plt.figure()
for plane in slices:
    pl = plt.imshow(plane, cmap='gray')
    planes.append([pl])
ani = animation.ArtistAnimation(fig, planes, interval=50, blit=True, repeat_delay=1000)
plt.show()

axial_mip = np.max(slices, axis=0)
plt.hist(axial_mip.ravel(), bins=256, density=True)
f = axial_mip.ravel().astype(np.float)
peak = getPeak(f,4, None)
T = peak[0] + 3*np.sqrt(peak[1])
# T=peak[0] + np.sqrt(peak[1])
print('T', T)
f_axis = f.ravel()
f_axis.sort()
plt.plot(f_axis,peak[2]*stats.norm.pdf(f_axis,peak[0],np.sqrt(peak[1])).ravel(), c='cyan')
plt.plot([T]*2, [0,0.0002])
plt.show()

fig, ax = plt.subplots(1, 2)
ax1, ax2 = ax.ravel()
axial_mask = np.where(axial_mip > T, 255, 0)
ax1.imshow(axial_mask, cmap='gray')
ax1.set_title('mask')
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(120,120))
opening = cv.morphologyEx(axial_mask.astype(np.uint8), cv.MORPH_CLOSE, kernel)
smoothed = gaussian_filter(opening, sigma=3)
ax2.imshow(smoothed, cmap='gray')
ax2.set_title('morphological processed mask')
plt.show()

skeleton = skeletonize(np.where(smoothed>0,1,0))
spured  = spur(np.where(skeleton>0,0,1))
plt.imshow(spured, cmap='gray')
plt.show()

curve = np.where(spured == 0)
curve = [(a,b) for a, b in zip(curve[0], curve[1])]
s = sorted(curve, key= lambda x: x[1])
x_min = s[0][1]
x_max = s[-1][1]
interval = (x_max - x_min)/10
xs=[]
ys=[]
for i in range(11):
    x = x_min+(i*interval)
    y=np.where(spured[:, int(x)] == 0)[0]
    if(len(y)>0):
        xs.append(x)
        ys.append(y[-1])
plt.imshow(smoothed, cmap='gray')
print(axial_mip.shape)

tck = interpolate.splrep(xs, ys, s=0)
ynew = interpolate.splev(range(x_min-10, x_max+10, 1), tck, der=0)
tt = interparc(500, range(x_min-10, x_max+10, 1), ynew)
# print(tt)
plt.plot(tt[:,0], tt[:,1])
# plt.show()

cs = CubicSpline(xs,ys)
x__ = range(x_min, x_max, 1)
y__ = cs(x__)

der = cs.derivative()
d = []
for point in s:
    ps = getDist(point[1], point[0], -1/der(point[1]), smoothed)
    ps = np.array(ps)
    if(ps.shape[0] > 1):
        if(np.random.random() > 0.9):
            plt.plot(ps[:,0], ps[:,1], c='y')
        # print(ps, math.sqrt((ps[1,0] - ps[0,0])**2 + (ps[1,1] - ps[0,1])**2))
        d.append(math.sqrt((ps[1,0] - ps[0,0])**2 + (ps[1,1] - ps[0,1])**2))
plt.show()
d.sort()
T = (np.sum(d)/len(d))/2
print(T)

up=[]
down=[]
for idx, point_a in enumerate(tt):
    m = -1/der(point_a[0])
    point_b = (point_a[0]+dx(T,m), point_a[1]+dy(T,m))
    other_possible_point_b = (point_a[0]-dx(T,m), point_a[1]-dy(T,m)) # going the other way
    if m>0:
        up.append(point_b)
        down.append(other_possible_point_b)
    else:
        up.append(other_possible_point_b)
        down.append(point_b)
up=np.array(up)
down=np.array(down)
plt.imshow(axial_mip, cmap='gray')
plt.plot(tt[:,0], tt[:,1])
plt.plot(up[:,0], up[:,1], c='r')
plt.plot(down[:,0], down[:,1], c='r')
plt.show()
        
mpr = np.empty((0, data.shape[0],500))
for dist in range(int(T), 0, -1):
    plane = np.zeros((data.shape[0],500))
    for idx, point_a in enumerate(tt):
        m = -1/der(point_a[0])
        point_b = (point_a[0]+dx(dist,m), point_a[1]+dy(dist,m))
        other_possible_point_b = (point_a[0]-dx(dist,m), point_a[1]-dy(dist,m)) # going the other way
        if(m > 0):
            b = point_b
        else:
            b = other_possible_point_b
        for ii in range(data.shape[0]):
            p = data[ii]
            plane[ii,idx] = p[int(b[1])][int(b[0])]
    mpr = np.append(mpr, [plane], axis=0)
plane = np.zeros((data.shape[0],500))
for ii, p in enumerate(data):
    for idx, point_a in enumerate(tt):
        plane[ii,idx] = p[int(tt[idx,1])][int(tt[idx,0])]
mpr = np.append(mpr, [plane], axis=0)
plt.imshow(plane, cmap='gray')
plt.title('mpr')
plt.axis('off')
plt.show()
for dist in range(1,int(T)+1, 1):
    plane = np.zeros((data.shape[0],500))
    for idx, point_a in enumerate(tt):
        m = -1/der(point_a[0])
        point_b = (point_a[0]+dx(dist,m), point_a[1]+dy(dist,m))
        other_possible_point_b = (point_a[0]-dx(dist,m), point_a[1]-dy(dist,m)) # going the other way
        if(m < 0):
            b = point_b
        else:
            b = other_possible_point_b
        for ii in range(data.shape[0]):
            p = data[ii]
            plane[ii,idx] = p[int(b[1])][int(b[0])]
    mpr = np.append(mpr, [plane], axis=0)
print(mpr.shape)

fig, ax = plt.subplots(1, 3)
ax1, ax2, ax3 = ax.ravel()

ax1.imshow(np.sum(mpr, axis=0), cmap='gray')
ax1.set_title('raw pixel-wise sum')

img = np.zeros((data.shape[0], 500))
s = 4000
for i in range(data.shape[0]):
    for j in range(500):
        tmp = np.sum(np.exp(mpr[:,i,j]/s))
        img[i,j] = s*np.log(tmp)
ax2.imshow(img, cmap='gray')
ax2.set_title('thresholded')

alpha = 0.9
g = cv.GaussianBlur(img,(5,5),0.8)
enhanced = alpha*img + (1-alpha)*(img-g)
ax3.imshow(enhanced, cmap='gray')
ax3.set_title('enhanced')
plt.show()
plt.imsave('opg.png', enhanced, cmap="gray")