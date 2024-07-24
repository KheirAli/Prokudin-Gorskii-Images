#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 14:26:19 2021

@author: alirezakheirandish
"""

def IND(b_img,g_img):
    
    B = np.uint64((b_img)[::8,::8])
    G = np.uint64((g_img)[::8,::8])
    l1 = int(B.shape[0]*0.05)
    l2 = int(B.shape[1]*0.05)
    
    ind = np.array([-1,-1])
    max = - (np.inf)    
    for i in range(-20,20):
        for j in range (-20,20):
            A = np.sum(G[l1:-l1,l2:-l2]/np.linalg.norm(G[l1:-l1,l2:-l2]) * B[l1+i:-l1+i,j+l2:-l2+j]/np.linalg.norm(B[l1+i:-l1+i,j+l2:-l2+j]))
            if (A > max):
                max = A
                ind = [i,j]
    l1 = 2*l1
    l2 = 2*l2
    ind0 = np.array([-1,-1])
    max0 = - (np.inf)    
    for i in range(-30,30):
        for j in range (-30,30):
            A = np.sum(G[l1:-l1,l2:-l2]/np.linalg.norm(G[l1:-l1,l2:-l2]) * B[l1+i:-l1+i,j+l2:-l2+j]/np.linalg.norm(B[l1+i:-l1+i,j+l2:-l2+j]))
            if (A > max0):
                max0 = A
                ind0 = [i,j]
    if max0 > max :
        ind = np.copy(ind0)
    ind1 = ind * np.array([-8,-8])
    
    
    B = np.uint64((b_img)[::4,::4])
    G = np.uint64((g_img)[::4,::4])
    
    B = np.roll(B,ind * np.array([-2,-2]),axis=[0,1])
    l1 = int(B.shape[0]*0.05)
    l2 = int(B.shape[1]*0.05)
    
    ind = np.array([-1,-1])
    max = - (np.inf)    
    for i in range(-1,2):
        for j in range (-1,2):
            A = np.sum(G[l1:-l1,l2:-l2]/np.linalg.norm(G[l1:-l1,l2:-l2]) * B[l1+i:-l1+i,j+l2:-l2+j]/np.linalg.norm(B[l1+i:-l1+i,j+l2:-l2+j]))
            if (A > max):
                max = A
                ind = [i,j]
    l1 = 2*l1
    l2 = 2*l2
    ind0 = np.array([-1,-1])
    max0 = - (np.inf)    
    for i in range(-1,2):
        for j in range (-1,2):
            A = np.sum(G[l1:-l1,l2:-l2]/np.linalg.norm(G[l1:-l1,l2:-l2]) * B[l1+i:-l1+i,j+l2:-l2+j]/np.linalg.norm(B[l1+i:-l1+i,j+l2:-l2+j]))
            if (A > max0):
                max0 = A
                ind0 = [i,j]
    if max0 > max :
        ind = np.copy(ind0)
    ind2 = ind * np.array([-4,-4])
    
    B = np.uint64((b_img)[::2,::2])
    G = np.uint64((g_img)[::2,::2])
    
    B = np.roll(B,ind * np.array([-2,-2]),axis=[0,1])
    l1 = int(B.shape[0]*0.05)
    l2 = int(B.shape[1]*0.05)
    
    ind = np.array([-1,-1])
    max = - (np.inf)    
    for i in range(-1,2):
        for j in range (-1,2):
            A = np.sum(G[l1:-l1,l2:-l2]/np.linalg.norm(G[l1:-l1,l2:-l2]) * B[l1+i:-l1+i,j+l2:-l2+j]/np.linalg.norm(B[l1+i:-l1+i,j+l2:-l2+j]))
            if (A > max):
                max = A
                ind = [i,j]
    l1 = 2*l1
    l2 = 2*l2
    ind0 = np.array([-1,-1])
    max0 = - (np.inf)    
    for i in range(-1,2):
        for j in range (-1,2):
            A = np.sum(G[l1:-l1,l2:-l2]/np.linalg.norm(G[l1:-l1,l2:-l2]) * B[l1+i:-l1+i,j+l2:-l2+j]/np.linalg.norm(B[l1+i:-l1+i,j+l2:-l2+j]))
            if (A > max0):
                max0 = A
                ind0 = [i,j]
    if max0 > max :
        ind = np.copy(ind0)
    ind3 = ind * np.array([-2,-2])
#     print (max)
    
    B = np.uint64((b_img))
    G = np.uint64((g_img))
    
    B = np.roll(B,ind * np.array([-2,-2]),axis=[0,1])
    l1 = int(B.shape[0]*0.05)
    l2 = int(B.shape[1]*0.05)
    
    ind = np.array([-1,-1])
    max = - (np.inf)    
    for i in range(-1,2):
        for j in range (-1,2):
            A = np.sum(G[l1:-l1,l2:-l2]/np.linalg.norm(G[l1:-l1,l2:-l2]) * B[l1+i:-l1+i,j+l2:-l2+j]/np.linalg.norm(B[l1+i:-l1+i,j+l2:-l2+j]))
            if (A > max):
                max = A
                ind = [i,j]
    l1 = 2*l1
    l2 = 2*l2
    ind0 = np.array([-1,-1])
    max0 = - (np.inf)    
    for i in range(-1,2):
        for j in range (-1,2):
            A = np.sum(G[l1:-l1,l2:-l2]/np.linalg.norm(G[l1:-l1,l2:-l2]) * B[l1+i:-l1+i,j+l2:-l2+j]/np.linalg.norm(B[l1+i:-l1+i,j+l2:-l2+j]))
            if (A > max0):
                max0 = A
                ind0 = [i,j]
    if max0 > max :
        ind = np.copy(ind0)
    ind4 = ind * np.array([-1,-1])
    
    return ind1+ind2+ind3+ind4
    



    

def ABCD(R_image,B_image,G_image,av):
    b = [0]
    c = [0]
    d = [0]
    L = 1200
    m = 0.5
    for i in range (1,300,1):
        if np.average(av[i,:])/256 < 20 :
            b.append(i)
            c.append(i)
            d.append(i)
            continue
            
        if np.sum(R_image[i,:]<L) > R_image.shape[1]*m:
            b.append(i)
        if np.sum(B_image[i,:]<L) > B_image.shape[1]*m:
            c.append(i)
        if np.sum(G_image[i,:]<L) > G_image.shape[1]*m:
            d.append(i)
    A = min (b[-1],c[-1],d[-1]) + 1
    if A == 300 :
        L = L/4
        m = m+0.05
        for i in range (1,300,1):
            if np.average(av[i,:])/256 < 20 :
                b.append(i)
                c.append(i)
                d.append(i)
                continue
            
            if np.sum(R_image[i,:]<L) > R_image.shape[1]*m:
                b.append(i)
            if np.sum(B_image[i,:]<L) > B_image.shape[1]*m:
                c.append(i)
            if np.sum(G_image[i,:]<L) > G_image.shape[1]*m:
                d.append(i)
        A = min (b[-1],c[-1],d[-1]) + 1
        L = L*4
        m = m-0.05
    
    b = [0]
    c = [0]
    d = [0]
    for i in range (-1,-300,-1):
        if np.average(av[i,:])/256 < 20 :
            b.append(i)
            c.append(i)
            d.append(i)
            continue
            
        if np.sum(R_image[i,:]<L) > R_image.shape[1]*m:
            b.append(i)
        if np.sum(B_image[i,:]<L) > B_image.shape[1]*m:
            c.append(i)
        if np.sum(G_image[i,:]<L) > G_image.shape[1]*m:
            d.append(i)
        
    B = max (b[-1],c[-1],d[-1]) - 1
    if B == -300 :
        L = L/4
        m = m+0.05
        for i in range (-1,-300,-1):
            if np.average(av[i,:])/256 < 20 :
                b.append(i)
                c.append(i)
                d.append(i)
                continue
            
            if np.sum(R_image[i,:]<L) > R_image.shape[1]*m:
                b.append(i)
            if np.sum(B_image[i,:]<L) > B_image.shape[1]*m:
                c.append(i)
            if np.sum(G_image[i,:]<L) > G_image.shape[1]*m:
                d.append(i)
        B = max (b[-1],c[-1],d[-1]) - 1
        L = L*4 
        m = m - 0.05
    b = [0]
    c = [0]
    d = [0]
    for i in range (1,300,1):
        if np.average(av[:,i])/256 < 20 :
            b.append(i)
            c.append(i)
            d.append(i)
            continue
            
        
        if np.sum(R_image[:,i]<L) > R_image.shape[0]*m:
            b.append(i)
        if np.sum(B_image[:,i]<L) > B_image.shape[0]*m:
            c.append(i)
        if np.sum(G_image[:,i]<L) > G_image.shape[0]*m:
            d.append(i)
        
    C = min (b[-1],c[-1],d[-1]) + 1
    if C == 300 :
        L = L/4
        m = m+0.05
        for i in range (1,300,1):
            if np.average(av[:,i])/256 < 20 :
                b.append(i)
                c.append(i)
                d.append(i)
                continue
            
            if np.sum(R_image[:,i]<L) > R_image.shape[0]*m:
                b.append(i)
            if np.sum(B_image[:,i]<L) > B_image.shape[0]*m:
                c.append(i)
            if np.sum(G_image[:,i]<L) > G_image.shape[0]*m:
                d.append(i)
        
        C = min (b[-1],c[-1],d[-1]) + 1
        L = L*4
        m = m + 0.05
    
    
    b = [0]
    c = [0]
    d = [0]
    for i in range (-1,-300,-1):
        if np.average(av[:,i])/256 < 20 :
            b.append(i)
            c.append(i)
            d.append(i)
            continue
            
        if np.sum(R_image[:,i]<L) > R_image.shape[0]*m:
            b.append(i)
        if np.sum(B_image[:,i]<L) > B_image.shape[0]*m:
            c.append(i)
        if np.sum(G_image[:,i]<L) > G_image.shape[0]*m:
            d.append(i)
    D =  max(b[-1],c[-1],d[-1]) - 1
    
    if D == -300:
        L = L/4
        m = m+0.05
        for i in range (-1,-300,-1):
            if np.average(av[:,i])/256 < 20 :
                b.append(i)
                c.append(i)
                d.append(i)
                continue
            
            if np.sum(R_image[:,i]<L) > R_image.shape[0]*m:
                b.append(i)
            if np.sum(B_image[:,i]<L) > B_image.shape[0]*m:
                c.append(i)
            if np.sum(G_image[:,i]<L) > G_image.shape[0]*m:
                d.append(i)
        D =  max(b[-1],c[-1],d[-1]) - 1
        L = L*4
        m = m - 0.05
    return [A,B,C,D]
    
import timeit

start = timeit.default_timer()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

path = "Amir.tiff"
img = mpimg.imread(path)
sl = np.int_(img.shape[0]/3)   #slice
b_img = img[:sl,:]
g_img = img[sl:sl*2,:]
r_img = img[sl*2:,:]

if r_img.shape[0] > g_img.shape[0]:
    r_img = r_img[:g_img.shape[0] - r_img.shape[0],:]

ind1 = IND(g_img,b_img)
ind2 = IND(r_img,b_img)

print ('green is rolling on blue with index of',[ind1[0],ind1[1]])
print ('red is rolling on blue with index of',[ind2[0],ind2[1]])


R_img = r_img
G_img = g_img
B_img = b_img

GG = np.roll(G_img,ind1,axis=[0,1])
RR = np.roll(R_img,ind2,axis=[0,1])
BB = B_img

k = cv2.merge((RR, GG, BB))
av = np.mean(k[:,:,:],axis = 2)
R = RR - av
B = BB - av
G = GG - av

[a,b,c,d] = ABCD(R,B,G,av)
print ('border deleted and the rows of new image is',[a,b],'of rolled image')
print ('border deleted and the column of new image is',[c,d],'of rolled image')


image_eq = cv2.merge((np.uint8(RR[a:b,c:d]/256), np.uint8(GG[a:b,c:d]/256), np.uint8(BB[a:b,c:d]/256)))
plt.imsave("res03-Amir.jpg",image_eq)


###########
########
####
##
#

path = "Mosque.tiff"
img = mpimg.imread(path)
sl = np.int_(img.shape[0]/3)   #slice
b_img = img[:sl,:]
g_img = img[sl:sl*2,:]
r_img = img[sl*2:,:]

if r_img.shape[0] > g_img.shape[0]:
    r_img = r_img[:g_img.shape[0] - r_img.shape[0],:]

ind1 = IND(g_img,b_img)
ind2 = IND(r_img,b_img)

print ('green is rolling on blue with index of',[ind1[0],ind1[1]])
print ('red is rolling on blue with index of',[ind2[0],ind2[1]])


R_img = r_img
G_img = g_img
B_img = b_img

GG = np.roll(G_img,ind1,axis=[0,1])
RR = np.roll(R_img,ind2,axis=[0,1])
BB = B_img

k = cv2.merge((RR, GG, BB))
av = np.mean(k[:,:,:],axis = 2)
R = RR - av
B = BB - av
G = GG - av

[a,b,c,d] = ABCD(R,B,G,av)
print ('border deleted and the rows of new image is',[a,b],'of rolled image')
print ('border deleted and the column of new image is',[c,d],'of rolled image')


image_eq = cv2.merge((np.uint8(RR[a:b,c:d]/256), np.uint8(GG[a:b,c:d]/256), np.uint8(BB[a:b,c:d]/256)))
plt.imsave("res04-Mosque.jpg",image_eq)


###########
########
####
##
#




path = "Train.tiff"
img = mpimg.imread(path)
sl = np.int_(img.shape[0]/3)   #slice
b_img = img[:sl,:]
g_img = img[sl:sl*2,:]
r_img = img[sl*2:,:]

if r_img.shape[0] > g_img.shape[0]:
    r_img = r_img[:g_img.shape[0] - r_img.shape[0],:]

ind1 = IND(g_img,b_img)
ind2 = IND(r_img,b_img)

print ('green is rolling on blue with index of',[ind1[0],ind1[1]])
print ('red is rolling on blue with index of',[ind2[0],ind2[1]])


R_img = r_img
G_img = g_img
B_img = b_img

GG = np.roll(G_img,ind1,axis=[0,1])
RR = np.roll(R_img,ind2,axis=[0,1])
BB = B_img

k = cv2.merge((RR, GG, BB))
av = np.mean(k[:,:,:],axis = 2)
R = RR - av
B = BB - av
G = GG - av

[a,b,c,d] = ABCD(R,B,G,av)
print ('border deleted and the rows of new image is',[a,b],'of rolled image')
print ('border deleted and the column of new image is',[c,d],'of rolled image')


image_eq = cv2.merge((np.uint8(RR[a:b,c:d]/256), np.uint8(GG[a:b,c:d]/256), np.uint8(BB[a:b,c:d]/256)))
plt.imsave("res05-Train.jpg",image_eq)


stop = timeit.default_timer()

print('Time: ', stop - start)  





