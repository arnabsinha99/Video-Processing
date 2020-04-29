#importing the necessary libraries
import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from Delaunay import *

ix,iy = -1,-1 #global coordinates for efficient access
def point(event,x,y,flags,param):
    global ix,iy #reference to global variables
    if event == cv2.EVENT_LBUTTONDOWN: #when left button of mouse is clicked
        ix,iy = y,x
    return x,y

#click a point, press enter, repeat for alternate image. Press 0 when you want to exit the clicking process.
def get_controlpoints(image1,image2):
    pointsrc = []
    pointdest = []
    while(1):
        cv2.namedWindow('image1')
        cv2.setMouseCallback('image1',point) #bind a callback to the mouse click so that a stimulus is set          
        cv2.imshow('image1',cv2.cvtColor(image1,cv2.COLOR_BGR2RGB))
        k = cv2.waitKey()
        if k==48: #if key entered is 0
            break
        cv2.circle(image1,(iy,ix), 3, (0, 0, 255), -1)
        cv2.imshow('image1',cv2.cvtColor(image1,cv2.COLOR_BGR2RGB))
        pointsrc.append((ix,iy)) #else append the points
        
        cv2.namedWindow('image2')
        cv2.setMouseCallback('image2',point) #bind a callback to the mouse click so that a stimulus is set
        cv2.imshow('image2',cv2.cvtColor(image2,cv2.COLOR_BGR2RGB))
        k = cv2.waitKey()
        cv2.circle(image2,(iy,ix), 3, (0, 0, 255), -1)
        cv2.imshow('image2',cv2.cvtColor(image2,cv2.COLOR_BGR2RGB))
        pointdest.append((ix,iy)) #else append the points
    cv2.destroyAllWindows()
    
    return pointsrc,pointdest

#roundoff decimal values
def roundoff(x):
    if (x-np.floor(x))<0.5:
        x = int(np.floor(x))
    else:
        x = int(np.ceil(x))
    return x

def inside_tri(tri,pnt):
    '''
    Returns true if point pnt is inside triangle tri
    Return False otherwise
    '''
    sign = 0
    for i in range(3) :
        pnt1 = tri[i]
        pnt2 = tri[(i+1)%3]
        lhs = (pnt[1]-pnt1[1])*(pnt2[0]-pnt1[0])
        rhs = (pnt[0]-pnt1[0])*(pnt2[1]-pnt1[1])
        sign += np.sign(lhs-rhs)
    if np.abs(sign) == 3:
        return True
    else :
        return False

#handling boundary pixels in case of error in input image fitting
def handleboundary(img_frame,rows,cols,k,FRAMES):
    #handle first row
    for i in range(cols):
        img_frame[0][i][0] = roundoff(((1-((k+1)/FRAMES))*im_1[0][i][0]) + (((k+1)/FRAMES)*im_2[0][i][0]))
        img_frame[0][i][1] = roundoff(((1-((k+1)/FRAMES))*im_1[0][i][1]) + (((k+1)/FRAMES)*im_2[0][i][1]))
        img_frame[0][i][2] = roundoff(((1-((k+1)/FRAMES))*im_1[0][i][2]) + (((k+1)/FRAMES)*im_2[0][i][2]))
        
    #handle last row
    for i in range(cols):
        img_frame[img_frame.shape[0]-1][i][0] = roundoff(((1-((k+1)/FRAMES))*im_1[img_frame.shape[0]-1][i][0]) + (((k+1)/FRAMES)*im_2[0][i][0]))
        img_frame[img_frame.shape[0]-1][i][1] = roundoff(((1-((k+1)/FRAMES))*im_1[img_frame.shape[0]-1][i][1]) + (((k+1)/FRAMES)*im_2[0][i][1]))
        img_frame[img_frame.shape[0]-1][i][2] = roundoff(((1-((k+1)/FRAMES))*im_1[img_frame.shape[0]-1][i][2]) + (((k+1)/FRAMES)*im_2[0][i][2]))
    
    #handle first column
    for i in range(rows):
        img_frame[i][0][0] = roundoff(((1-((k+1)/FRAMES))*im_1[i][0][0]) + (((k+1)/FRAMES)*im_2[i][0][0]))
        img_frame[i][0][1] = roundoff(((1-((k+1)/FRAMES))*im_1[i][0][1]) + (((k+1)/FRAMES)*im_2[i][0][1]))
        img_frame[i][0][2] = roundoff(((1-((k+1)/FRAMES))*im_1[i][0][2]) + (((k+1)/FRAMES)*im_2[i][0][2]))
    
    #handle last column
    for i in range(rows):
        img_frame[i][img_frame.shape[1]-1][0] = roundoff(((1-((k+1)/FRAMES))*im_1[i][img_frame.shape[1]-1][0]) + (((k+1)/FRAMES)*im_2[i][img_frame.shape[1]-1][0]))
        img_frame[i][img_frame.shape[1]-1][1] = roundoff(((1-((k+1)/FRAMES))*im_1[i][img_frame.shape[1]-1][1]) + (((k+1)/FRAMES)*im_2[i][img_frame.shape[1]-1][1]))
        img_frame[i][img_frame.shape[1]-1][2] = roundoff(((1-((k+1)/FRAMES))*im_1[i][img_frame.shape[1]-1][2]) + (((k+1)/FRAMES)*im_2[i][img_frame.shape[1]-1][2]))
    return img_frame

#get alpha and beta coefficients
def alphabeta(i,j,tri):
    p1 = tri[0]
    p2 = tri[1]
    p3 = tri[2]
    RHS = np.array([i-p1[0],j-p1[1]])
    LHS = np.array([ [ p2[0]-p1[0], p3[0]-p1[0] ],[ p2[1]-p1[1], p3[1]-p1[1] ] ])
    coeff = np.linalg.inv(LHS).dot(RHS)
    return coeff[0],coeff[1]  

#get the mapped points in source and destination image
def getpoints(tri,alp,beta):
    p1 = tri[0]
    p2 = tri[1]
    p3 = tri[2]
    pts = np.array([ [ p2[0]-p1[0], p3[0]-p1[0] ],[ p2[1]-p1[1], p3[1]-p1[1] ] ]).dot(np.array([alp,beta])) + np.array([p1[0],p1[1]])
    return int(pts[0]),int(pts[1]) 

#obtain the image
def getimage(name): 
    image = cv2.imread(name,1)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

#resize image to minimum dimensions 
def resizeimages(image1,image2):
    height = np.abs(image1.shape[0]-image2.shape[0])
    width = np.abs(image1.shape[1]-image2.shape[1])
    if image1.shape[0]>image2.shape[0]:
        if image1.shape[1]>image2.shape[1]:
            image1 = cv2.resize(image1,(image1.shape[1]-width,image1.shape[0]-height),interpolation = cv2.INTER_AREA)
        else:
            image1 = cv2.resize(image1,(image1.shape[1],image1.shape[0]-height),interpolation = cv2.INTER_AREA)
            image2 = cv2.resize(image2,(image2.shape[1]-width,image2.shape[0]),interpolation = cv2.INTER_AREA)
    else:
        if image2.shape[1]>image1.shape[1]:
            image2 = cv2.resize(image2,(image2.shape[1]-width,image2.shape[0]-height),interpolation = cv2.INTER_AREA)
        else:
            image1 = cv2.resize(image1,(image1.shape[1]-width,image1.shape[0]),interpolation = cv2.INTER_AREA)
            image2 = cv2.resize(image2,(image2.shape[1],image2.shape[0]-height),interpolation = cv2.INTER_AREA)    
    return image1,image2

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('im1', action="store")
    parser.add_argument('im2', action="store")
    parser.add_argument('-frames', action="store", default=5, type=int)

    results = parser.parse_args()

    im_1 = getimage(str(results.im1))
    im_2 = getimage(str(results.im2))
    FRAMES = results.frames

    #handling directory where the images will be saved
    if os.path.isdir("imgs"):
        for fl in os.listdir("imgs"):
            os.remove("imgs/" + fl)
    else:
        os.mkdir("imgs")

    im_1,im_2 = resizeimages(im_1,im_2)
    cv2.imwrite("imgs/"+str(0)+'.png',cv2.cvtColor(im_1,cv2.COLOR_BGR2RGB))
    print("Done till frame "+str(0))

    pt_src,pt_dest = get_controlpoints(im_1,im_2)
    #since the original images are edited because of cv2.circle() command. We ensure that the original image passes forward.
    im_1 = getimage(str(results.im1))
    im_2 = getimage(str(results.im2))

    ptsrc2name = {(0,0):"1",(0,im_1.shape[1]-1):"2",(im_1.shape[0]-1,0):"3",(im_1.shape[0]-1,im_1.shape[1]-1):"4"}
    count = 5
    for pt in pt_src:
        if pt not in ptsrc2name.keys():
            s = str(count)
            ptsrc2name[pt] = s
            count+=1
    ptname2src = dict(map(reversed, ptsrc2name.items()))

    ptdest2name = {(0,0):"1",(0,im_2.shape[1]-1):"2",(im_2.shape[0]-1,0):"3",(im_2.shape[0]-1,im_2.shape[1]-1):"4"}
    count = 5
    for pt in pt_dest:
        if pt not in ptdest2name.keys():
            s = str(count)
            ptdest2name[pt] = s
            count+=1
    ptname2dest = dict(map(reversed, ptdest2name.items()))

    de_src = Delaunay(im_1.shape)

    for pts in ptsrc2name.keys():
        if int(ptsrc2name[pts])>4:
            res = de_src.ins_point(pts)
            if res==False:
                print(pts)

    triangles_dest = []
    for tri in de_src.triangles:
        tri_coords = []
        for corners in tri:
            point = (corners[0],corners[1])
            name = ptsrc2name[point]
            tri_coords.append(ptname2dest[name])
        triangles_dest.append(tri_coords)
    triangles_dest = np.array(triangles_dest)

    #save triangulation
    plt_triangles(de_src.triangles)
    plt.imshow(im_1)
    plt.savefig('Source Triangulated.png',bbox_inches='tight', pad_inches=0)
    plt_triangles(triangles_dest)
    plt.imshow(im_2)
    plt.savefig('Destination Triangulated.png',bbox_inches='tight', pad_inches=0)


    #create the Delaunay triangulation for each frame
    frames_tri = []
    for frames in range(FRAMES):
        frames_tri.append(np.round((1-(frames+1)/FRAMES)*de_src.triangles+((frames+1)/FRAMES)*triangles_dest))


    frames_array = []
    for k in range(0,FRAMES):
        img_frame = np.zeros((im_1.shape[0],im_1.shape[1],3),dtype = np.uint8)
        #handling boundary cells
        img_frame = handleboundary(img_frame,img_frame.shape[0],img_frame.shape[1],k,FRAMES)

        #now handle other points
        l = []
        for i in range(1,img_frame.shape[0]-1):
            for j in range(1,img_frame.shape[1]-1):
                xy = (i,j)
                for ind,tri in enumerate(frames_tri[k]):
                    if inside_tri(tri,xy):
                        l.append(ind)
                        break 
                if len(l)==0:
                    print("Issue with point {},{}".format(i,j))
                #finding alpha and beta
                tri = frames_tri[k][ind]
                alpha,beta = alphabeta(i,j,tri)
                p_src = getpoints(de_src.triangles[ind],alpha,beta)
                p_dest = getpoints(triangles_dest[ind],alpha,beta)
                if p_dest[0]>=0 and p_dest[0]<=(im_2.shape[0]-1) and p_dest[1]>=0 and p_dest[1]<=(im_2.shape[1]-1) and p_src[0]>=0 and p_src[0]<=(im_1.shape[0]-1) and p_src[1]>=0 and p_src[1]<=(im_1.shape[1]-1):
                    img_frame[i][j][0] = roundoff(((1-((k+1)/FRAMES))*im_1[p_src[0]][p_src[1]][0]) + (((k+1)/FRAMES)*im_2[p_dest[0]][p_dest[1]][0]))
                    img_frame[i][j][1] = roundoff(((1-((k+1)/FRAMES))*im_1[p_src[0]][p_src[1]][1]) + (((k+1)/FRAMES)*im_2[p_dest[0]][p_dest[1]][1]))
                    img_frame[i][j][2] = roundoff(((1-((k+1)/FRAMES))*im_1[p_src[0]][p_src[1]][2]) + (((k+1)/FRAMES)*im_2[p_dest[0]][p_dest[1]][2]))
        print("Done till frame "+str(k+1))
        frames_array.append(img_frame)
        cv2.imwrite("imgs/"+str(k+1)+'.png',cv2.cvtColor(img_frame,cv2.COLOR_BGR2RGB))
    
    os.system("ffmpeg -framerate 40 -i ./imgs/%d.png -vb 20M output.mp4 -y")

