import numpy as np
import matplotlib.pyplot as plt

class Delaunay:
    def __init__(self,shp):
        dim = [shp[0]-1,shp[1]-1]
        self.triangles = np.array([[[0,0],[0,dim[1]],[dim[0],0]],
                                   [[dim[0],0],[0,dim[1]],[dim[0],dim[1]]]])
        
    def inside_tri(self,tri,pnt):
        """
        tri -> Triangle to be checked
        pnt -> Point to be checked
        Return True if point in Triangle else returns false
        """
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
    
    def find_quad(self,tri1,tri2):
        """
        tri1 -> Triangle 1
        tri2 -> Triangle 2
        Returns True if they form quadrilateral with switing condition else returns False
        """
        tri1s = set()
        for pnt in tri1:
            tri1s.add((pnt[0],pnt[1]))
        tri2s = set()
        for pnt in tri2:
            tri2s.add((pnt[0],pnt[1]))
        inter = tri1s.intersection(tri2s)
        if len(inter) == 2:
            
            p1,p2 = list(inter)
            p3 = list(tri1s.difference(inter))[0]
            pp = list(tri2s.difference(inter))[0]

            d12 = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            d13 = np.sqrt((p1[0]-p3[0])**2 + (p1[1]-p3[1])**2)
            d23 = np.sqrt((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)
            d1p = np.sqrt((p1[0]-pp[0])**2 + (p1[1]-pp[1])**2)
            d2p = np.sqrt((p2[0]-pp[0])**2 + (p2[1]-pp[1])**2)
            
            ang1 = np.arccos((d13**2+d23**2-d12**2)/(2*d13*d23))
            ang2 = np.arccos((d1p**2+d2p**2-d12**2)/(2*d1p*d2p))
            
            if (ang1+ang2)>np.pi:
                return True
            else:
                return False
        else:
            return False 
    
    def get_new_tri(self,tri1,tri2):
        """
        tri1 -> Triangle 1
        tri2 -> Triangle 2
        Returns New set of tringles formed when we change the diagnal of the Quadrilateral
        """
        tri1s = set()
        for pnt in tri1:
            tri1s.add((pnt[0],pnt[1]))
        tri2s = set()
        for pnt in tri2:
            tri2s.add((pnt[0],pnt[1]))
        inter = tri1s.intersection(tri2s)
        p1,p2 = list(inter)
        p3 = list(tri1s.difference(inter))[0]
        p4 = list(tri2s.difference(inter))[0]
        new_set = np.array([[[p1[0],p1[1]],[p3[0],p3[1]],[p4[0],p4[1]]],
                            [[p2[0],p2[1]],[p3[0],p3[1]],[p4[0],p4[1]]]])
        return new_set
    
    def ins_point(self,point):
        """
        point -> Point to be inserted
        Returns True if the insert was successful else returns False
        """
        ins = 0
        for i in range((self.triangles).shape[0]):
            tri = self.triangles[i]
            if self.inside_tri(tri,point):
                ins = 1
                new_tri = np.array([[list(tri[0]),list(tri[(1)%3]),[point[0],point[1]]],
                                    [list(tri[1]),list(tri[(2)%3]),[point[0],point[1]]],
                                    [list(tri[2]),list(tri[(0)%3]),[point[0],point[1]]]])
                self.triangles = np.delete(self.triangles,i,axis = 0)
                flg = 1
                while flg == 1:
                    flg = 0
                    for tr1i in range(self.triangles.shape[0]):
                        tr1 = self.triangles[tr1i]
                        flgin = 1
                        for tr2i in range(new_tri.shape[0]):
                            tr2 = new_tri[tr2i]
                            if self.find_quad(tr1,tr2):
                                flg = 1
                                self.triangles = np.delete(self.triangles,tr1i,axis = 0)
                                new_tri = np.delete(new_tri,tr2i,axis = 0)
                                new_tri = np.append(new_tri,self.get_new_tri(tr1,tr2),axis=0)
                                flgin = 0
                                break
                        if flgin==0:
                            break
                self.triangles = np.append(self.triangles,new_tri,axis = 0)
                break
        if ins==1:
            return True
        else :
            print("Their was error in the points given to Delaunay")
            return False

        
def plt_triangles(tri_list):
    """
    Helper function to plot all the triangles formed by Delaunay
    """
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    for tri in tri_list:
        p1 = (tri[0][1],tri[0][0])
        p2 = (tri[1][1],tri[1][0])
        p3 = (tri[2][1],tri[2][0])
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]])
        plt.plot([p2[0],p3[0]],[p2[1],p3[1]])
        plt.plot([p1[0],p3[0]],[p1[1],p3[1]])

