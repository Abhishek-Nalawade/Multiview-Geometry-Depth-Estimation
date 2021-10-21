import numpy as np
import cv2
import matplotlib.pyplot as plt
import random


#####groundtruth for dataset 1
cam0 = np.array([[5299.313, 0, 1263.818], [0, 5299.313, 977.763], [0, 0, 1]])
cam1 = np.array([[5299.313, 0, 1438.004], [0, 5299.313, 977.763], [0, 0, 1]])
doffs=174.186
baseline=177.288
width=2988
height=2008
ndisp=180
isint=0
vmin=54
vmax=147
dyavg=0
dymax=0

#####groundtruth for dataset 2
'''cam0=np.array([[4396.869, 0, 1353.072], [0, 4396.869, 989.702], [0, 0, 1]])
cam1=np.array([[4396.869, 0, 1538.86], [0, 4396.869, 989.702], [0, 0, 1]])
doffs=185.788
baseline=144.049
width=2880
height=1980
ndisp=640
isint=0
vmin=17
vmax=619
dyavg=0
dymax=0'''

######groundtruth or dataset 3
'''cam0=np.array([[5806.559, 0, 1429.219], [0, 5806.559, 993.403], [0, 0, 1]])
cam1=np.array([[5806.559, 0, 1543.51], [0, 5806.559, 993.403], [0, 0, 1]])
doffs=114.291
baseline=174.019
width=2960
height=2016
ndisp=250
isint=0
vmin=38
vmax=222
dyavg=0
dymax=0'''



def estimate_fundamental(pts1, pts2):
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    #normalizing the points so to have (2)**(0.5) as mean distance from the mean of the points
    mean1 = np.sum(pts1, axis = 0)/pts1.shape[0]
    mean2 = np.sum(pts2, axis = 0)/pts2.shape[0]
    mean1 = np.reshape(mean1, (1,mean1.shape[0]))
    mean2 = np.reshape(mean2, (1,mean2.shape[0]))
    translated_pts1 = pts1 - mean1
    translated_pts2 = pts2 - mean2

    sum_sq1 = np.sum(translated_pts1**2, axis = 1)

    sum_sq1 = np.reshape(sum_sq1, (translated_pts1.shape[0],1))

    mean_distance1 = np.sum(sum_sq1**(1/2), axis = 0)/translated_pts1.shape[0]
    scale_factor1 = 2**(1/2)/mean_distance1[0]

    sum_sq2 = np.sum(translated_pts2**2, axis = 1)
    sum_sq2 = np.reshape(sum_sq2, (translated_pts2.shape[0],1))
    mean_distance2 = np.sum(sum_sq2**(1/2), axis = 0)/translated_pts2.shape[0]
    scale_factor2 = 2**(1/2)/mean_distance2[0]

    #these are the final normalized points
    normalized_pts1 = scale_factor1 * translated_pts1
    normalized_pts2 = scale_factor2 * translated_pts2

    #forming the translation and scaling matrices so to obtain the unnormalized fundamental matrix
    translation_mat1 = np.array([[1,0,-mean1[0][0]],[0,1,-mean1[0][1]],[0,0,1]])
    scale_mat1 = np.array([[scale_factor1,0,0],[0,scale_factor1,0],[0,0,1]])

    translation_mat2 = np.array([[1,0,-mean2[0][0]],[0,1,-mean2[0][1]],[0,0,1]])
    scale_mat2 = np.array([[scale_factor2,0,0],[0,scale_factor2,0],[0,0,1]])

    T1 = np.dot(scale_mat1, translation_mat1)
    T2 = np.dot(scale_mat2, translation_mat2)

    #froming the A matrix to solve using SVD
    A = np.zeros((pts1.shape[0],9))
    for i in range(pts1.shape[0]):
        A[i,:] = [normalized_pts2[i][0]*normalized_pts1[i][0], normalized_pts2[i][0]*normalized_pts1[i][1], normalized_pts2[i][0], normalized_pts2[i][1]*normalized_pts1[i][0], normalized_pts2[i][1]*normalized_pts1[i][1], normalized_pts2[i][1], normalized_pts1[i][0], normalized_pts1[i][1], 1]

    U, S, Vt = np.linalg.svd(A)

    V = Vt.T
    V = V[:,-1]
    F = np.zeros((3,3))
    count = 0
    for i in range(3):
        for j in range(3):
            F[i,j] = V[count]
            count += 1

    #computing SVD of F to enforce the rank(2) contraint by setting the smallest singular value to zero
    u, s, vt = np.linalg.svd(F)

    s[-1] = 0
    s_new = np.zeros((3,3))
    for i in range(3):
        s_new[i,i] = s[i]

    F_new = np.dot((np.dot(u,s_new)),vt)

    unnormalized_F = np.dot(np.dot(T2.T, F_new),T1)

    unnormalized_F = unnormalized_F/unnormalized_F[-1,-1]

    return unnormalized_F


#computes the ransac soltuion for the fundamental matrix
def RANSAC(feat_1,feat_2):
    threshold =0.05
    max_inliers= 0
    F_final = []
    p = 0.99
    N = 2000
    count = 0
    while N > count:
        no_of_inlier= 0
        points8_1 = []
        points8_2 = []
        #generate a set of random 8 points
        random_list = np.random.randint(len(feat_1), size = 8)
        for k in random_list:
            points8_1.append(feat_1[k])
            points8_2.append(feat_2[k])
        #Perform 8-point Hartley algorithm to determine F using the generated 8 random points
        F = estimate_fundamental(points8_1, points8_2)
        ones = np.ones((len(feat_1),1))
        x1 = np.hstack((feat_1,ones))
        x2 = np.hstack((feat_2,ones))
        e1, e2 = x1 @ F.T, x2 @ F
        error = np.sum(e2* x1, axis = 1, keepdims=True)**2 / np.sum(np.hstack((e1[:, :-1],e2[:,:-1]))**2, axis = 1, keepdims=True)
        inliers = error<=threshold
        #print(error)
        no_of_inlier = np.sum(inliers)
        #print(no_of_inlier)
        #Record the best F
        if max_inliers <  no_of_inlier:
            max_inliers = no_of_inlier
            #print(inliers)
            coor = np.where(inliers == True)
            #print("length ",len(coor[0]))
            #print("inliers ",max_inliers)
            #print(coor)
            x1_ar = np.array(feat_1)
            x2_ar = np.array(feat_2)
            #print(x1_ar.shape)
            inlier_x1 = x1_ar[coor[0][:]]
            inlier_x2 = x2_ar[coor[0][:]]
            #print("inlier shape ",inlier_x2.shape)
            #print(error)

            F_final = F
        #Iterations to run the RANSAC for every frame
        inlier_ratio = no_of_inlier/len(feat_1)
        if np.log(1-(inlier_ratio**8)) == 0:
            continue
        N = np.log(1-p)/np.log(1-(inlier_ratio**8))
        count += 1
    return F_final, inlier_x1, inlier_x2


#computes the essential matrix
def estimate_essential(F):
    E = np.dot(np.dot(cam1.T,F),cam0)
    U,S,Vt = np.linalg.svd(E)
    new_S = np.zeros((3,3))
    for i in range(3):
        new_S[i,i] = 1
    new_S[-1,-1] = 0
    new_E = np.dot(np.dot(U,new_S),Vt)
    return new_E

#computes the rotation and translation matrix
def rotation_and_translation(E):

    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    U, S, Vt = np.linalg.svd(E)

    #print(np.dot(translation,rotation))

    #four possible cases of rotation and translation
    R1 = np.dot(np.dot(U,W),Vt)
    C1 = U[:,2]
    if np.linalg.det(R1) < 0:
        R1 = -R1
        C1 = -C1
    R2 = np.dot(np.dot(U,W),Vt)
    C2 = -U[:,2]
    if np.linalg.det(R2) < 0:
        R2 = -R2
        C2 = -C2
    R3 = np.dot(np.dot(U,W.T),Vt)
    C3 = U[:,2]
    if np.linalg.det(R3) < 0:
        R3 = -R3
        C3 = -C3
    R4 = np.dot(np.dot(U,W.T),Vt)
    C4 = -U[:,2]
    if np.linalg.det(R4) < 0:
        R4 = -R4
        C4 = -C4

    C1 = np.reshape(C1,(3,1))
    C2 = np.reshape(C2,(3,1))
    C3 = np.reshape(C3,(3,1))
    C4 = np.reshape(C4,(3,1))

    rotation_mats = [R1,R2,R3,R4]
    translation_mats = [C1,C2,C3,C4]
    return rotation_mats, translation_mats

#estimating the 3D points to check the cheirality condition which is required for rectification
def point_3D(R2,C2, pts1,pts2):
    C1 = np.array([[0],[0],[0]])
    R1 = np.identity(3)
    R2 = R2.T
    R1_C1 = -np.dot(R1,C1)
    R2_C2 = -np.dot(R2,C2)

    conc1 = np.concatenate((R1, R1_C1), axis =1)
    conc2 = np.concatenate((R2, R2_C2), axis =1)

    P1 = np.dot(cam0,conc1)
    P2 = np.dot(cam1,conc2)

    X = list()
    for i in range(len(pts1)):
        x1 = np.array(pts1[i])
        x2 = np.array(pts2[i])

        #print(x1.shape)
        x1 = np.reshape(x1,(2,1))
        b = np.array([1])
        b = np.reshape(b,(1,1))
        x1 = np.concatenate((x1,b), axis = 0)
        x2 = np.reshape(x2,(2,1))
        x2 = np.concatenate((x2,b), axis = 0)
        #print(x1)
        #print(x2)
        x1_skew = np.array([[0,-x1[2][0],x1[1][0]],[x1[2][0], 0, -x1[0][0]],[-x1[1][0], x1[0][0], 0]])
        x2_skew = np.array([[0,-x2[2][0],x2[1][0]],[x2[2][0], 0, -x2[0][0]],[-x2[1][0], x2[0][0], 0]])
        A1 = np.dot(x1_skew, P1)
        A2 = np.dot(x2_skew, P2)
        #print("A1 ",A1)
        #print("A2 ",A2)
        A = np.zeros((6,4))
        for i in range(6):
            if i<=2:
                A[i,:] = A1[i,:]
            else:
                A[i,:] = A2[i-3,:]
        #print(A)
        U,S,Vt = np.linalg.svd(A)
        Vt = Vt[3]
        Vt = Vt/Vt[-1]
        X.append(Vt)
    return np.array(X)

#checking the cheirality condition to eliminate the other possibilities of rotation and translation
def cheirality_check(Rs,Cs,pts1,pts2):
    countli = list()
    for i in range(4):
        Z = point_3D(Rs[i],Cs[i],pts1,pts2)
        count = 0
        for j in range(Z.shape[0]):
            #print(Rs[i])
            #print(Cs[i])
            coor = Z[j,:].reshape(-1,1)
            if np.dot(Rs[i][2], (coor[0:3] - Cs[i])) > 0 and coor[2]>0:
                count += 1
        countli.append(count)
    #print(countli)
    idx = countli.index(max(countli))
    #print(idx)
    if Cs[idx][2]>0:
        Cs[idx] = -Cs[idx]
    #print(Rs[idx])
    #print(Cs[idx])
    return Rs[idx], Cs[idx]


def LeastSq(x11, x22_dash):
    li = list()

    #forming the X matrix
    X = x11
    Y = np.reshape(x22_dash, (x22_dash.shape[0], 1))

    #computing B matrix as (X'X)^-1 (X'Y)
    ds = np.dot(X.T, X)
    ab = np.linalg.inv(ds)
    df = np.dot(X.T, Y)
    B = np.dot(ab, df)

    #computing the y coordinates and forming a list to return
    newy = np.dot(X, B)
    for i in newy:
        for a in i:
            li.append(a)

    return B

#rectifies the image
def rectification(R, t, F, pts1, pts2):

    #getting the index of the singular value = 0
    u,s,vt = np.linalg.svd(F)
    sing = np.where(s < 0.00000001)

    #finding the epipoles of both the images
    v = vt.T
    el = v[:,sing[0][0]]
    er = u[:,sing[0][0]]
    el = np.reshape(el,(el.shape[0],1))
    er = np.reshape(er,(er.shape[0],1))

    #translating the second image center at (0,0,1) in homogeneous coordinates
    T1 = np.array([[1,0,-(640/2)],[0,1,-(480/2)],[0,0,1]])
    e = np.dot(T1,er)
    e = e[:,:]/e[-1,:]
    len = ((e[0][0])**(2)+(e[1][0])**(2))**(1/2)
    #print(theta)
    if e[0][0] >= 0:
        alpha = 1
    else:
        alpha = -1

    #applying rotation to the translated image to place the epipole on horizontal axis
    T2 = np.array([[(alpha*e[0][0])/len, (alpha*e[1][0])/len, 0],[-(alpha*e[1][0])/len, (alpha*e[0][0])/len, 0],[0, 0, 1]])
    e = np.dot(T2,e)


    #bringing the epipole to infinity
    T3 = np.array([[1, 0, 0],[0, 1, 0],[((-1)/e[0][0]), 0, 1]])
    e = np.dot(T3,e)
    phi2 = np.dot(np.dot(np.linalg.inv(T1),T3),np.dot(T2,T1))
    #print(phi2.shape)

    gt = np.array([1,1,1])
    gt = np.reshape(gt,(1,3))
    ex = np.array([[0,-el[2][0],el[1][0]],[el[2][0],0,-el[0][0]],[-el[1][0],el[0][0],0]])
    M = np.dot(ex,F) + np.dot(el,gt)

    hom = np.dot(phi2,M)
    b = np.ones((pts1.shape[0],1))
    pts1 = np.concatenate((pts1,b), axis = 1)
    pts2 = np.concatenate((pts2,b), axis = 1)
    x11 = np.dot(hom,pts1.T)
    x11 = x11[:,:]/x11[2,:]
    x11 = x11.T
    x22 = np.dot(phi2,pts2.T)
    x22 = x22[:,:]/x22[2,:]
    x22 = x22.T
    x22_dash = np.reshape(x22[:,0], (x22.shape[0],1))
    #m = np.linalg.lstsq(x11, x22_dash, rcond=None)[0]

    #least square solution to find the homography matrix
    my = LeastSq(x11,x22_dash)
    Ha = np.array([[my[0][0],my[1][0],my[2][0]],[0,1,0],[0,0,1]])
    phi1 = np.dot(np.dot(Ha,phi2),M)


    return phi1, phi2

#draws the epipolar lines
def drawlines(img1,img2,lines,pts1,pts2):

    sh = img1.shape
    r = sh[0]
    c = sh[1]
    #img1 = cv2.cvtColor(img1,cv.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        pt1 = [int(pt1[0]),int(pt1[1])]
        pt2 = [int(pt2[0]),int(pt2[1])]
        #print("pt1 ",pt1)
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),2,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),2,color,-1)
    return img1,img2


#reads the images and returns the corresponding mathcing points using SIFT
def sift_match():
    img1 = cv2.imread('im0.png')
    img2 = cv2.imread('im1.png')
    img1 = cv2.resize(img1, (640,480))
    img2 = cv2.resize(img2, (640,480))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)


    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20], img2 ,flags=2)
    pts1 = list()
    pts2 = list()
    for m in matches:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

    return pts1, pts2, img1, img2


#compares the pixel values
def checking_pixels_diff(rgb1, rgb22):

    if rgb1.shape != rgb22.shape:
        return -1

    diff = np.sum(abs(rgb1 - rgb22))
    return diff


#compares the windows of two images
def compare_window(y, x, b_left, right_window, block_size=5):

    #search window dimensions
    x_min = max(0, x - restricting_window_in_image)
    x_max = min(right_window.shape[1], x + restricting_window_in_image)

    check1 = 1
    min_sad = None
    min_index = None
    for x in range(x_min, x_max):
        block_right = right_window[y: y+block_size,x: x+block_size]
        sad = checking_pixels_diff(b_left, block_right)

        if check1 == 1:
            min_sad = sad
            min_index = (y, x)
            check1 = 2
        else:
            if sad < min_sad:
                min_sad = sad
                min_index = (y, x)

    return min_index




def find_disparity(imh1,imh2):

    la = np.array(imh1)
    ra = np.array(imh2)
    la = la.astype(int)
    ra = ra.astype(int)

    #print(la.shape)
    h, w , g = la.shape

    disparity_map = np.zeros((h, w))

    # iterating through pixels
    for y in range(window_size, h-window_size):
        for x in range(window_size, w-window_size):
            block_left = la[y:y + window_size,x:x + window_size]
            min_index = compare_window(y, x, block_left, ra, block_size=window_size)
            disparity_map[y, x] = abs(min_index[1] - x)


    #print(disparity_map)

    return disparity_map

#finds the depth map of the image from the disparity
def depth(img_map):
    depth_img = img_map.copy()
    depth_img1 = img_map.copy()
    print("depth shape ",depth_img.shape)
    depth_img[depth_img[:,:]==0] = 1
    depth_img1[:,:] = ((baseline*cam0[0][0])/depth_img)
    return depth_img1

#main function that calls functions required to compute step 1 and 2 from assignment
def step_1_2():
    points1, points2, image1, image2 = sift_match()

    image1_rect = image1.copy()
    image2_rect = image2.copy()

    Fg, good_x1, good_x2 = RANSAC(points1,points2)

    points1 = np.array(points1)
    points2 = np.array(points2)
    Fcv, mask = cv2.findFundamentalMat(points1,points2,cv2.FM_LMEDS)

    essential_mat = estimate_essential(Fg)

    rotations, translations = rotation_and_translation(essential_mat)
    rotation, translation = cheirality_check(rotations,translations, points1,points2)
    print("The rotation matrix is: \n", rotation)
    print("The translation vector is: \n",translation)

    lines1 = cv2.computeCorrespondEpilines(good_x2.reshape(-1,1,2), 2,Fg)
    lines1 = lines1.reshape(-1,3)
    lines2 = cv2.computeCorrespondEpilines(good_x1.reshape(-1,1,2), 1,Fg)
    lines2 = lines2.reshape(-1,3)

    image1, image2 = drawlines(image1,image2,lines1,good_x1,good_x2)
    image1, image2 = drawlines(image2,image1,lines2,good_x1,good_x2)
    #print(H0)
    #print(image11)

    onet = np.ones((good_x1.shape[0],1))
    good_x1 = np.concatenate((good_x1,onet),axis = 1)
    good_x2 = np.concatenate((good_x2,onet),axis = 1)
    #print(good_x1.shape)

    H0, H1 = rectification(rotation,translation,Fg,points1,points2)
    print("Homography for first image: \n", H0)
    print("Homography for second image: \n", H1)

    image11 = cv2.warpPerspective(image1, H0, (640,480))
    image12 = cv2.warpPerspective(image2, H1, (640,480))
    #image12 = cv2.resize(image12,(640,480))

    image1_rect = cv2.warpPerspective(image1_rect, H0, (640,480))
    image2_rect = cv2.warpPerspective(image2_rect, H1, (640,480))

    cv2.imshow("im0",image11)
    cv2.imshow("im1",image12)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image1_rect, image2_rect

#main function that calls functions required to compute step 3 and 4 from assignment
def step_3_4(imge1,imge2):
    image11 = imge1.copy()
    image12 = imge2.copy()
    dispar = find_disparity(image11,image12)
    plt.imshow(dispar, cmap='hot', interpolation='nearest')
    plt.title("Diparity heat map")
    plt.savefig('disparity_image_heat.png')
    plt.show()
    plt.imshow(dispar, cmap='gray', interpolation='nearest')
    plt.title("Diparity gray map")
    plt.savefig('disparity_image_gray.png')
    plt.show()

    depth_map = depth(dispar)
    plt.imshow(depth_map, cmap='hot', interpolation='nearest')
    plt.title("Depth heat map")
    plt.savefig('depth_image_heat.png')
    plt.show()
    plt.imshow(depth_map, cmap='gray', interpolation='nearest')
    plt.title("Depth gray map")
    plt.savefig('depth_image_gray.png')
    plt.show()
    #cv2.imshow("rect1",image1_rect)
    #cv2.imshow("rect2",image1_rect)

#main body of the code
window_size = 7
restricting_window_in_image = 56

while True:
    im1, im2 = step_1_2()
    inp = input("\n Are the epipolar lines aligned and straight? \n Enter y or n: ")
    if inp =='y':
        step_3_4(im1, im2)
        break
    elif inp == 'n':
        continue
    else:
        print("Entered invalid input please run the code again! \n")
        break
