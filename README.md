# Multiview-Geometry-Depth-Estimation

## Use the link below to download the data
https://drive.google.com/drive/folders/15gBoik7dpRhlzmwqeSnyyZe9AnFKeioz?usp=sharing

## Instructions
Make sure to download the data required to run the code and save it in the same directory as that of the code.
The code will first rectify the images and ask for user input whether the epipolar lines are aligned or not.
After the user input the code takes upto 2-3 mins to generate the disparity and the depth map.

## Notes
### Calibration:
1) The matching feature points are found using SIFT.
2) The estimation of fundamental matrix is done as follows:
a) First the points are normalized such that the points have (2)**(0.5) as mean distance from the mean or centroid of the points.
b) After forming the A matrix the solution is computed by applying SVD to this A matrix which gives us the fundamental matrix.
c) To enforce the condition of rank(2) of the fundamental matrix we compute the SVD of the fundamental matrix and then set the smallest singular value to zero. The error occurs because of the error in the matching feature points.
d) Once the rank(2) constraint is done the fundamental matrix is rebuilt and then unnormalized.
3) The process in point 2 is done iteratively using the RANSAC function to obtain the fundamental matrix that is not bothered by any outliers in the matching feature points.
4) To obtain the rotation and translation matrix and vector we need to compute the essential matrix first.
5) The essential matrix is then decomposed to obtain 4 possible combinations of rotation and translation.
6) Then using the cheirality check we can eliminate the three other possibilities of rotation and translation. This is done by checking r3(X−C) > 0 where r3 is the third row of the rotation matrix, X is 3D point and C is the translation vector.
7) For each of the possible combinations of rotations and translations the combination in which we get maximum number of points satisfying the above condition is the final rotation and translation.
### Rectification:
1) In rectification the images are transformed such that the images are made parallel to each each other.
2) This is done in order to reduce the number of iterations required while matching each pixel from the left image to right image.
3) By rectifying the images, now for each point in the left image we need to only search one line of corresponding height in the right image.
To obtain the rectified images following is the procedure followed:
a) First the left image is translated such the center of the image is taken to the point (0,0,1) in homogeneous coordinates. (T1)
b) The same transformation is applied to the epipole.
c) After this translation, the image is then rotated so that the epipole now lies on the horizontal axis such that the new coordinates of the epipole are of the form (𝑒𝑥,0,1). (T2)
d) Now we only need to shift the resulting epipole to infinity and then the epipolar lines on first image become parallel to each other.(T3)
e) The resulting transformation is given by H2 = (T1)-1T3 T2 T1
f) To make the right image parallel to the left the required transformation is of the form H1 = HA H2 M, where M = [e]X F + evT where [e]X is a skew symmetric matrix, vT = [1,1,1] and e is epipole.
g) HA is found by computing the least square solution of the points obtained by the following transformation (points * (H1*M)) and (points * H1)
h) Thus the left and right images are rectified using the transformations H1 and H2 respectively.
### Correspondence:
1) Using SSD the pixels in left image are matched in the right image.
2) The disparity is calculated by the difference in image pixels along width.
