# As we need to find a transformation matrix  "XR-Y" from English to French vector space embeddings.
# Such a transformation matrix is nothing else but a matrix that rotates and scales vector spaces.
# There are three main vector transformations:
# Scaling
# Translation
# Rotation
import numpy as np                     # Import numpy for array manipulation
import matplotlib.pyplot as plt        # Import matplotlib for charts
from utils_nb import plot_vectors

# Create a 2 x 2 matrix
R = np.array([[2, 0],
              [0, -2]])

x = np.array([[1, 1]])
#The dot product between a vector and a square matrix produces a rotation and a scaling of the original vector.
y = np.dot(x, R) # Apply the dot product between x and R

plot_vectors([x], axes=[4, 4], fname='transform_x.svg')
plot_vectors([x, y], axes=[4, 4], fname='transformx_and_y.svg')
angle = 100 * (np.pi / 180) #convert degrees to radians

Ro = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle), np.cos(angle)]])

x2 = np.array([2, 2]).reshape(1, -1) # make it a row vector
y2 = np.dot(x2, Ro)

print('Rotation matrix')
print(Ro)
print('\nRotated vector')
print(y2)

print('\n x2 norm', np.linalg.norm(x2))
print('\n y2 norm', np.linalg.norm(y2))
print('\n Rotation matrix norm', np.linalg.norm(Ro))

plot_vectors([x2, y2], fname='transform_02.svg')
#The norm of the input vector is the same as the norm of the output vector. Rotations matrices do not modify the norm of the vector, only its direction.
# Frobenius form is the generalizwtion to R^2 of the alreadyknownn norm function for vectors

A = np.array([[2, 2],
              [2, 2]])
A_squared = np.square(A)
A_Frobenius = np.sqrt(np.sum(A_squared))
print('Frobenius norm of the Rotation matrix')
print(np.sqrt(np.sum(Ro * Ro)), '== ', np.linalg.norm(Ro))
































