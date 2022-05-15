import numpy as np
from numpy import linalg
import scipy as scipy
import matplotlib.pyplot as plt
import matplotlib
import math
import imageio
from skimage.color import rgb2gray
from scipy import fftpack

# ! not used for the final paper :( 
def jpegCompress(img):
    A=imageio.imread(img)
    A=rgb2gray(A)
    A=np.double(A)

    # add a row/col of 0 to make the dimensions of the img multiples of 8
    tmp = np.zeros((A.shape[0] + 1,A.shape[1] + 1))
    tmp[:A.shape[0], :A.shape[1]] = A
    A = tmp

    # show original img
    plt.figure(2,dpi=150)
    plt.imshow(A)
    plt.set_cmap('gray')
    plt.show()

    dct_blocks = np.zeros(A.shape)
    # apply DCT transform to all blocks
    for i in range(0, A.shape[0], 8):
        for j in range(0, A.shape[1], 8):
            # perform 2D DCT
            dct_blocks[i:i+8,j:j+8] = scipy.fftpack.dct(
                scipy.fftpack.dct(A[i:i+8, j:j+8], axis=0), axis=1)

    # show DCT of img
    plt.figure(2,dpi=150)
    plt.imshow(dct_blocks)
    plt.set_cmap('gray')
    plt.show()

    # quantize with quantization mtx (JPEG standard)
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    quant_dct_blocks = np.zeros(A.shape)
    for i in range(0, A.shape[0], 8):
        for j in range(0, A.shape[1], 8):
            quant_dct_blocks[i:i+8, j:j+8] = np.round(dct_blocks[i:i+8, j:j+8] / Q)

    
    # decode to get original image
    decoded_img = np.zeros(A.shape)
    for i in range(0, A.shape[0], 8):
        for j in range(0, A.shape[1], 8):
            # decode with 2d inverse DCT
            decoded_img[i:i+8, j:j+8] = scipy.fftpack.idct(
                scipy.fftpack.idct(quant_dct_blocks[i:i+8, j:j+8] * Q,axis=0, norm='ortho' ), axis=1, norm='ortho' )

    plt.figure(2,dpi=150)
    plt.imshow(decoded_img)
    plt.set_cmap('gray')
    plt.show()

def svdCompress(img):
    # load in original image
    A=imageio.imread(img)
    Acol = np.double(A)
    A=rgb2gray(A)
    A=np.double(A)
    A=A-np.mean(A)

    # color vsn of cattle
    plt.figure(1,dpi=150)
    plt.imshow(Acol)
    plt.show()

     # gray vsn of cattle
    plt.figure(1,dpi=150)
    plt.imshow(A)
    plt.set_cmap('gray')
    plt.show()

    # get the SVD of the image matrix, A
    U,Sigma,V=np.linalg.svd(A)
    print("The amount of non-zero singular values is:", len(np.nonzero(Sigma)[0]))

    # use 5 singular values to reconstruct the image
    A_5 = U[:, :5] @ np.diag(Sigma[:5]) @ V[:5,:]
    plt.figure(1,dpi=150)
    plt.imshow(A_5)
    plt.set_cmap('gray')
    plt.title("k = 5")
    plt.show()

    # use 30 singular values to reconstruct the image
    A_30 = U[:, :30] @ np.diag(Sigma[:30]) @ V[:30,:]
    plt.figure(1,dpi=150)
    plt.imshow(A_30)
    plt.set_cmap('gray')
    plt.title("k = 30")
    plt.show()

    # use 60 singular values to reconstruct the image
    A_60 = U[:, :60] @ np.diag(Sigma[:60]) @ V[:60,:]
    plt.figure(1,dpi=150)
    plt.imshow(A_60)
    plt.set_cmap('gray')
    plt.title("k = 60")
    plt.show()

    # use 120 singular values to reconstruct the image
    A_120 = U[:, :120] @ np.diag(Sigma[:120]) @ V[:120,:]
    plt.figure(1,dpi=150)
    plt.imshow(A_120)
    plt.set_cmap('gray')
    plt.title("k = 120")
    plt.show()

    print("explained variance - k = 5", sum(np.square(Sigma[:5])/ sum(np.square(Sigma))))
    print("explained variance - k = 30", sum(np.square(Sigma[:30])/ sum(np.square(Sigma))))
    print("explained variance - k = 60", sum(np.square(Sigma[:60])/ sum(np.square(Sigma))))
    print("explained variance - k = 120", sum(np.square(Sigma[:120])/ sum(np.square(Sigma))))

def main():
    img = "cow.jpg"
    print("BEGINNING SVD COMPRESSION")
    svdCompress(img)

    # ! not used for this project
    # print("BEGINNING JPEG COMPRESSION")
    # jpegCompress(img)
main()