
from __future__ import print_function
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
import numpy as np
import random
import cv2 as cv

# https://github.com/opencv/opencv/blob/master/modules/features2d/src/draw.cpp
# https://docs.opencv.org/trunk/d2/d29/classcv_1_1KeyPoint.html
# https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/peak.py

'''
Normaliza los valores de la imagen, poniendo el valor maximo a 1.0 y el mínimo a 0.0.
'''
def normalize(im):
    max_val = np.amax(im)
    min_val = np.amin(im)
    im = (im - min_val) / (max_val - min_val)
    return im

'''
Pinta una imagen psada por parámetro, el título por defecto será "Imagen". Y sus valores serán normalizados salvo que se indique lo contrario.
'''
def pintaI(im, titulo="Imagen", norm = True):

    if (norm):
        img = normalize(im)
    else:
        img = im

    plt.axis("off")
    plt.title(titulo)
    plt.rcParams['figure.figsize'] = [15, 15]

    if isBW(im):
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    cv.waitKey(0)
    plt.show()

'''
Compone imágenes de forma horizontal. Para ello crea bordes blancos en aquellas que tengan menor altura.
Se peude elegir el color del borde.
'''
def hstack(vim, border=(1,1,1), norm = True):

    aux = []
    # Calculamos la altura maxima
    altura = max(im.shape[0] for im in vim)

    for i,im in enumerate(vim):
        aux.append(np.copy(im))

        if(norm):
            aux[i] = normalize(im)

        if isBW(im):
            aux[i] = cv.cvtColor(im, cv.COLOR_GRAY2BGR)

        # Si no llega a la altura máxima, añadimos un borde blanco.
        if im.shape[0] < altura:
            aux[i] = cv.copyMakeBorder(
                aux[i], 0, altura - vim[i].shape[0],
                0, 0, cv.BORDER_CONSTANT, value = border
            )
    return np.hstack(aux)

'''
Utiliza la función anterior para pintar varias imágenes juntas.
'''
def pintaMI(vim, titulo='Imagenes', border=(1,1,1), norm = True):
    pintaI(hstack(vim, border, norm), titulo, norm)

'''
Esta función hace lo mismo que hstack pero en vertical.
'''
def vstack(vim, border=(1,1,1), norm = True):

    aux = []
    anchura = max(im.shape[1] for im in vim)

    for i,im in enumerate(vim):
        aux.append(np.copy(im))
        if(norm):
            aux[i] = normalize(im)

        if isBW(im):
            aux[i] = cv.cvtColor(im, cv.COLOR_GRAY2BGR)


        if im.shape[1] < anchura:
            aux[i] = cv.copyMakeBorder(
                aux[i], 0, 0,
                0, anchura - aux[i].shape[1], cv.BORDER_CONSTANT, value = border
            )
    return np.vstack(aux)

'''
Transforma una imagen de codificación BGR a RGB, realizando una llamada a cv.cvtColor
'''
def BGRtoRGB(im):
    return cv.cvtColor(im, cv.COLOR_BGR2RGB)

'''
Comprueba si una imagen es monobanda.
'''
def isBW(im):
    return len(im.shape) == 2

def savei(im, file, titulo = "", norm=True):

    if (norm):
        img = normalize(im)
    else:
        img = im

    if isBW(im):
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    plt.title(titulo)
    plt.axis("off")
    plt.savefig('samples/' + file, bbox_inches='tight')

'''
Lee una imagen dado un nombre de archivo "filename" y un flag que indica si la imagen es a color o no "flagColor".
Pasamos los valores a f32 ya que hay funciones de opencv como cvtcolor que no admiten mas precisión.
'''
def leeimagen(filename, flagColor):
    if flagColor == 1:
        return BGRtoRGB(cv.imread(filename, flagColor)).astype("float32")
    return cv.imread(filename, flagColor).astype("float32")


PATH = "imagenes/"
yos1 = leeimagen(PATH + "yosemite_full/yosemite1.jpg", 1).astype("uint8")
yos2 = leeimagen(PATH + "yosemite_full/yosemite2.jpg", 1).astype("uint8")
yos3 = leeimagen(PATH + "yosemite_full/yosemite3.jpg", 1).astype("uint8")
yos4 = leeimagen(PATH + "yosemite_full/yosemite4.jpg", 1).astype("uint8")
yos5 = leeimagen(PATH + "yosemite_full/yosemite5.jpg", 1).astype("uint8")
yos6 = leeimagen(PATH + "yosemite_full/yosemite6.jpg", 1).astype("uint8")
yos7 = leeimagen(PATH + "yosemite_full/yosemite7.jpg", 1).astype("uint8")
tab1 = leeimagen(PATH + "Tablero1.jpg", 1).astype("uint8")

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

"""
Aplica un alisado y un reescalado a la mitad en ambos ejes a una imagen.
"""
def blur_and_downsample(img, s, border = 2):
    if isBW(img):
         return cv.GaussianBlur(img, ksize = (0,0), sigmaX = s)[::2,::2]
    cv.GaussianBlur(img, ksize = (0,0), sigmaX = s)[::2,::2,:]

'''
Construye la piramide gaussiana de una imagen.
Almacena cada escala de la imagen en res.
'''
def piramide_gaussiana(img, s = 0.8, size = 4, border = 2):

    res = []
    aux = img
    for i in range(0, size):
        # Aplciamos reescalado y blur a la imagen.
        aux = blur_and_downsample(aux, s, border)
        res.append(aux)

    # Construimos la pirámide y la devolvemos.
    return res

'''
This function calculates the closest bigger power of two of the given number.
'''
def next_power_of_two(n):
    return 2**(n-1).bit_length()

'''
Returns the given image with the smallest black border that makes the image have power of two size.
'''
def borded_image(img):

    tamx = int(next_power_of_two(img.shape[0]) - img.shape[0])
    tamy = int(next_power_of_two(img.shape[1]) - img.shape[1])

    borded_img = np.zeros([img.shape[0]+tamx, img.shape[1]+tamy, 3]).astype(np.float32)
    borded_img[0:-tamx,0:-tamy,:] = img
    return borded_img, tamx, tamy

'''
Calculates the harmonic mean of the given valures.
'''
def harmonic_mean(x,y):
    np.seterr(divide='ignore', invalid='ignore')
    return x*y/(x+y)

"""
Uses Harris' descriptor to detect corners.
"""
def harris_detector(input_img, output_img, n_keypoints, block_size, ksize, scales, threshold):

    # Get grayscale of the image
    gray = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)

    blured_img = cv.GaussianBlur(gray, ksize = (0,0), sigmaX = 4.5)

    # Get gradients over each coordenate
    g1 = cv.Sobel(blured_img, -1, 1, 0)
    g2 = cv.Sobel(blured_img, -1, 0, 1)

    # Compute the gaussian piramid of each gradient and the image.
    piramide_g1 = [g1] + piramide_gaussiana(g1, 1, scales)
    piramide_g2 = [g2] + piramide_gaussiana(g2, 1, scales)
    piramide = [gray] + piramide_gaussiana(gray, 1, scales)

    # K is going to store all the keypoints.
    k = []

    # This is going to be the vector of images to return.
    imgs = []

    # For each of the scales.
    for scale in range(scales):

        # Get the gradient directions
        dx = piramide_g1[scale]
        dy = piramide_g2[scale]

        # Get the eigenvalues of the corresponding scale of the piramid.
        data = cv.cornerEigenValsAndVecs(piramide[scale], block_size, ksize)

        e1 = data[:,:,0]
        e2 = data[:,:,1]

        # Calculate matrix of harmonic means
        h_means = harmonic_mean(e1, e2)
        h_means[np.isnan(h_means)] = 0

        # Calculate local maxima
        peaks = peak_local_max(h_means, min_distance = block_size//2, num_peaks = n_keypoints, threshold_abs = threshold)
        print("\tMáximos en la escala ", scale, ": ", len(peaks))

        size = (scales - scale + 1)*block_size
        # Auxiliar array with the keypoints of the current scale.
        k_aux = []
       
        # For each of the local peaks.
        for peak in peaks:

            # Retrieve local maxima coordenates.
            x = peak[0]
            y = peak[1]
            # Calculate the angle of the gradient in that point.
            norm = np.sqrt(dx[x][y] * dx[x][y] + dy[x][y] * dy[x][y])
            sin = dy[x][y] / norm if norm > 0 else 0
            cos = dx[x][y] / norm if norm > 0 else 0
            angle = np.degrees(np.arctan2(sin, cos)) + 180

            # Add Keypoints to current vector.
            k_aux += [cv.KeyPoint(
                y, x,
                _size = size,
                _angle = angle
            )]

            # Add keypoints to global keypoints vector with the transformed coordenates.
            k += [cv.KeyPoint(
                y * (2**scale), x * (2**scale),
                _size = size,
                _angle = angle
            )]

        # Copy current scale
        scale_img = np.copy(piramide[scale])

        # Draw keypoints over it
        cv.drawKeypoints(piramide[scale], k_aux, scale_img, 0, 5)

        # Add it to the return array
        imgs += [scale_img]

    # Add all keypoints over the original color image and add it to the return array.
    imgs = [cv.drawKeypoints(input_img, k, output_img, flags = 5)] + imgs

    return imgs, k

"""
Calculates subpixel coordenates of the corners.
"""
def corners(img, keypoints):
    zoom = 5
    ret = []
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    win_size = (3,3)
    zero_zone = (-1, -1)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1)

    points = cv.KeyPoint_convert(keypoints)
    corners = points
    cv.cornerSubPix(gray, corners, win_size, zero_zone, criteria)
    indexes = random.sample(range(0, len(points) - 1), 3)

    for index in indexes:
        y, x = points[index].astype("uint32")
        ry, rx = corners[index]
        im = cv.resize(img, None, fx = zoom, fy = zoom)
        im = cv.circle(im, (zoom*y, zoom*x), 2, (255,0,0))
        im = cv.circle(im, (int(zoom*ry), int(zoom*rx)), 2, (0, 255, 0))
       
        t = max( zoom*(x-5), 0 )
        b = zoom*(x+5)
        l = max( zoom*(y-5), 0 )
        r = zoom*(y+5)
        window = im[t:b, l:r]
        ret.append(window)
       
    return ret

def ejercicio1(img = yos1, scales = 4, threshold = 15):

    borded, tamx, tamy = borded_image(img)
    res = harris_detector(borded, np.copy(borded), 1000, 3, 3, scales, threshold = threshold)
    imgs = res[0]
    keypoints = res[1]

    imgs[0] = imgs[0][0:-tamx, 0:-tamy, :]
    # pintaI(imgs[0])
    for i in range(0, scales):
        imgs[i+1] = imgs[i+1][0:-int(tamx/(2**i)), 0:-int(tamy/(2**i))]
        # pintaI(imgs[i+1])
    pintaI(hstack([imgs[0], hstack([imgs[1], vstack([*imgs[2:]], norm=False)], norm=False)], norm=False))
    savei(  hstack([imgs[0], hstack([imgs[1], vstack([*imgs[2:]], norm=False)], norm=False)], norm=False), "ej1abc.jpg")
    print("\tCoordenadas subpixel en Yosemite 1.")
    c = corners(img, keypoints)
    pintaMI(c)

"""
Retuns AKAZE's keypoints and descriptors.
"""
def akaze_descriptor(img):
    return cv.AKAZE_create().detectAndCompute(img, None)

"""
Returns matches between both descriptors using bruteforce
"""
def match_bf(desc1, desc2):

    matcher = cv.BFMatcher_create(crossCheck=True)
    matches = matcher.match(desc1,desc2)

    return matches

def match_2nn(desc1, desc2):


    matcher = cv.BFMatcher_create()
    matches = matcher.knnMatch(desc1, desc2, k = 2)

    ret = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            ret.append(m)

    return ret

"""
Returns two images with the corresponding keypoings using AKAZE, one using bruteforce and the other
with Lowe-Average-2NN
"""
def get_matches(img1, img2):

    kpts1, desc1 = akaze_descriptor(img1)
    kpts2, desc2 = akaze_descriptor(img2)

    print('\tA-KAZE Matching Results')
    print('\t*******************************')
    print('\t Keypoints 1:                        \t', len(kpts1))
    print('\t Keypoints 2:                        \t', len(kpts2))

    bf_matches = random.sample(match_bf(desc1, desc2), 100)
    bf_img = cv.drawMatches(img1, kpts1, img2, kpts2, bf_matches, None,
                            flags = cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


    nn_matches = random.sample(match_2nn(desc1, desc2), 100)
    nn_img = cv.drawMatches(img1, kpts1, img2, kpts2, nn_matches, None,
                             flags = cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    return bf_img.astype(np.float32), nn_img.astype(np.float32)

def ejercicio2():

    bf_img, nn_img = get_matches(yos1, yos2)

    print("\tBruteForce + crossCheck en Yosemite")
    pintaI(bf_img)
    print("\tLowe-Average-2NN en Yosemite")
    pintaI(nn_img)


def ejercicio3(img1 = yos1.astype(np.float32), img2 = yos2.astype(np.float32)):

    h, w = img1.shape[0], 940
    #Construimos un canvas
    canvas = np.zeros( (h, w, 3), dtype = np.float32)

    # La homografía que lleva la primera imagen en el canvas es la homografía identidad.
    canvas[:img1.shape[0], :img1.shape[1]] = img1

    # Calculamos la homografía de la segunda imagen a la primera
    # Para ello utilizamos el descriptor AKAZE
    kpts1, desc1 = akaze_descriptor(img1)
    kpts2, desc2 = akaze_descriptor(img2)

    matches = match_2nn(desc2, desc1)

    q = np.array([kpts2[match.queryIdx].pt for match in matches])
    t = np.array([kpts1[match.trainIdx].pt for match in matches])

    H_21 = cv.findHomography(q, t, cv.RANSAC, 1)[0]

    canvas = cv.warpPerspective(img2, H_21, (w, h), dst = canvas, borderMode = cv.BORDER_TRANSPARENT)

    pintaI(canvas)


def get_homography(im1, im2):
    """Estima una homografía de 'im1' a 'im2'."""

    kp1, desc1 = akaze_descriptor(im1)
    kp2, desc2 = akaze_descriptor(im2)

    matches = match_2nn(desc1, desc2)

    query = np.array([kp1[match.queryIdx].pt for match in matches])
    train = np.array([kp2[match.trainIdx].pt for match in matches])

    return cv.findHomography(query, train, cv.RANSAC)[0]


def ejercicio4(imgs):

    # Center index
    index_img_center = len(imgs)//2
    # center img
    img_center =  imgs[index_img_center]

    w = sum([img.shape[1] for img in imgs])
    h = imgs[0].shape[0]*2

    canvas = np.zeros( (h, w, 3), dtype = np.float32)

    # Calculamos la homografía que lleva la imagen central al mosaico
    H_0 = np.array([
        [1, 0, (w - img_center.shape[1])/2],
        [0, 1, (h - img_center.shape[0])/2],
        [0, 0, 1]])

    # Trasladamos la imagen central al mosaico
    canvas = cv.warpPerspective(imgs[index_img_center], H_0, (w, h),
                                dst = canvas, borderMode = cv.BORDER_TRANSPARENT)

    # Calculamos las homografías entre cada dos imágenes
    Hom = []
    for i in range(len(imgs)):
        if i != index_img_center:
            j = i + 1 if i < index_img_center else i - 1

            Hom.append(get_homography(imgs[i], imgs[j]))

        else: # No se usa la posición central
            Hom.append(np.array([]))

    H = H_0
    G = H_0
    for i in range(index_img_center)[::-1]:
        H = H @ Hom[i]
        canvas = cv.warpPerspective(imgs[i], H, (w, h), dst = canvas, borderMode = cv.BORDER_TRANSPARENT)

        j = 2 * index_img_center - i
        if j < len(imgs):
            G = G @ Hom[j]
            canvas = cv.warpPerspective(imgs[j], G, (w, h), dst = canvas, borderMode = cv.BORDER_TRANSPARENT)

    pintaI(canvas)

def main():
    print(" EJERCICIO 1: Detector de Harris.")
    ejercicio1()

    print("EJERCICIO 2: Correspondencias entre KeyPoints")
    ejercicio2()

    print("EJERCICIO 3: Mosaico con 2 imágenes")
    ejercicio3()

    print("EJERCICIO 4: Mosaico con N imágenes")
    ejercicio4(
        [yos1.astype(np.float32),
         yos2.astype(np.float32),
         yos3.astype(np.float32),
         yos4.astype(np.float32),
        ])
    ejercicio4(
        [yos5.astype(np.float32),
         yos6.astype(np.float32),
         yos7.astype(np.float32),
        ])

main()
