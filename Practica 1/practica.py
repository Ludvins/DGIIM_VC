import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from numpy.linalg import matrix_rank
import scipy

'''
Transforma una imagen de codificación BGR a RGB, realizando una llamada a cv2.cvtColor
'''
def BGRtoRGB(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

'''
Comprueba si una imagen es monobanda.
'''
def isBW(im):
    return len(im.shape) == 2

'''
Normaliza los valores de la imagen, poniendo el valor maximo a 1.0 y el mínimo a 0.0.
'''
def normalize(im):
    max_val = np.amax(im)
    min_val = np.amin(im)
    im = (im - min_val) / (max_val - min_val)
    return im

'''
Lee una imagen dado un nombre de archivo "filename" y un flag que indica si la imagen es a color o no "flagColor".
Pasamos los valores a f32 ya que hay funciones de opencv como cvtcolor que no admiten mas precisión.
'''
def leeimagen(filename, flagColor):
    if flagColor == 1:
        return BGRtoRGB(cv2.imread(filename, flagColor)).astype("float32")
    return cv2.imread(filename, flagColor).astype("float32")

'''
Transforma una imagen a escala de grises.
'''
def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
Guarda una imagen "im", en el archivo "file".
Se puede especificar un título de imagen y si se desea normalizar antes de guardarla.
'''
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
    cv2.waitKey(0)
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
            aux[i] = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        # Si no llega a la altura máxima, añadimos un borde blanco.
        if im.shape[0] < altura:
            aux[i] = cv2.copyMakeBorder(
                aux[i], 0, altura - vim[i].shape[0],
                0, 0, cv2.BORDER_CONSTANT, value = border
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
            aux[i] = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        if im.shape[1] < anchura:
            aux[i] = cv2.copyMakeBorder(
                aux[i], 0, 0,
                0, anchura - aux[i].shape[1], cv2.BORDER_CONSTANT, value = border
            )
    return np.vstack(aux)

def representaIm(vim, titulos, norm = True):
    pintaMI(vim, titulo = " | ".join(titulos), norm = norm)

'''
Declara las imágenes de la práctica
'''
marilyn = normalize(leeimagen("imagenes/marilyn.bmp", 1))
einstein = normalize(leeimagen("imagenes/einstein.bmp", 1))
cat = normalize(leeimagen("imagenes/cat.bmp", 1))
dog = normalize(leeimagen("imagenes/dog.bmp", 1))
fish = normalize(leeimagen("imagenes/fish.bmp", 1))
submarine = normalize(leeimagen("imagenes/submarine.bmp", 1))
plane = normalize(leeimagen("imagenes/plane.bmp", 1))
bird = normalize(leeimagen("imagenes/bird.bmp", 1))
bici = normalize(leeimagen("imagenes/bicycle.bmp", 1))
moto = normalize(leeimagen("imagenes/motorcycle.bmp", 1))
im = normalize(leeimagen("imagenes/cat.bmp", 1))

'''
Función encargada de añadir los bordes a una imagen.
En caso de no indicar un tamaño para el borde en el eje y, se seleccionará el mismo que el del eje x.
Permite crear tres tipos de bordes
+ Constante. Dado por un valor pasado por parámetro. Por defecto negro.
+ Reflejado. Hace un efecto espejo en los bordes.
+ Replicado. Replica la el ultimo pixel de la imagen en el borde.
'''
def make_border(img, tamx, tamy=0, tipo = 2, value = 0):

    if tamy == 0:
        tamy = tamx

    # Rellena la imagen a devolver con 0.
    ret = np.zeros([img.shape[0]+2*tamy, img.shape[1]+2*tamx])

    if tipo == 0:
        # En caso de ser borde constante, convierte toda la matriz a dicho color e inserta la imagen en el centro.
        ret = ret+value
        ret[tamy:-tamy,tamx:-tamx] = img

    if tipo == 1:
        # Copiamos las filas y las columnas en los bordes, haciendo efecto espejo
        ret[tamy:-tamy,tamx:-tamx] = img
        for i in range(0, tamy):
            ret[i][tamx:-tamx] = img[tamy-i]
        for i in range(0, tamx):
            ret.T[i][tamy:-tamy] = img.T[tamx-i]
        for i in range(-tamy, 0):
            ret[i][tamx:-tamx] = img[-tamy-i-1]
        for i in range(-tamx, 0):
            ret.T[i][tamy:-tamy] = img.T[-tamx-i-1]

    if tipo == 2:
        # Copiamos la última fila y la última columna en los bordes.
        ret[tamy:-tamy,tamx:-tamx] = img
        for i in range(0, tamy):
            ret[i][tamx:-tamx] = img[0]
        for i in range(0, tamy):
            ret.T[i][tamy:-tamy] = img.T[0]
        for i in range(-tamy, 0):
            ret[i][tamx:-tamx] = img[-1]
        for i in range(-tamy, 0):
            ret.T[i][tamy:-tamy] = img.T[-1]

    return ret.astype("float32")

'''
Ejemplo de ejecución de la funcion de bordes.
'''
def ejemplo_make_border():
    pintaI(hstack(
        [
            make_border(normalize(leeimagen("imagenes/cat.bmp", 0)), 20, tipo = 0, value = 0.1),
            make_border(normalize(leeimagen("imagenes/cat.bmp", 0)), 20, tipo = 1),
            make_border(normalize(leeimagen("imagenes/cat.bmp", 0)), 20)
        ],
        "Constante | Reflejar | Replicar"), "ejemplo_make_border.bmp")

'''
Aplica la convolución de un vector con una máscara 1D. El calculo se realiza de la forma mas rudimentaria.
'''
def m_convolve(v, k):
    # Invertimos el vector de la máscara
    k = k[::-1]
    l = len(k)//2
    aux = np.copy(v)
    # Recorremos el vector principal. Ignorando los bordes.
    for i in range(l, len(v)-l):
        aux2 = 0
        # Recorremos el segundo vector.
        for j in range(-l, l+1):
            if (i + j >= 0 and i+j < len(v)):
                aux2 += v[i+j]*k[j+l]
        aux[i] = aux2
    return aux

'''
Aplica la convolución de una imagen con dos máscaras 1D.
Se puede especificar el tipo de borde a usar y el valor del borde constante.
Se puede especificar el modo "manual" para utilizar la convolución declarada antes. En otro caso se utilzará la convolución de numpy.
'''
def convolucion2D_monobanda(img, vx, vy, border_type = 2, value = 0, manual=False):
    nrows = img.shape[0]
    ncols = img.shape[1]

    kx = len(vx) // 2
    ky = len(vy) // 2
    # Añadimos bordes a la imagen.
    img_res = make_border(img, kx, ky, border_type, value)

    # Hacemos la convolución por filas.
    for i in range(0, nrows + 2 * ky):
        if (manual):
            img_res[i] = m_convolve(img_res[i], vx)
        else:
            img_res[i] = np.convolve(img_res[i], vx, "same")

    # Hacemos la convolución por columnas.
    for j in range(0, ncols + 2 * kx):
        if (manual):
            img_res[:,j] = m_convolve(img_res[:,j], vy)
        else:
            img_res[:,j] = np.convolve(img_res[:,j], vy, "same")

    return img_res[ky:-ky, kx:-kx].astype("float32")

'''
Aplica la convolucion a cada capa de una imagen multibanda. Tambien funciona en imagenes monobanda.
'''
def convolucion2D(img, vx, vy, border_type = 2, value = 0, manual = False):

    if isBW(img):
        return convolucion2D_monobanda(img, vx, vy, border_type, value, manual)

    aux = np.copy(img)
    for dimension in range(img.shape[2]):
        aux[:,:,dimension] = convolucion2D_monobanda(img[:,:,dimension],vx,vy, border_type, value, manual)

    return aux

'''
Función de ejemplo de la función de convolución
'''
def ejemplo_convolucion():
    blur = [1.0/9 for i in range(9)]
    im1 = normalize(leeimagen("imagenes/cat.bmp", 0))
    im2 = convolucion2D(im1, blur, blur)
    im = normalize(leeimagen("imagenes/cat.bmp", 1))
    im3 = convolucion2D(im, blur, blur)
    pintaMI([im1, im2], titulo="Blanco y negro")
    pintaMI([im,im3], "Color")

'''
Función Gaussiana de media 0 sin denominador. sig corresponde a la desviación tipica de la función.
'''
def gaussian(x, sig):
    return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))

'''
Aplica un filtro Gaussiano a una imagen dada. Acepta ambos valores de sigma por parámetros, en caso de solo especificar 1, el segundo se iguala al primero. Lo mismo ocurre con los tamaños de los kernels.
Por defecto se utilizará un tamaño de máscara de [-3sigma, 3sigma].
'''
def gaussiana(img, sx, sy=0, tamx = 0, tamy = 0, border_type = 2, value = 0):

    if sy == 0:
        sy = sx

    vx = []
    vy = []

    x_bound = tamx // 2
    y_bound = tamy // 2

    if (x_bound == 0):
        x_bound = int(3*sx)

    if (y_bound == 0):
        y_bound = int(3*sy)

    # Añadimos los valores de la máscara del eje x.
    for i in range(-x_bound, x_bound+1):
        vx.append(gaussian(i,sx))
    # Añadimos los valores de la máscara del eje y
    for i in range(-y_bound, y_bound+1):
        vy.append(gaussian(i,sy))

    # Normalizamos según la norma 1.
    vx = vx/np.sum(vx)
    vy = vy/np.sum(vy)

    return convolucion2D(img, vx, vy, border_type, value)

'''
Función que ejecuta un caso de ejemplo de filtrado por una máscara gaussiana.
'''
def ejemplo_gaussiana():
    im = normalize(leeimagen("imagenes/cat.bmp", 1))
    representaIm([im, gaussiana(im, 3), gaussiana(im, 15)], ["Original", "sX = xY = 3","sX = sY = 15"])
    representaIm([im, gaussiana(im, 1, 5), gaussiana(im, 5, 1)], ["Original", "sX = 1, sY = 5", "sX = 5, sY = 1"])
    representaIm([im, gaussiana(im, 15, 15, 3, 3), gaussiana(im,15 , 15, 5, 5)], ["Original", "sX = sY = 15, tam=3", "sX = sY = 15  tam=5"])
    representaIm([gaussiana(im,3, border_type=0), gaussiana(im, 3, border_type= 1), gaussiana(im, 3, border_type = 2)], ["Borde tipo 0", "Borde tipo 1","Borde tipo 2"])

'''
Realiza llamadas a getDerivKernels para obtener máscaras de derivadas normalizadas con los parámetros buscados.
'''
def derivadas(dx, dy, tam):
    return cv2.getDerivKernels(dx,dy,tam, normalize=True)

'''
Muestra máscaras de derivadas de tamaño 3 y 5.
'''
def ejemplo_vectores_derivadas():
    ders = [(0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
    tams = [3, 5]

    for tam in tams:
        print("Tamaño = {}".format(tam))
        for dx, dy in ders:
            print("  ({}, {})".format(dx, dy), end = ": ")
            print("{}, {}".format(*map(np.transpose, derivadas(dx, dy, tam))))
    input("(Vectores mostrados. Pulsa cualquier tecla para continuar...)")
def norm_abs(im):
    return normalize(abs(im))

'''
Funcion de ejemplo de aplicación de máscaras derivadas.
'''
def ejemplo_derivadas():
    representaIm( [abs(convolucion2D(bird, derivadas(0,1,3)[0].T[0], derivadas(0,1,3)[1].T[0])),
                   abs(convolucion2D(bird, derivadas(1,0,3)[0].T[0], derivadas(1,0,3)[1].T[0])),
                   abs(convolucion2D(bird, derivadas(2,0,3)[0].T[0], derivadas(2,0,3)[1].T[0])),
                   abs(convolucion2D(bird, derivadas(2,0,7)[0].T[0], derivadas(2,0,7)[1].T[0]))
    ],
                  ["dy - tam = 3", "dx - tam = 3", "dxx - tam = 3", "dxx - tam = 7"])

'''
Aplica un filtro de Laplaciana-de-Gaussiana a una imagen, con los parámetros correspondientes.
'''
def laplaciano(im, s, tam = 3, border = 2, value = 0):
    # Aplicamos la Gaussiana
    g = gaussiana(im, s, tamx = tam, border_type = border, value=value)
    # Calculamos las mascaras de derivadas
    d = derivadas(2, 0, tam)

    # Calculamos las derivadas.
    g1 = convolucion2D(g, d[0].T[0], d[1].T[0], border_type = border, value = value)
    g2 = convolucion2D(g, d[1].T[0], d[0].T[0], border_type = border, value = value)
    return abs(g1+g2)

'''
Función de ejemplo de Laplaciano-de-Gaussiana
'''
def ejemplo_laplaciana():
    representaIm([laplaciano(plane, 2, 3), laplaciano(plane, 2, 5)],
             ["Tamaño 3"] + ["Tamaño 5"])
    representaIm([laplaciano(plane, 3, 3, 0), laplaciano(plane, 3, 3, 1)],
             ["Borde constante"] + ["Borde reflejado"])
    representaIm([laplaciano(plane, 1, 7), laplaciano(plane, 3, 7)],
             ["sigma 1"] + ["sigma 3"])

'''
Aplica un alisado y un reescalado a la mitad en ambos ejes a una imagen.
'''
def blur_and_downsample(img, s, border = 2):
    if isBW(img):
        return normalize(gaussiana(img,s, border_type = border)[::2,::2])
    return normalize(gaussiana(img, s, border_type = border)[::2,::2,:])

'''
Aplica un reescalado al doble en ambos ejes y un alisado a una imagen.
'''
def blur_and_upsample(img, s, shape):
    return normalize(
        gaussiana(
            cv2.resize(img, (shape[1], shape[0]), 0, 0, cv2.INTER_NEAREST), s
        )
    )

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
    return (hstack([img, vstack(res)]), res)

'''
Función de ejemplo para la pirámide gaussiana.
'''
def ejemplo_piramide_g():
    pintaI(piramide_gaussiana(im, 0.7, border = 0)[0], norm = False)
    pintaI(piramide_gaussiana(im, 0.7, border = 2)[0], norm = False)
    pintaI(piramide_gaussiana(im, 2)[0], norm=False)

'''
Función auxiliar, aplica un paso de la pirámide laplaciana. Calcula las dos imagenes auxiliares de cada etapa y las devuelve.
'''
def laplacian_step(img ,s, border):
    d = blur_and_downsample(img, s, border)
    u = blur_and_upsample(d, s, img.shape)
    return (d, img - u)

'''
Construye la pirámide Laplaciana de una imagen. Calcula cada una de las etapas de la pirámide laplaciana de una imagen. Inserta al final la imagen que nos permitirá reconstuir la original.
'''
def piramide_laplaciana(img, s=1, size=4, border=2):
    
    res = []
    aux, l = laplacian_step(img, s, border)
    
    for i in range(0, size-1):
        aux, aux2 = laplacian_step(aux, s, border)
        res.append(aux2)

    res.append(aux)
    aux = np.copy(res)
    res.insert(0,l)
    return (hstack([l, vstack(aux)]),res )

'''
Muestra un ejemplo de piramide Laplaciana
'''
def ejemplo_piramide_l():
    pintaI(piramide_laplaciana(im)[0])
    pintaI(piramide_laplaciana(im, border=0)[0])

'''
Reconstruye una imagen a partir de su pirámide laplaciana.
Para reconstuir la imagen original vamos sumando las imagenes que componen la piramide.
'''
def reconstruct_original(l, s):
    
    aux = l[-1]
    for i in l[0:-1][::-1]:
        aux = blur_and_upsample(aux, s, i.shape)
        aux += i
    
    return normalize(aux)

'''
Función de ejemplo para reconstruir una imagen
'''
def ejemplo_reconstruir():
    s = 2
    l = piramide_laplaciana(im, s)
    a = reconstruct_original(l[1], s)
    print("Maximo valor de la diferencia ", np.amax(abs(im-a)))
    input("(Pulsa cualquier tecla para continuar...)")
    pintaMI( [a,im ], "Reconstrucción | Original",norm = False)

'''
Funcion auxiliar, calcula los vecinos de un punto en el cubo formado por 3 escalas.
'''
def get_neighbours(img, imga, imgd,  i, j):

    x = [-1, 0, 1, -1, 1, -1, 0, 1]
    y = [-1, -1, -1, 0, 0, 1, 1, 1]
    # Añadimos el mismo punto en las otras dos escalas
    aux = [imga [i][j], imgd[i][j] ]
    for k in range(0, 8):
        pos = (i + x[k], j + y[k])
        if (i + x[k] in range(0, img.shape[0]) and j+y[k] in range(0, img.shape[1])):
            # añadimos todos los vecinos en las 3 escalas.
            aux.append( img [ i+x[k] ] [ j+y[k] ] )
            aux.append( imga [ i+x[k] ] [ j+y[k] ] )
            aux.append( imgd [ i+x[k] ] [ j+y[k] ] )
    return aux

'''
Función auxiliar. Busca los maximos de una escala. Tambien conprueba que sea mayores que un umbral.
'''
def get_local_max(img, imga, imgd, threshold = 0.01):
    aux = []
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if (img[i][j] > max(threshold, np.amax(get_neighbours(img, imga, imgd, i, j)))):
                aux.append((j,i))
    return aux

'''
Construye el espacio de escalas Laplaciano de una imagen.
'''
def escalas(img, s, n, k = 1.2):
    l = []

    # Calculamos todas las escalas.
    c = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i in range(0, n+2):
        l.append(np.square(normalize(laplaciano(img, s*(k**i), tam=7))))

    # Para cada escala intermedia, pintamos los maximos.
    for i in range(1, n+1):
        for item in get_local_max(l[i], l[i-1], l[i+1]):
            cv2.circle(c, item, int(2*s*s), (0.8*s,0.2*s,0.4*s))
        s = s*k
    return c.astype(np.double)

'''
Función de ejemplo de espacio de escalas de Laplaciano.
'''
def ejemplo_escalas():
    pintaI(escalas(gray(moto), 1, 6))

'''
Imprime las frecuencias altas de una imagen, las bajas de otra y la suma de ambas.
'''
def print_low_high_hybrid(im1, im2, s1, s2):
    low = gaussiana(im1, s1)
    high = im2 - gaussiana(im2, s2)
    pintaMI([low, high, low+high], "Low | High | Low+High")

'''
Muestra ejemplos de imágenes híbridas con parámetros ajustados.
'''
def ejemplo_hibrida_bn():
    print_low_high_hybrid(gray(submarine), gray(fish), 3 , 6)
    print_low_high_hybrid(gray(marilyn), gray(einstein), 3, 9)
    print_low_high_hybrid(gray(dog), gray(cat), 8, 8)

'''
Construye la imagen híbrida de dos dadas juntos con sus parámetros.
'''
def hibrid(im1, im2, s1, s2):
    low = gaussiana(im1, s1)
    high = im2 - gaussiana(im2, s2)
    return low+high

'''
Muestra la piramide Gaussiana de unas imagenes hibridas con parámetros ajustados.
'''
def ejemplo_piramide_h():
    pintaI(piramide_gaussiana(hibrid(gray(marilyn), gray(einstein), 3, 9))[0])
    pintaI(piramide_gaussiana(hibrid(gray(dog), gray(cat), 8,8))[0])
    pintaI(piramide_gaussiana(hibrid(gray(submarine), gray(fish), 3, 6))[0])

'''
Calcula las dos máscaras 1D que forman la máscara 2D en caso de ser posible.
'''
def convolucion2D_m(img, k, border_type = 2, value = (0,0,0)):
    if (matrix_rank(k) == 1):
        u = k[0]
        v = k.T[0]

        # Buscamos una fila no nula.
        for i in range(0, k.shape[0]):
            if (np.count_nonzero(k[i]) != 0):
                u = k[i]
                break
        # Buscamos una columna no nula.
        for i in range(0, k.shape[1]):
            if (np.count_nonzero(k.T[i]) != 0):
                v = k.T[i]
                break

        # Los comvertimos en Matrices
        u = np.matrix(u).T
        v = np.matrix(v)

        # Normalizamos uno de ellos, de forma que su producto resulte la matriz original.
        u = u*np.amax(k/(u@v))

        # Los convertimos en arrays
        u = np.squeeze(np.asarray(u))
        v = np.squeeze(np.asarray(v))

        return convolucion2D(img, u, v, border_type, value, manual=True)

    # si la matriz no tiene rango 1, devolvemos la original.
    return img

'''
Muestra ejemplo de máscara 2D separable.
'''
def ejemplo_bonus1():
    N=9
    M=9
    a = np.empty((N,M))
    a[:] = 1/(N*M)
    im2 = convolucion2D_m(im, a)
    blur = [1.0/9 for i in range(9)]
    pintaMI([convolucion2D(im,blur , blur), im2], norm=False, titulo="Kernels 1D | Kernel 2D")

'''
Muestra pirámides gaussianas de imágenes hibridas a color, con parámetros ya ajustados.
'''
def ejemplo_bonus2():
    pintaI(piramide_gaussiana(hibrid(marilyn, einstein, 3,7))[0])
    pintaI(piramide_gaussiana(hibrid(dog,cat, 8,8))[0])
    pintaI(piramide_gaussiana(hibrid(submarine, fish, 3, 6))[0])
    pintaI(piramide_gaussiana(hibrid(plane, bird, 8,8), 0.8)[0])
    pintaI(piramide_gaussiana(hibrid(bici, moto, 7,10), 0.8)[0])


def ejercicio1A():
    print("Apartado A")
    print("Ejemplo Gaussiana.")
    ejemplo_gaussiana()
    print("Ejemplo Vectores Derivadas.")
    ejemplo_vectores_derivadas()
    print("Ejemplo derivadas.")
    ejemplo_derivadas()
    input("(Apartado A finalizado. Pulsa cualquier tecla para continuar...)")

def ejercicio1B():
    print("Apartado B")
    print("Ejemplo Laplaciana.")
    ejemplo_laplaciana()
    input("(Apartado B finalizado. Pulsa cualquier tecla para continuar...)")

def ejercicio2A():
    print("Apartado A")
    print("Ejemplo Piramide Gaussiana")
    ejemplo_piramide_g()
    input("(Apartado A finalizado. Pulsa cualquier tecla para continuar...)")

def ejercicio2B():
    print("Apartado B")
    print("Ejemplo Piramide Laplaciana")
    ejemplo_piramide_l()
    print("Ejemplo reconstruir")
    ejemplo_reconstruir()
    input("(Apartado B finalizado. Pulsa cualquier tecla para continuar...)")

def ejercicio2C():
    print("Apartado C")
    print("Ejemplo escalas")
    ejemplo_escalas()
    input("(Apartado C finalizado. Pulsa cualquier tecla para continuar...)")

def ejercicio3():
    print("Ejercicio 3 -----")
    print("Ejemplo imagenes híbridas")
    ejemplo_hibrida_bn()
    print("Ejemplo Piramides imágenes hibridas")
    ejemplo_piramide_h()
    input("(Ejercicio 3 finalizado. Pulsa cualquier tecla para continuar...)")

def bonus1():
    print("Bonus 1 -----")
    print("Ejemplo bordes")
    ejemplo_make_border()
    print("Ejemplo convolución")
    ejemplo_convolucion()
    print("Ejemplo mascara 2D separable")
    ejemplo_bonus1()
    input("(Bonus 1 finalizado. Pulsa cualquier tecla para continuar...)")

def bonus2():
    print("Bonus 2 ------")
    ejemplo_bonus2()
    input("(Bonus 2 finalizado. Pulsa cualquier tecla para continuar...)")

def main():
    print("Ejercicio 1 ------")
    ejercicio1A()
    ejercicio1B()
    print("Ejercicio 2 ------")
    ejercicio2A()
    ejercicio2B()
    ejercicio2C()
    ejercicio3()
    bonus1()
    bonus2()

main()
