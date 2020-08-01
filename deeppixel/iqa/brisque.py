# Getting our dependencies
import collections
from itertools import chain
import urllib.request as request
import pickle 
import numpy as np
import scipy.signal as signal
import scipy.special as special
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import skimage
from skimage import io,transform
import cv2
import libsvm
from libsvm import svmutil

#Natural Scene Statistics
def normalized(kernel):
    return kernel / np.sum(kernel)

def gaussian_kernel2d(n, SIGMA):
    '''
    Compute the locally normalized luminances via local mean subtraction and divide it by the local deviation
    '''       
    Y, X = np.indices((n, n)) - int(n/2)
    gaussian_kernel = 1 / (2 * np.pi * SIGMA ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * SIGMA ** 2)) 
    return normalized(gaussian_kernel)

def local_mean(image, kernel):
    """
    Finding local mean by applying Gaussian filter to the color image
    """
    return signal.convolve2d(image, kernel, 'same')

def local_deviation(image, local_mean, kernel):
    '''
    Calclulating the deviation by finding the square root of observed value and the local_mean    
    '''
    sigma = image ** 2
    sigma = signal.convolve2d(sigma, kernel, 'same')
    return np.sqrt(np.abs(local_mean ** 2 - sigma))

def MSCN_coefficients(image, kernel_size=6,SIGMA=7/6): 
    '''
    Normalize the image using Mean Substracted Contrast Normalization (MSCN).
    '''  
    C = 1/255
    kernel = gaussian_kernel2d(kernel_size, SIGMA=SIGMA)
    local_mean = signal.convolve2d(image, kernel, 'same')
    local_var = local_deviation(image, local_mean, kernel)
    
    return (image - local_mean) / (local_var + C)


def generalized_gaussian_dist(x, alpha, sigma):
    beta = sigma * np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
    
    coefficient = alpha / (2 * beta() * special.gamma(1 / alpha))
    return coefficient * np.exp(-(np.abs(x) / beta) ** alpha)

"""
adjacent coefficients also exhibit a regular structure, which gets disturbed in the presence of distortion.
To avoid this pairwise products of neighboring MSCN coefficients along 
four directions (1) horizontal H, (2) vertical V, (3) main-diagonal D1 and (4) secondary-diagonal D2 are used
"""
def calculate_pair_product_coefficients(mscn_coefficients):    
    return collections.OrderedDict({                           
        'mscn': mscn_coefficients,
        'horizontal': mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:],
        'vertical': mscn_coefficients[:-1, :] * mscn_coefficients[1:, :],
        'main_diagonal': mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:],
        'secondary_diagonal': mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
    })
    

def detect_blur_fft(image, size=60, thresh=20, vis=False):
    """
    we get the dimensions of the image and
    derive the center (x, y) coordinates 	
    """
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))
	fft = np.fft.fft2(image)          # we compute the FFT to find the frequency transform
	fftShift = np.fft.fftshift(fft)   # shiting the zero frequency component to the center

	if vis:
		plt.imshow(image)
		plt.show()

	fftShift[cY - size:cY + size, cX - size:cX + size] = 0   # zero-out the center of the FFT shift
	fftShift = np.fft.ifftshift(fftShift)    # apply the inverse shift so that component once again becomes the top-left
	recon = np.fft.ifft2(fftShift)      # get the inverse fourier feature transform 
	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)
	return (mean, mean <= thresh)   #mean value < thresh to signify if image is blurry or not
    

def asymmetric_generalized_gaussian_fit(x): 
    """
    An Asymmetric Generalized Gaussian Distribution (AGGD) is fit to each of the four pairwise product images.
    AGGD is an asymmetric form of Generalized Gaussian Fitting (GGD).
    It has four parameters — shape, mean, left variance and right variance
    """
    def estimate_phi(ALP):
        '''
        Calculate γ where Nₗ is the number of negative samples and Nᵣ is the number of positive samples.
        '''
        numerator = special.gamma(2 / ALP) ** 2
        denominator = special.gamma(1 / ALP) * special.gamma(3 / ALP)
        return numerator / denominator
    
    
    def estimate_r_hat(x):
        size = np.prod(x.shape)
        return (np.sum(np.abs(x)) / size) ** 2 / (np.sum(x ** 2) / size)
    
    def estimate_R_hat(r_hat, gamma):
        '''
        Calculate R hat using γ and r hat estimations.
        '''
        num = (gamma ** 3 + 1) * (gamma + 1)
        den = (gamma ** 2 + 1) ** 2
        return r_hat * num / den
    
    #calculatin mean squares using filtered values
    def mean_squares_sum(x, filter = lambda z: z == z):
        filtered_values = x[filter(x)]
        squares_sum = np.sum(filtered_values ** 2)
        return squares_sum / ((filtered_values.shape))
    

    def estimate_gamma(x):
        left_squares = mean_squares_sum(x, lambda z: z < 0)
        right_squares = mean_squares_sum(x, lambda z: z >= 0)
        return np.sqrt(left_squares) / np.sqrt(right_squares)
    
    def estimate_alpha(x):
        '''
        Estimate α using the approximation of the inverse generalized Gaussian ratio
        '''
        r_hat = estimate_r_hat(x)
        gamma = estimate_gamma(x)
        R_hat = estimate_R_hat(r_hat, gamma)

        solution = optimize.root(lambda z: estimate_phi(z) - R_hat, [0.2]).x
        return solution[0]
    

    def estimate_sigma(x, alpha, filter = lambda z: z < 0):
        return np.sqrt(mean_squares_sum(x, filter))
    
    
    def estimate_mean(alpha, sigma_l, sigma_r):
        return (sigma_r - sigma_l) * constant * (special.gamma(2 / alpha) / special.gamma(1 / alpha))
    
    #Estimate left and right scale parameters.
    ALPHA = estimate_alpha(x)
    Sigma_LEFT = estimate_sigma(x, ALPHA, lambda z: z < 0)
    Sigma_RIGHT = estimate_sigma(x, ALPHA, lambda z: z >= 0)
    
    constant = np.sqrt(special.gamma(1 / ALPHA) / special.gamma(3 / ALPHA))
    Mean = estimate_mean(ALPHA, Sigma_LEFT, Sigma_RIGHT)
    return ALPHA, Mean, Sigma_LEFT, Sigma_RIGHT    

def calculate_brisque_features(image, kernel_size=7, sigma=7/6):
    """
    The features needed to calculate the image quality are the result of fitting the 
    MSCN coefficients and shifted products to the Generalized Gaussian Distributions(GGD). First, 
    we need to fit the MSCN coefficients to the GGD, then the pairwise products to the AsymmetricGGD.
    """
    def calculate_features(coefficients_name, coefficients, accum=np.array([])):
        alpha, mean, sigma_l, sigma_r = asymmetric_generalized_gaussian_fit(coefficients)

        if coefficients_name == 'mscn':
            var = (sigma_l ** 2 + sigma_r ** 2) / 2
            return [alpha, var]
        
        return [alpha, mean, sigma_l ** 2, sigma_r ** 2]
    
    mscn_coefficients = MSCN_coefficients(image, kernel_size, sigma)
    coefficients = calculate_pair_product_coefficients(mscn_coefficients)
    
    features = [calculate_features(name, coeff) for name, coeff in coefficients.items()]
    flatten_features = list(chain.from_iterable(features))
    return np.array(flatten_features)

def imgurl(url):
    '''
    Loading the image from any url using the scikit learn package
    '''
    image_stream = request.urlopen(url)
    image=io.imread(image_stream, plugin='pil')
    gray_image=skimage.color.rgb2gray(image)
    return image,gray_image

def plot_histogram(x, label=""):
    '''
    plotting a histogram using matplotlib
    '''
    n, bins = np.histogram(x.ravel(), bins=50)
    n = n / np.max(n)
    plt.plot(bins[:-1], n, label=label, marker='o')
    
def imgread(path):
    '''
    Loading the image
    '''
    image = io.imread(path,as_gray=False)
    gray_image = skimage.color.rgb2gray(image)
    return image,gray_image
    

def imgshow(image,title=""): 
    '''
     Showing the image to the user
    '''
    plt.axis("off")
    plt.title(title) 
    plt.imshow(image)
    plt.show()
    
    
def image_coefficients(gray_image):
    mscn_coefficients = MSCN_coefficients(gray_image, 7, 7/6)
    coefficients = calculate_pair_product_coefficients(mscn_coefficients)
    return coefficients
    
def histogram_analysis(gray_image,x=10,y=5):
    '''
    Fit all the coefficients and show the Generalized Gaussian Distribution using a histogram
    '''
    coefficients=image_coefficients(gray_image)
    plt.rcParams["figure.figsize"] = x, y
    for name, coeff in coefficients.items():
        plot_histogram(coeff.ravel(), name)

    plt.axis([-2.5, 2.5, 0, 1.05])
    plt.legend()
    plt.show()
    

def BRISQUE_FEATURE(gray_image):
    brisque_features = calculate_brisque_features(gray_image, kernel_size=7, sigma=7/6)
    downscaled_image = cv2.resize(gray_image, None, fx=1/2, fy=1/2, interpolation = cv2.INTER_CUBIC)
    downscale_brisque_features = calculate_brisque_features(downscaled_image, kernel_size=7, sigma=7/6)
    brisque_features = np.concatenate((brisque_features, downscale_brisque_features))
    
    return brisque_features
    
def scaled(features):
    '''
    In order to have good results, we need to scale the features to [-1, 1]
    '''
    with open(r"deeppixel\iqa\trained.pickle", 'rb') as handle:
        scale_params = pickle.load(handle)
    min_ = np.array(scale_params['min_'])
    max_ = np.array(scale_params['max_'])
    return -1 + (2.0 / (max_ - min_) * (features - min_))

def calculate_image_quality_score(brisque_features):
    '''
    Using a pre-trained SVR model we calculate the quality assessment scores.
    '''
    model = svmutil.svm_load_model(r"deeppixel\iqa\brisque_svm.txt")
    scaled_brisque_features = scaled(brisque_features)
    
    x, idx = svmutil.gen_svm_nodearray(
        scaled_brisque_features,
        isKernel=(model.param.kernel_type == svmutil.PRECOMPUTED))
    
    nr_classifier = 1
    prob_estimates = (svmutil.c_double * nr_classifier)()
    
    return svmutil.libsvm.svm_predict_probability(model, x, prob_estimates)

def imgscore(image,blur="False"):
    '''
    Calculating and returning the image Quality Score along the Blur Scores using Fft
    '''
    gray_image = skimage.color.rgb2gray(image)
    brisque_features = BRISQUE_FEATURE(gray_image)
    score=calculate_image_quality_score(brisque_features)
    res=None
    if blur=="True":          
        gray_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        blur_score,result = detect_blur_fft(gray_image)
        res = "Blurry ({:.4f})" if result else "Not Blurry ({:.4f})"
        res = res.format(blur_score)
    return score,res       
