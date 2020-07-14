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

def normalized(kernel):
    return kernel / np.sum(kernel)


def gaussian_kernel2d(n, SIGMA):
    Y, X = np.indices((n, n)) - int(n/2)
    gaussian_kernel = 1 / (2 * np.pi * SIGMA ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * SIGMA ** 2)) 
    return normalized(gaussian_kernel)


def local_mean(image, kernel):
    return signal.convolve2d(image, kernel, 'same')


def local_deviation(image, local_mean, kernel):
    sigma = image ** 2
    sigma = signal.convolve2d(sigma, kernel, 'same')
    return np.sqrt(np.abs(local_mean ** 2 - sigma))


def MSCN_coefficients(image, kernel_size=6,SIGMA=7/6):
    C = 1/255
    kernel = gaussian_kernel2d(kernel_size, SIGMA=SIGMA)
    local_mean = signal.convolve2d(image, kernel, 'same')
    local_var = local_deviation(image, local_mean, kernel)
    
    return (image - local_mean) / (local_var + C)


def generalized_gaussian_dist(x, alpha, sigma):
    beta = sigma * np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
    
    coefficient = alpha / (2 * beta() * special.gamma(1 / alpha))
    return coefficient * np.exp(-(np.abs(x) / beta) ** alpha)


def calculate_pair_product_coefficients(mscn_coefficients):
    return collections.OrderedDict({
        'mscn': mscn_coefficients,
        'horizontal': mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:],
        'vertical': mscn_coefficients[:-1, :] * mscn_coefficients[1:, :],
        'main_diagonal': mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:],
        'secondary_diagonal': mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
    })
    
def detect_blur_fft(image, size=60, thresh=20, vis=False):    	
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))
	fft = np.fft.fft2(image)
	fftShift = np.fft.fftshift(fft)

	if vis:
		plt.imshow(image)
		plt.show()

	fftShift[cY - size:cY + size, cX - size:cX + size] = 0
	fftShift = np.fft.ifftshift(fftShift)
	recon = np.fft.ifft2(fftShift)
	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)
	return (mean, mean <= thresh)
    
    
def asymmetric_generalized_gaussian_fit(x):
    def estimate_phi(ALP):
        numerator = special.gamma(2 / ALP) ** 2
        denominator = special.gamma(1 / ALP) * special.gamma(3 / ALP)
        return numerator / denominator
    

    def estimate_r_hat(x):
        size = np.prod(x.shape)
        return (np.sum(np.abs(x)) / size) ** 2 / (np.sum(x ** 2) / size)
    

    def estimate_R_hat(r_hat, gamma):
        num = (gamma ** 3 + 1) * (gamma + 1)
        den = (gamma ** 2 + 1) ** 2
        return r_hat * num / den
    

    def mean_squares_sum(x, filter = lambda z: z == z):
        filtered_values = x[filter(x)]
        squares_sum = np.sum(filtered_values ** 2)
        return squares_sum / ((filtered_values.shape))
    

    def estimate_gamma(x):
        left_squares = mean_squares_sum(x, lambda z: z < 0)
        right_squares = mean_squares_sum(x, lambda z: z >= 0)
        return np.sqrt(left_squares) / np.sqrt(right_squares)
    

    def estimate_alpha(x):
        r_hat = estimate_r_hat(x)
        gamma = estimate_gamma(x)
        R_hat = estimate_R_hat(r_hat, gamma)

        solution = optimize.root(lambda z: estimate_phi(z) - R_hat, [0.2]).x
        return solution[0]
    

    def estimate_sigma(x, alpha, filter = lambda z: z < 0):
        return np.sqrt(mean_squares_sum(x, filter))
    
    
    def estimate_mean(alpha, sigma_l, sigma_r):
        return (sigma_r - sigma_l) * constant * (special.gamma(2 / alpha) / special.gamma(1 / alpha))
    
    
    ALPHA = estimate_alpha(x)
    Sigma_LEFT = estimate_sigma(x, ALPHA, lambda z: z < 0)
    Sigma_RIGHT = estimate_sigma(x, ALPHA, lambda z: z >= 0)
    
    constant = np.sqrt(special.gamma(1 / ALPHA) / special.gamma(3 / ALPHA))
    Mean = estimate_mean(ALPHA, Sigma_LEFT, Sigma_RIGHT)
    return ALPHA, Mean, Sigma_LEFT, Sigma_RIGHT
    

def calculate_brisque_features(image, kernel_size=7, sigma=7/6):
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
    image_stream = request.urlopen(url)
    image=io.imread(image_stream, plugin='pil')
    gray_image=skimage.color.rgb2gray(image)
    return image,gray_image

def plot_histogram(x, label=""):
    n, bins = np.histogram(x.ravel(), bins=50)
    n = n / np.max(n)
    plt.plot(bins[:-1], n, label=label, marker='o')
    
    
def imgread(path):
    image = io.imread(path,as_gray=False)
    gray_image = skimage.color.rgb2gray(image)
    return image,gray_image
    

def imgshow(image,title=""): 
    plt.axis("off")
    plt.title(title) 
    plt.imshow(image)
    plt.show()
    
    
def image_coefficients(gray_image):
    mscn_coefficients = MSCN_coefficients(gray_image, 7, 7/6)
    coefficients = calculate_pair_product_coefficients(mscn_coefficients)
    return coefficients
    

def histogram_analysis(gray_image,x=10,y=5):
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
    with open("trained.pickle", 'rb') as handle:
        scale_params = pickle.load(handle)
    
    min_ = np.array(scale_params['min_'])
    max_ = np.array(scale_params['max_'])
    
    return -1 + (2.0 / (max_ - min_) * (features - min_))

def calculate_image_quality_score(brisque_features):
    model = svmutil.svm_load_model("brisque_svm.txt")
    scaled_brisque_features = scaled(brisque_features)
    
    x, idx = svmutil.gen_svm_nodearray(
        scaled_brisque_features,
        isKernel=(model.param.kernel_type == svmutil.PRECOMPUTED))
    
    nr_classifier = 1
    prob_estimates = (svmutil.c_double * nr_classifier)()
    
    return svmutil.libsvm.svm_predict_probability(model, x, prob_estimates)
    
def imgscore(image,blur=False):
    gray_image = skimage.color.rgb2gray(image)
    brisque_features = BRISQUE_FEATURE(gray_image)
    score=calculate_image_quality_score(brisque_features)
    if blur==True:
        gray_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        blur_score,result = detect_blur_fft(gray_image)
        res = "Blurry ({:.4f})" if result else "Not Blurry ({:.4f})"
        res = res.format(blur_score)
        print("Blur Detection Results n(Scores) : ", res)
        
    print("Assessed Image has a score of : ", score)  
    
    

    