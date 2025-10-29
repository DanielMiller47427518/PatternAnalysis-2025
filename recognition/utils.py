import nibabel as nib
import numpy as np
from tqdm import tqdm
from skimage.transform import resize

SEG_TRAIN_PATH = "hipmri_data/keras_slices_data/keras_slices_seg_train"
IMAGE_TRAIN_PATH = "hipmri_data/keras_slices_data/keras_slices_train"

SEG_TEST_PATH = "hipmri_data/keras_slices_data/keras_slices_seg_test"
IMAGE_TEST_PATH = "hipmri_data/keras_slices_data/keras_slices_test"

SEG_VAL_PATH = "hipmri_data/keras_slices_data/keras_slices_seg_validate"
IMAGE_VAL_PATH = "hipmri_data/keras_slices_data/keras_slices_validate"

BATCH_SIZE = 16

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128

def to_channels(arr: np.ndarray , dtype = np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels :
        c = int(c)
        res [..., c:c+1][arr == c] = 1
    return res

# load medical image functions
def load_data_2D(imageNames, normImage=False, dtype=np.float32, getAffines=False, early_stop=False, image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    '''
    Load medical image data from names , cases list provided into a list for each .

    This function pre - allocates 4D arrays for conv2d to avoid excessive memory &
    usage .

    normImage : bool ( normalise the image 0.0 -1.0)
    early_stop : Stop loading pre - maturely , leaves arrays mostly empty , for quick &
    loading and testing scripts .
    '''
    affines = []

    # # get fixed size
    num = len(imageNames)
    # first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    # if len(first_case.shape) == 3:
    #     first_case = first_case [:,:,0] # sometimes extra dims , remove

    # rows, cols = first_case.shape
    # set size of images based on provided target shape
    images = np.zeros((num, image_shape[0], image_shape[1]) , dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching = 'unchanged') # read disk only
        affine = niftiImage.affine
        
        if len(inImage.shape) == 3:
            inImage = inImage [:,:,0] # sometimes extra dims in HipMRI_study data

        inImage = inImage.astype(dtype)

        # not all images are correct dimensions, resize
        inImage = resize(inImage, (image_shape[0], image_shape[1]), order=1, preserve_range=True)

        if normImage :
            #~ inImage = inImage / np. linalg . norm ( inImage )
            #~ inImage = 255. * inImage / inImage . max ()
            inImage = (inImage - inImage.mean()) / inImage.std()
        
        images [i,:,:] = inImage

        affines . append ( affine )
        if i > 20 and early_stop :
            break

    if getAffines :
        return images , affines
    else :
        return images