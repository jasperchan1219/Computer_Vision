import numpy as np
from PIL import Image
from tqdm import tqdm
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from scipy.spatial.distance import cdist

CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}

########################################
###### FEATURE UTILS              ######
###### use TINY_IMAGE as features ######
########################################

step_n = 5

###### Step 1-a
def get_tiny_images(img_paths):
    '''
    Input : 
        img_paths (N) : list of string of image paths
    Output :
        tiny_img_feats (N, d) : ndarray of resized and then vectorized 
                                tiny images
    NOTE :
        1. N is the total number of images
        2. if the images are resized to 16x16, d would be 256
    '''
    
    #################################################################
    # TODO:                                                         #
    # To build a tiny image feature, you can follow below steps:    #
    #    1. simply resize the original image to a very small        #
    #       square resolution, e.g. 16x16. You can either resize    #
    #       the images to square while ignoring their aspect ratio  #
    #       or you can first crop the center square portion out of  #
    #       each image.                                             #
    #    2. flatten and normalize the resized image.                #
    #################################################################

    tiny_img_feats = []
    for path in tqdm(img_paths):
        img = Image.open(path)
        img = img.resize((16, 16))
        img = np.array(img)
        img = img.flatten()
        img = img / np.linalg.norm(img)
        tiny_img_feats.append(img)
    
    #################################################################
    #                        END OF YOUR CODE                       #
    #################################################################

    return tiny_img_feats

#########################################
###### FEATURE UTILS               ######
###### use BAG_OF_SIFT as features ######
#########################################

###### Step 1-b-1
def build_vocabulary(img_paths, vocab_size=250):
    '''
    Input : 
        img_paths (N) : list of string of image paths (training)
        vocab_size : number of clusters desired
    Output :
        vocab (vocab_size, sift_d) : ndarray of clusters centers of k-means
    NOTE :
        1. sift_d is 128
        2. vocab_size is up to you, larger value will works better (to a point) 
           but be slower to compute, you can set vocab_size in p1.py
    '''
    
    ##################################################################################
    # TODO:                                                                          #
    # To build vocabularies from training images, you can follow below steps:        #
    #   1. create one list to collect features                                       #
    #   2. for each loaded image, get its 128-dim SIFT features (descriptors)        #
    #      and append them to this list                                              #
    #   3. perform k-means clustering on these tens of thousands of SIFT features    #
    # The resulting centroids are now your visual word vocabulary                    #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful functions                                                          #
    #   Function : dsift(img, step=[x, x], fast=True)                                #
    #   Function : kmeans(feats, num_centers=vocab_size)                             #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful tips if it takes too long time                                     #
    #   1. you don't necessarily need to perform SIFT on all images, although it     #
    #      would be better to do so                                                  #
    #   2. you can randomly sample the descriptors from each image to save memory    #
    #      and speed up the clustering, which means you don't have to get as many    #
    #      SIFT features as you will in get_bags_of_sift(), because you're only      #
    #      trying to get a representative sample here                                #
    #   3. the default step size in dsift() is [1, 1], which works better but        #
    #      usually become very slow, you can use larger step size to speed up        #
    #      without sacrificing too much performance                                  #
    #   4. we recommend debugging with the 'fast' parameter in dsift(), this         #
    #      approximate version of SIFT is about 20 times faster to compute           #
    # You are welcome to use your own SIFT feature                                   #
    ##################################################################################
    
    features = []
    for path in tqdm(img_paths):
        img = Image.open(path)
        #normalize
        img = np.array(img) / 255
        _keypoints, descriptors = dsift(img, step=[step_n, step_n], fast=True) 
        features.append(descriptors)
    features = np.vstack(features)
    vocab = kmeans(features.astype(np.float32),num_centers=vocab_size)
    

    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################
    
    return vocab
    # return None

###### Step 1-b-2
def get_bags_of_sifts(img_paths, vocab):
    '''
    Input :
        img_paths (N) : list of string of image paths
        vocab (vocab_size, sift_d) : ndarray of clusters centers of k-means
    Output :
        img_feats (N, d) : ndarray of feature of images, each row represent
                           a feature of an image, which is a normalized histogram
                           of vocabularies (cluster centers) on this image
    NOTE :
        1. d is vocab_size here
    '''

    ############################################################################
    # TODO:                                                                    #
    # To get bag of SIFT words (centroids) of each image, you can follow below #
    # steps:                                                                   #
    #   1. for each loaded image, get its 128-dim SIFT features (descriptors)  #
    #      in the same way you did in build_vocabulary()                       #
    #   2. calculate the distances between these features and cluster centers  #
    #   3. assign each local feature to its nearest cluster center             #
    #   4. build a histogram indicating how many times each cluster presents   #
    #   5. normalize the histogram by number of features, since each image     #
    #      may be different                                                    #
    # These histograms are now the bag-of-sift feature of images               #
    #                                                                          #
    # NOTE:                                                                    #
    # Some useful functions                                                    #
    #   Function : dsift(img, step=[x, x], fast=True)                          #
    #   Function : cdist(feats, vocab)                                         #
    #                                                                          #
    # NOTE:                                                                    #
    #   1. we recommend first completing function 'build_vocabulary()'         #
    ############################################################################

    img_feats = []
    for path in tqdm(img_paths):
        img = Image.open(path)
        img = np.array(img)
        #normalize
        img = img / 255
        _keypoints, descriptors = dsift(img, step=[step_n, step_n], fast=True)
        distances = cdist(descriptors, vocab)
        labels = np.argmin(distances, axis=1)
        hist, _ = np.histogram(labels, bins=len(vocab), range=(0, len(vocab)))
        hist = hist / len(descriptors)
        img_feats.append(hist)
    

    ############################################################################
    #                                END OF YOUR CODE                          #
    ############################################################################
    
    return img_feats

################################################
###### CLASSIFIER UTILS                   ######
###### use NEAREST_NEIGHBOR as classifier ######
################################################

###### Step 2
def nearest_neighbor_classify(train_img_feats, train_labels, test_img_feats):
    '''
    Input : 
        train_img_feats (N, d) : ndarray of feature of training images
        train_labels (N) : list of string of ground truth category for each 
                           training image
        test_img_feats (M, d) : ndarray of feature of testing images
    Output :
        test_predicts (M) : list of string of predict category for each 
                            testing image
    NOTE:
        1. d is the dimension of the feature representation, depending on using
           'tiny_image' or 'bag_of_sift'
        2. N is the total number of training images
        3. M is the total number of testing images
    '''

    CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
           'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
           'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

    CAT2ID = {v: k for k, v in enumerate(CAT)}

    ###########################################################################
    # TODO:                                                                   #
    # KNN predict the category for every testing image by finding the         #
    # training image with most similar (nearest) features, you can follow     #
    # below steps:                                                            #
    #   1. calculate the distance between training and testing features       #
    #   2. for each testing feature, select its k-nearest training features   #
    #   3. get these k training features' label id and vote for the final id  #
    # Remember to convert final id's type back to string, you can use CAT     #
    # and CAT2ID for conversion                                               #
    #                                                                         #
    # NOTE:                                                                   #
    # Some useful functions                                                   #
    #   Function : cdist(feats, feats)                                        #
    #                                                                         #
    # NOTE:                                                                   #
    #   1. instead of 1 nearest neighbor, you can vote based on k nearest     #
    #      neighbors which may increase the performance                       #
    #   2. hint: use 'minkowski' metric for cdist() and use a smaller 'p' may #
    #      work better, or you can also try different metrics for cdist()     #
    ###########################################################################

    k=6
    test_predicts = []
    #normalize
    train_img_feats = np.array(train_img_feats)
    train_img_feats = train_img_feats / np.linalg.norm(train_img_feats)
    test_img_feats = np.array(test_img_feats)
    test_img_feats = test_img_feats / np.linalg.norm(test_img_feats)
    #calculate distance
    distances = cdist(test_img_feats, train_img_feats, metric='minkowski', p=0.5)
    #get k nearest
    for i in range(len(test_img_feats)):
        k_nearest_labels = np.argpartition(distances[i], k)[:k]
        #vote
        vote = np.zeros(len(CAT))
        for label in k_nearest_labels:
            vote[CAT2ID[train_labels[label]]] += 1
        predict = np.argmax(vote)
        test_predicts.append(CAT[predict])
    
    ###########################################################################
    #                               END OF YOUR CODE                          #
    ###########################################################################
    
    return test_predicts
