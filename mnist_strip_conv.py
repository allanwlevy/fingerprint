#train on gradient images of Databases/FVC2006/DB_1A and then test on Databases/FVC2006/DB_1B 
#uses a convolutional network


from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, MaxPooling2D, Flatten, ZeroPadding2D,  Activation, Reshape
from keras.layers.merge import Concatenate
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy
from scipy import misc
from PIL import Image


image_size = 28

#same thing as cosine similarity
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

#no clue what this does
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

#some type of loss function i think. figure out why they used this specific one later?
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))



def getXTrain():     

    toReturn = []

    for i in range(700):
        name = 'mnist_stripe_data_smaller/' + str(i) + 'overlayed.png'
        im = misc.imread(name)
        toReturn += [im]
    return np.array(toReturn)

def getXTest():

    toReturn = []

    for i in range(700, 1000):
        name = 'mnist_stripe_data_smaller/' + str(i) + 'overlayed.png'
        im = misc.imread(name)
        toReturn += [im]
    return np.array(toReturn)





def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    toShuffle = list(zip(np.array(pairs), np.array(labels)))
    random.shuffle(toShuffle)
    pairs, labels = zip(*toShuffle)
    return np.array(pairs), np.array(labels)





    


def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    
    seq.add(Conv2D(32, (3, 3),
                 activation='relu',
                 input_shape=input_dim, name="firstConv"))
    seq.add(Dropout(0.25, name="firstDrop"))
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    #seq.load_weights("mnist_conv_model.h5", by_name=True)
    return seq


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return np.mean(labels == (predictions.ravel() > 0.5))

# actual fingerprint part


img_rows, img_cols = image_size, image_size

(X_train, y_train), (X_test, y_test) = mnist.load_data()

''''
X_train = getXTrain()
X_test = getXTest()
y_test = y_train[700:1000]      #not a typo
y_train = y_train[:700]
'''


#some data manipulation to fit the images in the 2DConvolution format
if K.image_dim_ordering() == 'th':    #i'm assuming reshaping the image vectors depends on which backend you use
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#preprocessing: center and normalize
for j in range(len(X_train[0])):
    mean = np.mean(X_train[:,j])
    X_train[:,j] -= mean
    X_test[:,j] -= mean
    norm = np.std(X_train[:,j])
    if norm == 0:
        norm = 1.0
    X_train[:,j] /= norm
    X_test[:,j] /= norm

input_dim = (image_size, image_size, 1)
nb_epoch = 3



# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(10)]
tr_pairs, tr_y = create_pairs(X_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(10)]
te_pairs, te_y = create_pairs(X_test, digit_indices)


base_network = create_base_network(input_dim)

input_a = Input(shape=input_dim)
input_b = Input(shape=input_dim)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

print(base_network.output_shape)
#distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([out_a, out_b])

#borrowed this merge part from https://keras.io/getting-started/functional-api-guide/#shared-layers
concatenated = Concatenate()([processed_a, processed_b])

temp_model = Model(input=[input_a, input_b], output=concatenated)
adam = Adam(lr=0.0001)
temp_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
print("conc")
print(temp_model.output_shape)
print("resa")
reshaped = Reshape((2,128))(concatenated)
print(reshaped.output_shape)
compared = Conv2D(128, (1, 128), activation='relu')(reshaped)
predictions = Dense(1, init='normal', activation='sigmoid')(compared)

model = Model(input=[input_a, input_b], output=predictions)



# train
adam = Adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
          batch_size=128,
          nb_epoch=nb_epoch)


#again, for debugging
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
print("prediction")
print(pred)

print("labels")
print(tr_y)


#computing accuracy
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)


#roc curve for test set
te_fpr, te_tpr, thresh = roc_curve(te_y, pred.ravel())
te_auc = auc(te_fpr, te_tpr)

plt.figure()
lw = 2
plt.plot(te_fpr, te_tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % te_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example - training')
plt.legend(loc="lower right")
plt.show()


print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))