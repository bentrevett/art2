import matplotlib.image as mpimg
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Take in an image, calculate a Markov chain over each channel, generate an image from that Markov chain')
parser.add_argument('--path', type=str, required=True,
                    help='path to the image')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed')
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)

img = mpimg.imread(args.path)

#png images are between 0-1, but jpgs are between 0-255
#as we save our images as png which must have values between 0-1
#we need to normalise values if any are >1

if np.max(img) > 1: 
    img = img/255

def build_markov_from_channel(array):
    """
    Takes in a channel, an array shape imgH x imgW
    Flattens it
    Builds a count of how many times values follow each other
    Normalise the count to turn it into a probability
    Returns the probabilities and the "nodes" (the unique pixel values)
    """
    array = array.reshape(-1)
    assert len(array.shape) == 1

    nodes = np.unique(array)
    counts = np.zeros((len(nodes), len(nodes)))

    for idx, pixel in enumerate(array[:-1]):
        start = array[idx]
        end = array[idx+1]

        x = np.where(nodes==start)[0][0]
        y = np.where(nodes==end)[0][0]
        counts[x][y] += 1

    for i, row in enumerate(counts):
        counts[i] = counts[i]/counts[i].sum()

    return counts, nodes

def build_image_from_markov(probs, nodes, flatten_img_size):
    """
    Uses the markov chain to generate an flattened image
    """
    
    new_c = np.zeros(flatten_img_size)
    
    first_pixel = np.random.choice(nodes)

    new_c[0] = first_pixel

    x = np.where(nodes==first_pixel)[0][0]

    for i, _ in enumerate(new_c[1:]):

        next_pixel = np.random.choice(nodes, p=probs[x])

        new_c[i] = next_pixel

        x = np.where(nodes==next_pixel)[0][0]

    return new_c

#round the values image to the nearest 1dp
img = np.around(img, decimals=1)

flatten_img_size = img.shape[0]*img.shape[1]

new_img = np.zeros((img.shape))

#loop over the channels, and for each construct a markov chain model
#and then generate a new channel using values from that markov chain model

for i in range(img.shape[2]):
    probs, nodes = build_markov_from_channel(img[:,:,i])
    new_c = build_image_from_markov(probs, nodes, flatten_img_size)
    new_img[:,:,i] = new_c.reshape((new_img.shape[0], new_img.shape[1]))

mpimg.imsave('art.png',  new_img)