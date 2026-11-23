# MNIST data

`mnist.npz` contains the original coursework archive's 60,000 training images and 9,999 test images. `mnist-quick.npz` contains its smaller subset of 2,900 training images and 494 test images. Both archives contain `train_images`, `train_labels`, `test_images`, and `test_labels` as unsigned 8-bit NumPy arrays. Images are 28 × 28 grayscale. Labels come from the digit directories. Packaging the images as arrays preserves their pixel values and removes file metadata.

MNIST is a derivative of NIST datasets. Yann LeCun and Corinna Cortes hold copyright in MNIST, which is distributed under [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The same terms apply to the bundled data and the MNIST test image in the examples folder. [MNIST project](https://yann.lecun.com/exdb/mnist/) · [License attribution](https://keras.io/api/datasets/mnist/)
