import matplotlib.pyplot as plt

def display_sample(image,label,predicted=999):
    #Reshape the 768 values to a 28x28 image
    plt.title('Label: %d, Predicted: %d' % (label, predicted))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
