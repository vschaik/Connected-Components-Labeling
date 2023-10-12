import numpy
import matplotlib.pyplot
from labelers.ccl import label_components


def main():
    images = ["test_data/example_img_1.txt"] #["test_data/example_img_1.txt", "test_data/example_img_2.txt", "test_data/example_img_3.txt"]

    imgs = []
    for arg in images:
        text_file = open(arg, "r")
        lines = text_file.read().split(',')
        vals = [int(line) for line in lines]
        imgs.append(numpy.array(vals))

    for img in imgs:
        s = int(numpy.sqrt(img.shape[0]))
        img = numpy.reshape(img, (s, s))

        labelled_img = label_components(img)

        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(121)

        matplotlib.pyplot.imshow(img, cmap='gray')
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.title("Original Image")

        ax = fig.add_subplot(122)
        matplotlib.pyplot.imshow(labelled_img, cmap='rainbow')
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.title("Labelled Image")

        for (j, i), label in numpy.ndenumerate(labelled_img):
            ax.text(i, j, int(label), ha='center', va='center', fontsize=6)

        matplotlib.pyplot.show()
 

if __name__ == "__main__":
    main()
    exit()
