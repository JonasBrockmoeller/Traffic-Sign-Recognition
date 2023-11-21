import pickle
from pandas.io.parsers import read_csv
import np
from matplotlib import pyplot


def show_example_img_per_each_class(X_train, classes, class_indices, class_counts, sign_names):
    # Visualizations of image datasets for each class
    for c, c_i, c_count in zip(classes, class_indices, class_counts):
        print(c, ". Class : ", sign_names[c])
        fig = pyplot.figure(figsize=(3, 1))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        for i in range(3):
            axis = fig.add_subplot(1, 3, i + 1, xticks=[], yticks=[])
            random_indices = np.random.randint(c_i, c_i + c_count, 10)
            axis.imshow(X_train[random_indices[i], :, :, :])
            # axis.text(0, 0, '{}: {}'.format(c, sign_names[c]), color='k',backgroundcolor='c', fontsize=8)

        pyplot.show()


def basic_text_analysis(X_train, y_train):
    n_train = X_train.shape[0]
    image_shape = X_train[0].shape
    classes, class_indices, class_counts = np.unique(y_train, return_index=True, return_counts=True)
    n_classes = len(class_counts)
    print("Number of training examples =", n_train)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)
    return classes, class_indices, class_counts


def load_data():
    training_file = './Dataset/train.p'

    sign_names = read_csv("./Dataset/signname.csv").values[:, 1]

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    return X_train, y_train, sign_names


def run():
    print("Letse gooooo")
    X_train, y_train, sign_names = load_data()
    classes, class_indices, class_counts = basic_text_analysis(X_train, y_train)
    show_example_img_per_each_class(X_train, classes[:14], class_indices[:14], class_counts[:14], sign_names)


if __name__ == '__main__':
    run()