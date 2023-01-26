import pandas as pd
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

meta_file = "data/VGG-Face2/meta/identity_meta.csv"
train_data_root = "data/VGG-Face2/data/train/"
test_data_root = ""


meta = pd.read_csv(meta_file,quotechar='"',skipinitialspace=True)
large_classes = meta[meta['Sample_Num']>=500]['Class_ID'].values.tolist()

large_classes_exist = []
for _dir in large_classes:
    try:
        os.listdir(os.path.join(train_data_root, _dir))
        large_classes_exist.append(_dir)
    except:
        pass


print ("Number of classes with more than 500 samples: \t", len(large_classes_exist))

selected_classes = np.random.choice(large_classes_exist, 110)

lacuna100 = selected_classes[:100]
lacuna10 = selected_classes[:10]


def get_image(image_path, index, resize_to):
    image = Image.open(image_path)
    if resize_to is not None:
        image = image.resize(resize_to)

    image = np.expand_dims(image, axis=0)
    label = np.array([index])
    label = np.expand_dims(label, axis=0)
    return image, label

def make_dataset(data_root, classes, split=False, resize_to=None, num_samples=500, dest="data/lacuna100"):

    try:
        os.makedirs(os.path.join(dest,'train'))
        os.makedirs(os.path.join(dest,'test'))
    except:
        pass

    dataset_train = None
    targets_train = None
    dataset_test = None
    targets_test = None
    for idx, folder in tqdm(enumerate(classes)):
        images = []
        for fil in os.listdir(os.path.join(train_data_root, folder)):
            if fil.endswith('.jpg'):
                images.append(fil)
        selected_images = np.random.choice(images, num_samples)
        if split == True:
            selected_images_train = selected_images[:400]
            selected_images_test = selected_images[400:]
        else:
            selected_images_train = selected_images
            selected_images_test = []


        for img in selected_images_train:
            image_path = os.path.join(data_root, folder, img)
            image, label = get_image(image_path, idx, resize_to)
            if dataset_train is None:
                dataset_train = image
            else:
                dataset_train = np.concatenate((dataset_train, image), axis=0)

            if targets_train is None:
                targets_train = label
            else:
                targets_train = np.concatenate((targets_train,label), axis=0)

        for img in selected_images_test:
            image_path = os.path.join(data_root, folder, img)
            image, label = get_image(image_path, idx, resize_to)
            if dataset_test is None:
                dataset_test = image
            else:
                dataset_test = np.concatenate((dataset_test, image), axis=0)

            if targets_test is None:
                targets_test = label
            else:
                targets_test = np.concatenate((targets_test,label), axis=0)


    if dataset_train is not None and targets_train is not None:
        np.save(os.path.join(dest, "train", 'data.npy'), dataset_train)
        np.save(os.path.join(dest, "train", 'label.npy'), targets_train.reshape(-1))
        print ("OK! train set saved")
        print ("dataset size: {}\tlabels size: {}".format(dataset_train.shape,targets_train.shape))
    else:
        print ("Error! train set did not saved as the sizes are zero")

    if dataset_test is not None and targets_test is not None:
        np.save(os.path.join(dest, "test", 'data.npy'), dataset_test)
        np.save(os.path.join(dest, "test", 'label.npy'), targets_test.reshape(-1))
        print ("OK! test set saved")
        print ("dataset size: {}\tlabels size: {}".format(dataset_test.shape,targets_test.shape))
    else:
        print ("Error! test set did not saved as the sizes are zero")

if __name__ == "__main__":
    make_dataset(train_data_root, lacuna100, split=True, resize_to=(32,32), dest="data/lacuna100")
    make_dataset(train_data_root, lacuna10, split=True, resize_to=(32,32), dest="data/lacuna10")

