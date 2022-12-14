import random
import numpy as np
import torch
import openslide


def one_hot_encode_labels(labels):
    """Takes integer labels with values [0-9] and converts them to one hot encoded labels.
    Uses sklearn instead of keras!
    """
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()
    labels_transformed = mlb.fit_transform(labels)
    print("Original classes are : " + str(mlb.classes_))
    print("One-hot-encoded classes are : " + str(np.unique(labels_transformed, axis=0)))

    return labels_transformed


def load_data_paths(subtypes: list, path_to_slide_info: str):
    """loads all wsi slides at a given location dependent on their datapaths and a list of disease subtypes.
    Returns only list of names/paths and list of labels"""
    data = []
    labels = []

    label = 0
    for subtype in subtypes:
        list_id = np.loadtxt(
            path_to_slide_info + f"{subtype}.txt", delimiter=",", dtype="str"
        ).tolist()

        # for slide in range(len(list_id[:10])):
        for slide in range(len(list_id)):
            data.append(list_id.pop(0))
            labels.append(label)
        label += 1

    return data, labels


def get_wsi_info(wsi, label: int, number_of_patches: int, patch_info_path: str):
    slideID = wsi

    positions = np.loadtxt(
        f"{patch_info_path}/{slideID}.csv", delimiter=",", dtype="int"
    )  # Load patch (40x) locations
    random.shuffle(positions)  # shuffle patches order of 1 wsi
    if positions.shape[0] > number_of_patches:
        patches = positions[0:number_of_patches, :]
    else:
        patches = positions

    return [patches, slideID, label]


def patch_wsi(wsi_info, transform, path_to_data: str, magnification: str = "40x"):
    patch_size = 224

    patch_amount = wsi_info[0]
    slide_ID = wsi_info[1]
    label = wsi_info[2]

    svs = openslide.OpenSlide(f"{path_to_data}/{slide_ID}.svs")  # Load 1 wsi

    patches = torch.empty(
        len(patch_amount), 3, patch_size, patch_size, dtype=torch.float
    )

    for i, pos in enumerate(patch_amount):
        if magnification == "40x":  # get patch(224 x 224)
            img = svs.read_region(
                (pos[0], pos[1]), 0, (patch_size, patch_size)
            ).convert("RGB")
        elif magnification == "20x":  # get patch(448 x 448)
            img = svs.read_region(
                (pos[0] - (int(patch_size / 2)), pos[1] - (int(patch_size / 2))),
                0,
                (patch_size * 2, patch_size * 2),
            ).convert("RGB")
        elif magnification == "10x":  # get patch(224 x 224)
            img = svs.read_region(
                (
                    pos[0] - (int(patch_size * 3 / 2)),
                    pos[1] - (int(patch_size * 3 / 2)),
                ),
                1,
                (patch_size, patch_size),
            ).convert("RGB")
        elif magnification == "5x":  # get patch(448 x 448)
            img = svs.read_region(
                (
                    pos[0] - (int(patch_size * 7 / 2)),
                    pos[1] - (int(patch_size * 7 / 2)),
                ),
                1,
                (patch_size * 2, patch_size * 2),
            ).convert("RGB")

        img = transform(img)
        patches[i] = img
    svs.close()
    return patches, label


def convert_to_tile_dataset(wsis, labels):
    """Convert data and label pairs into combined format
    Inputs:
        a list of data/bags and a list of (bag)-labels
    Outputs:
        Returns a dataset (list) containing (stacked tiled instance data, bag label)
    """
    dataset = []
    tmp_y = []
    for wsi, wsi_label in zip(wsis, labels):
        number_of_patches = wsi.shape[0]
        tmp_y.append(np.tile(wsi_label, reps=number_of_patches).tolist())

    tmp_x = torch.cat(wsis, dim=0)
    del wsis
    tmp_y = np.concatenate(tmp_y)
    assert tmp_x.shape[0] == len(tmp_y)

    for index, (patch, patch_label) in enumerate(zip(tmp_x, tmp_y)):
        dataset.append((patch, patch_label))

    del tmp_x, tmp_y

    return dataset


def feature_extract_bag(feature_extractor, data):
    data_features = []
    feature_extractor.cuda()
    feature_extractor.eval()
    for bag in data:
        bag_features = []
        for img in bag:
            with torch.no_grad():
                img_features = feature_extractor(img.unsqueeze(dim=0).cuda())
            bag_features.append(img_features.cpu().numpy())
        data_features.append(np.concatenate(bag_features))
    return data_features


def convert_to_bag_dataset(data, labels):
    """
    Inputs:
        a list of bags and a list of bag-labels
    Outputs:
        Returns a dataset (list) containing (stacked tiled instance data, bag label)
    """
    dataset = []

    for index, (bag, bag_label) in enumerate(zip(data, labels)):
        bag_data = np.asarray(bag, dtype="float32")
        bag_label = np.asarray(bag_label, dtype="float32")
        dataset.append((bag_data, bag_label))

    return dataset
