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

        for slide in range(len(list_id[:5])):
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


def patch_wsi(args, wsi_info, transform, path_to_data: str, magnification: str = "40x"):
    patch_size = 224

    patch_amount = wsi_info[0]
    slide_ID = wsi_info[1]
    label = wsi_info[2]

    svs = openslide.OpenSlide(f"{path_to_data}/{slide_ID}.svs")  # Load 1 wsi

    if (
        args.baseline == "clam"
    ):  # perform wsi patching to make MIL-based bags with one label
        patches = torch.empty(
            len(patch_amount), 3, patch_size, patch_size, dtype=torch.float
        )
    else:
        X, y = [], []

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
        if args.baseline == "clam":
            patches[i] = img
        else:
            X.append(img)
            y.append(label)

    if args.baseline == "clam":
        return patches, label
    else:
        return X, y


def convert_to_tile_dataset(wsis, labels):
    """Convert data and label pairs into combined format
    Inputs:
        a list of data/bags and a list of (bag)-labels
    Outputs:
        Returns a dataset (list) containing (stacked tiled instance data, bag label)
    """
    dataset = []

    for index, (wsi, wsi_label) in enumerate(zip(wsis, labels)):
        dataset.append((wsi, wsi_label))

    return dataset


class PatchDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset, wsi_path: str, magnification: str = "40x", transform=None
    ):

        self.transform = transform
        self.magnification = magnification

        self.wsi_collection = []
        for wsi in wsi_collection:  # wsi_collection = [ slideID, lable ]
            slideID = wsi[0]
            label = wsi[1]

            positions = np.loadtxt(
                f"{csv_PATH}/{slideID}.csv", delimiter=",", dtype="int"
            )  # Load patch (40x) locations

            random.shuffle(positions)  # shuffle patches order of 1 wsi

            if pos.shape[0] > number_of_patches:
                patches = pos[0:number_of_patches, :]
            else:
                patches = pos

            self.wsi_collection.append([patches, slideID, label])

        random.shuffle(self.wsi_collection)  # shuffle all wsi order

    def __len__(self):
        return len(self.bag_list)

    def __getitem__(self, idx):
        patches = self.bag_list[idx][0]
        slideID = self.bag_list[idx][1]
        label = self.bag_list[idx][2]

        patch_size = 224

        svs_list = os.listdir(f"{DATA_PATH}")
        svs_fn = [s for s in svs_list if slideID in s]
        svs = openslide.OpenSlide(f"{DATA_PATH}/{svs_fn[0]}")  # Load 1 wsi

        bag = torch.empty(len(patches), 3, 224, 224, dtype=torch.float)

        for i, pos in enumerate(patches):
            if self.mag == "40x":  # get patch(224 x 224)
                img = svs.read_region(
                    (pos[0], pos[1]), 0, (patch_size, patch_size)
                ).convert("RGB")
            elif self.mag == "20x":  # get patch(448 x 448)
                img = svs.read_region(
                    (pos[0] - (int(patch_size / 2)), pos[1] - (int(patch_size / 2))),
                    0,
                    (patch_size * 2, patch_size * 2),
                ).convert("RGB")
            elif self.mag == "10x":  # get patch(224 x 224)
                img = svs.read_region(
                    (
                        pos[0] - (int(patch_size * 3 / 2)),
                        pos[1] - (int(patch_size * 3 / 2)),
                    ),
                    1,
                    (patch_size, patch_size),
                ).convert("RGB")
            elif self.mag == "5x":  # get patch(448 x 448)
                img = svs.read_region(
                    (
                        pos[0] - (int(patch_size * 7 / 2)),
                        pos[1] - (int(patch_size * 7 / 2)),
                    ),
                    1,
                    (patch_size * 2, patch_size * 2),
                ).convert("RGB")
            img = self.transform(img)  # resize patch to (224 x 224)
            bag[i] = img

        return bag, slideID, label
