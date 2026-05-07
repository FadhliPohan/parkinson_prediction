PROFILE_NAME = "without_augment"
DISPLAY_NAME = "Tanpa augmentasi"
RUN_TAG = "tanpa_augmentasi"
AUGMENTATION_ENABLED = False


def build_training_augmentation(seed=None):
    return None


def describe_augmentation_policy():
    return [
        "Tidak ada augmentasi tambahan pada data train.",
        "Model hanya melihat gambar hasil split train asli.",
        "Validation dan testing tetap memakai data asli tanpa augmentasi acak.",
        "Mode ini cocok sebagai baseline komparasi melawan eksperimen dengan augmentasi.",
    ]
