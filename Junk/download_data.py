from datasets import load_dataset
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm
from PIL import Image
import io
from competition_toolkit.competition_toolkit.dataloader import download_dataset
data_type = "test"
task = 1
dataset = download_dataset(data_type, task, get_dataset=True)
#dataset = load_dataset("sjyhne/mapai_dataset")
masks = dataset['mask']
mask = dataset[0]['mask']
exit(0)
for sample in dataset:
    sample['image'].save(f'sjyhne/mapai_evaluation_data/test/images/{sample["filename"]}')
    sample['mask'].save(f'sjyhne/mapai_evaluation_data/test/masks/{sample["filename"]}')

# Specify the directory to store the dataset


exit(0)

for i, sample in tqdm(enumerate(dataset)):
    image = sample['image']
    mask = sample['mask']
    filename = sample['filename']

    img_filepath = f"data2/images/{filename}"  # Customize the file name as per your requirements
    mask_filepath = f"data2/masks/{filename}"  # Customize the file name as per your requirements

    tifffile.imwrite(img_filepath, image)
    tifffile.imwrite(mask_filepath, mask)