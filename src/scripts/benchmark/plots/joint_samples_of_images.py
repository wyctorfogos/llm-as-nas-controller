import os
import matplotlib.pyplot as plt
import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils
import utils.load_local_variables as load_local_variables

def load_image(image_name_folder_path: str):
    try:
        loaded_image = cv2.imread(image_name_folder_path)
        if loaded_image is None:
            raise ValueError("Image not found")
        return loaded_image
    except Exception as e:
        print(f"Error loading image {image_name_folder_path}. Error: {e}")
        return None

def main(image_folder_path:str, dataset_name:str, wanted_image_list: list):
    # Create a figure for the plot
    f, axarr = plt.subplots(nrows=2, ncols=4, figsize=(12, 8))  # Adjust the grid size to match the number of images
    axarr = axarr.ravel()  # Flatten the array of axes for easy indexing

    for i, item in enumerate(wanted_image_list):
        image_class = item["class"]
        image_name = item["image"]
        img = load_image(os.path.join(image_folder_path, image_name))
        
        if img is not None:
            axarr[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper display in matplotlib
            axarr[i].set_title(image_class)
            axarr[i].axis('off')  # Hide axis for a cleaner image display

    plt.tight_layout()  # To adjust spacing between subplots
    plt.savefig(f"./images/samples_of_images_of_dataset_{dataset_name}.png")
    plt.show()

if __name__ == "__main__":
    # Dados das imagens a serem juntadas
    local_variables = load_local_variables.get_env_variables()
    dataset_folder_name = local_variables["dataset_folder_name"]
    dataset_folder_path = "/data/ISIC-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input" # local_variables["dataset_folder_path"]
    
    # wanted_image_list = [
    #     {"class":"BCC", "image":"PAT_46_881_939.png"}, 
    #     {"class":"ACK", "image":"PAT_705_4015_413.png"}, 
    #     {"class":"SCC", "image":"PAT_380_1540_959.png"},
    #     {"class":"SEK", "image":"PAT_107_160_609.png"}, 
    #     {"class":"NEV", "image":"PAT_793_1512_327.png"}, 
    #     {"class":"MEL", "image":"PAT_680_1289_182.png"}
    # ]
    wanted_image_list = [
        {"class":"NV", "image":"ISIC_0000000.jpg"}, 
        {"class":"MEL", "image":"ISIC_0000002.jpg"}, 
        {"class":"BKL", "image":"ISIC_0010491.jpg"},
        {"class":"VASC", "image":"ISIC_0024370.jpg"}, 
        {"class":"SCC", "image":"ISIC_0024372.jpg"}, 
        {"class":"BCC", "image": "ISIC_0024403.jpg"},
        {"class":"AK", "image": "ISIC_0024468.jpg"},
        {"class":"DF", "image":"ISIC_0024386.jpg"}
    ]

    main(dataset_name=dataset_folder_name, image_folder_path=dataset_folder_path, wanted_image_list=wanted_image_list)
