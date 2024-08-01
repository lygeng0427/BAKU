import pickle
import imageio
import os

# Open the pickle file in binary read mode
with open('/home/lgeng/BAKU/expert_demos/libero/libero_single_task_success_failure/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_failure_big.pkl', 'rb') as file:
    # Load the data from the file
    my_dict = pickle.load(file)
    print(len(my_dict['observations']))
    imagess = [my_dict['observations'][i]['pixels'] for i in range(len(my_dict['observations']))]
    # Create a directory to save the videos
    save_dir = '/home/lgeng/BAKU/expert_demos/libero/libero_single_task_success_failure/videos/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_failure_big'
    os.makedirs(save_dir, exist_ok=True)

    # Save the videos in the created directory
    for i in range(len(imagess)):
        images = [img.astype('uint8') for img in imagess[i]]
        video_path = os.path.join(save_dir, f'{i}.mp4')
        imageio.mimsave(video_path, images, fps=20)