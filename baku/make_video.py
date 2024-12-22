import pickle
import imageio
import os

# Open the pickle file in binary read mode
with open('/home/lgeng/BAKU/expert_demos/libero/libero_success/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_failure_9500_epoch.pkl', 'rb') as file:
    # Load the data from the file
    my_dict = pickle.load(file)
    print(len(my_dict['observations']))
    imagess = [my_dict['observations'][i]['pixels'] for i in range(len(my_dict['observations']))]
    # Create a directory to save the videos
    save_dir = '/home/lgeng/BAKU/expert_demos/libero/libero_success/videos/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_failure_9500_epoch'
    os.makedirs(save_dir, exist_ok=True)

    assert len(imagess) >= 20
    # Save the videos in the created directory
    for i in range(20):
        images = [img.astype('uint8') for img in imagess[i]]
        video_path = os.path.join(save_dir, f'{i}.mp4')
        imageio.mimsave(video_path, images, fps=20)