from api_keys import OPENAI_KEY
from openai import OpenAI
import base64
import os
import cv2
import time
import pickle

MODEL = "gpt-4o"

TASK_CONFIG = {
    "open_the_bottom_drawer_of_the_cabinet": {
        "prompt": """
        As the timesteps progress, does the robotic arm open the BOTTOM drawer of the cabinet in the LAST timestep? 
        Please respond with only 'Yes' or 'No'. 
        """, 
        "camera": "pixels", # "pixels" OR "pixels_egocentric"
    },
    "open_the_top_drawer_of_the_cabinet": {
        "prompt": """
        As the timesteps progress, does the robotic arm open the TOP drawer of the cabinet in the LAST timestep? 
        Please respond with only 'Yes' or 'No'. 
        """, 
        "camera": "pixels", # "pixels" OR "pixels_egocentric"
    },
    "open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it": {
        "prompt": """
        As the timesteps progress, does the robotic arm open the TOP drawer of the cabinet and put the bowl in it in the LAST timestep? 
        Please respond with only 'Yes' or 'No'. 
        """, 
        "camera": "pixels", # "pixels" OR "pixels_egocentric"
    },
    "put_the_black_bowl_on_the_plate": {
        "prompt": """
        As the timesteps progress, does the robotic arm put the black bowl on the plate in the LAST timestep?
        Please respond with only 'Yes' or 'No'. 
        """,
        "camera": "pixels", # "pixels" OR "pixels_egocentric"
    },
    "put_the_black_bowl_on_top_of_the_cabinet": {
        "prompt": """
        As the timesteps progress, does the robotic arm put the black bowl on TOP of the cabinet in the LAST timestep?
        Please respond with only 'Yes' or 'No'. 
        """,
        "camera": "pixels", # "pixels" OR "pixels_egocentric"
    },
}

# TASK_CONFIG_LIKELIHOOD = {
#     "open_the_bottom_drawer_of_the_cabinet": {
#         "prompt": """
#         As timesteps progresses, what is the likelihood that the robotic arm will open the bottom drawer of the cabinet in the LAST timestamp? 
#         Please respond only with one of the following five likelihood levels: 'Very unlikely', 'Unlikely', 'Fair', 'Likely', or 'Very likely'.
#         """, 
#         "camera": "pixels", # "pixels" OR "pixels_egocentric"
#     },
# }

TASK_CONFIG_LIKELIHOOD = {
    "open_the_bottom_drawer_of_the_cabinet": {
        "prompt": """
        As timesteps progresses, is it likely that the robotic arm will FINALLY open the bottom drawer of the cabinet? 
        Please respond with only 'Yes' or 'No'. 
        """, 
        "camera": "pixels", # "pixels" OR "pixels_egocentric"
    },
}

def get_success_rate(responses, groung_truth="Yes"):
    success_rate = sum([1 for response in responses if groung_truth in response]) / len(responses)
    return success_rate

class OpenAIClient:
    def __init__(self, task, pkl_data_path, img_save_dir, every_n=5):
        self.client = OpenAI(api_key=OPENAI_KEY)
        self.task = task
        self.pkl_data_path = pkl_data_path
        self._get_traj_num()

        self.prompt = TASK_CONFIG[task]["prompt"]
        self.camera = TASK_CONFIG[task]["camera"]

        self.likelihood_prompt = TASK_CONFIG_LIKELIHOOD[task]["prompt"]
        self.likelihood_camera = TASK_CONFIG_LIKELIHOOD[task]["camera"]

        self.img_save_dir = img_save_dir
        os.makedirs(self.img_save_dir, exist_ok=True)
        self.every_n = every_n

    def _get_traj_num(self):
        with open(self.pkl_data_path, 'rb') as file:
            data_dict = pickle.load(file)
            traj_num = len(data_dict['observations'])
        self.n_traj = traj_num

    def get_image_list(self, traj_idx):
        # latest_img_folder = sorted(os.listdir(self.img_save_dir))[-1]
        with open(self.pkl_data_path, 'rb') as file:
            data_dict = pickle.load(file)
            assert traj_idx < len(data_dict['observations'])
            
            if self.camera in ['pixels', 'pixels_egocentric']:
                traj = data_dict['observations'][traj_idx][self.camera]
            else:
                raise ValueError(f"Invalid camera type: {self.camera}")
        return traj
    
    def get_encoded_images(self, traj_idx):
        img_list = self.get_image_list(traj_idx)
        # take every every_n images, including the last image, from the image list
        img_list = img_list[::-self.every_n][::-1]
        # # save these images
        # for idx, img in enumerate(img_list):
        #     cv2.imwrite(f"{self.img_save_dir}/traj_{traj_idx}_timestep_{idx}.png", img)
        # breakpoint()
        encoded_images = []
        for img in img_list:
            # Convert the image to a byte array
            _, buffer = cv2.imencode('.jpg', img)
            img_bytes = buffer.tobytes()
            # breakpoint()
            # Encode the byte array to base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            # Append the encoded image to the list
            encoded_images.append(img_base64)

        return encoded_images

    def get_image_prompts(self, encoded_images):
        image_prompts = []
        for idx, encoded_image in enumerate(encoded_images):
            image_prompts.append(
                {"type": "text", 
                 "text": f"The following is an image taken at timestep {idx}"
                }
            )
            image_prompts.append(
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}"}
                }
            )
        return image_prompts
    
    def get_response(self, traj_idx):
        encoded_images = self.get_encoded_images(traj_idx)
        image_prompts = self.get_image_prompts(encoded_images)

        completion = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": self.prompt}]
                },
                {
                    "role": "user",
                    "content": image_prompts
                }
            ],
            temperature=0.0,
        )

        response = completion.choices[0].message.content

        return response

    def get_responses_all_trajs(self):
        responses = []
        for traj_idx in range(self.n_traj):
            response = self.get_response(traj_idx)
            responses.append(response)
            print(response)
        return responses

    def get_likelihood_responses(self, traj_idx):
        encoded_images = self.get_encoded_images(traj_idx)
        image_prompts = self.get_image_prompts(encoded_images)
        responses = []
        for i in range(1, len(encoded_images)):
            completion = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": self.likelihood_prompt}]
                    },
                    {
                        "role": "user",
                        "content": image_prompts[:2*i]
                    }
                ],
                temperature=0.0,
            )

            response = completion.choices[0].message.content
            print(response)
            responses.append(response)
        
        return responses
    
    def get_likelihood_responses_all_trajs(self):
        responses = []
        for traj_idx in range(self.n_traj):
            response = self.get_likelihood_responses(traj_idx)
            responses.append(response)
        return responses
    
if __name__ == "__main__":
    task = "open_the_bottom_drawer_of_the_cabinet"
    # task = "open_the_top_drawer_of_the_cabinet"
    # task = "open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it"
    # task = "put_the_black_bowl_on_the_plate"
    # task = "put_the_black_bowl_on_top_of_the_cabinet"
    # task = "open_the_bottom_drawer_of_the_cabinet"

    pkl_data_path = '/home/lgeng/BAKU/expert_demos/libero/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.pkl'
    # pkl_data_path = '/home/lgeng/BAKU/expert_demos/libero/libero_90/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet.pkl'
    # pkl_data_path = '/home/lgeng/BAKU/expert_demos/libero/libero_90/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it.pkl'
    # pkl_data_path = '/home/lgeng/BAKU/expert_demos/libero/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.pkl'
    # pkl_data_path = '/home/lgeng/BAKU/expert_demos/libero/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet.pkl'
    # pkl_data_path = '/home/lgeng/BAKU/expert_demos/libero/libero_single_task_success_failure/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_eval_failure_big.pkl'

    img_save_dir = '/home/lgeng/BAKU/baku/libero/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet'
    # img_save_dir = '/home/lgeng/BAKU/baku/libero/libero_single_task_success_failure/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_eval_failure_big' # for debug purposes

    # ground_truth = "Yes"
    # ground_truth = "No"

    client = OpenAIClient(task, pkl_data_path, img_save_dir)

    traj_idx = 0

    likelihood_responses = client.get_likelihood_responses(traj_idx)
    print(likelihood_responses)
    # responses = client.get_responses_all_trajs()
    # prediction_success_rate = get_success_rate(responses, ground_truth)
    # print(f"Prediction success rate: {prediction_success_rate}")
    