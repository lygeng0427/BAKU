import pickle as pkl

# Function to merge two dictionaries
def merge_dicts(dict1, dict2):
    merged_dict = dict1.copy()  # Start with dict1's keys and values
    for key, value in dict2.items():
        if key == "observations" or key == "actions" or key == "rewards":
            assert type(value) == list, "observations should be a list"
            merged_dict[key].extend(value)
        elif key == "task_emb":
            continue
    return merged_dict

# Load pickle files
with open('/home/lgeng/BAKU/expert_demos/libero/libero_failure/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_bad_checkpoints.pkl', 'rb') as f:
    data1 = pkl.load(f)

with open('/home/lgeng/BAKU/expert_demos/libero/libero_failure/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_noise.pkl', 'rb') as f:
    data2 = pkl.load(f)

# Merge the data
merged_data = merge_dicts(data1, data2)

# Save the merged data to a new pickle file
with open('/home/lgeng/BAKU/expert_demos/libero/libero_single_task_success_failure/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_mixed.pkl', 'wb') as f:
    pkl.dump(merged_data, f)
