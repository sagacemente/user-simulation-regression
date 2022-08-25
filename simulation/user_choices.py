import pandas as pd
import random
import numpy as np


def load_image_features():
    path = '../data/DEF_images_data.xlsx'
    image_df = pd.read_excel(path, index_col=0)
    return image_df


def simulate_user_choices(user, grid_sequence):
    feature_df = load_image_features()

    image_grid = grid_sequence.get_image_grid_sequence()
    grid_info_dict = {}
    for key, grid in image_grid.items():
        grid_info_dict[key] = []
        for image in grid:
            image_info = {}
            image_info_from_table = feature_df.loc[image.split("\\")[-1]]
            image_info['topic'] = image_info_from_table['topic']
            image_info['true_fake'] = image_info_from_table['t_f']
            image_info['authenticity'] = image_info_from_table['authenticity']
            image_info['virality'] = image_info_from_table['virality']
            grid_info_dict[key].append(image_info)

    selected_images = {}

    for key, grid in image_grid.items():
        matching_image_ids = {}
        image_info = grid_info_dict[key]
        for idx, image in enumerate(image_info):
            matching_image_ids[idx] = 0
            # if image['topic'] in user.topic_preferences:
            #    matching_image_ids[idx] += 1
            if image['topic'] in user.topic_knowledge:
                matching_image_ids[idx] += 1
            # if user.critical_thinking == True and image['true_fake'] == 'T':
            #    matching_image_ids[idx] += 1
            # if user.critical_thinking == False and image['true_fake'] == 'F':
            #    matching_image_ids[idx] += 1
            if user.recognize_manipulation == True and image['true_fake'] == 'T':
                matching_image_ids[idx] += 1
            # if user.recognize_manipulation == False and image['true_fake'] == 'F':
            #    matching_image_ids[idx] += 1
            if user.fake_preference == True and image['authenticity'] == 'F':
                matching_image_ids[idx] += 1
            if user.fake_preference == False and image['authenticity'] == 'T':
                matching_image_ids[idx] += 1
            if user.viral_preference == True and image['virality'] > 3:
                matching_image_ids[idx] += 1
            if user.viral_preference == False and image['virality'] <= 3:
                matching_image_ids[idx] += 1
        matching_image_ids = {k: v for k, v in sorted(matching_image_ids.items(), key=lambda item: item[1])}
        relevant_images = [k for k, v in matching_image_ids.items() if v == matching_image_ids[list(matching_image_ids.keys())[-1]]]
        selected_image_id = random.choice(relevant_images)
        if grid_info_dict[key][selected_image_id]['true_fake'] == 'T':
            image_credibility = np.random.normal(loc=user.get_user_features()[8], scale=0.1)
        else:
            image_credibility = 1 - (np.random.normal(loc=user.get_user_features()[8], scale=0.1))
        if image_credibility < 0:
            image_credibility = 0
        if image_credibility > 1:
            image_credibility = 1
        if grid_info_dict[key][selected_image_id]['true_fake'] == 'T':
            manipulation_visibility = np.random.normal(loc=user.get_user_features()[11], scale=0.1)
        else:
            manipulation_visibility = 1 - (np.random.normal(loc=user.get_user_features()[11], scale=0.1))
        if manipulation_visibility < 0:
            manipulation_visibility = 0
        if manipulation_visibility > 1:
            manipulation_visibility = 1
        selected_images[key] = (grid[selected_image_id].split('/')[-1], image_credibility, manipulation_visibility)

    return selected_images
