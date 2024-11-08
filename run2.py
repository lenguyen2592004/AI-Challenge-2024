MODEL_DIR = '.\model'
# Đường dẫn đến các tệp chỉ mục và metadata
input_index_path = r"E:\AIC\archive\vectordb-blip2-12\vector_database.usearch"
input_metadata_path = r"E:\AIC\archive\vectordb-blip2-12\image_metadata.csv"
input_index_path_sen = r"E:\AIC\archive\vietocr-embedding2\vector_database_text.usearch"
input_metadata_path_sen = r"E:\AIC\archive\vietocr-embedding2\image_metadata_text.csv"
dir_img=r'D:\INSECLAB\AIC_2024\aic-frames\output'
object_dir = r"E:\AIC\archive\object"
audio__dir = r"E:\AIC\archive\audio"
local=True

import time
import json
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from usearch.index import Index
from tqdm import tqdm
import matplotlib.pyplot as plt
from lavis.models import load_model_and_preprocess
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from pyvi.ViTokenizer import tokenize
import sys
from contextlib import redirect_stdout, redirect_stderr
from contextlib import contextmanager
import logging
from sklearn.metrics.pairwise import cosine_distances
import streamlit as st
import csv
from io import BytesIO
from io import StringIO
from PIL import Image, UnidentifiedImageError
from sklearn.preprocessing import MinMaxScaler
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.nn.functional as F

st.set_page_config(page_title="Image Retrieval Interface", page_icon="🖼️", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    /* Cài đặt màu nền cho phần chính */
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
    }

    /* Tùy chỉnh nút bấm */
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 12px 28px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }

    /* Tùy chỉnh ô nhập liệu */
    .stTextInput>div>div>input {
        padding: 10px;
        font-size: 16px;
        border-radius: 4px;
        border: 1px solid #ccc;
    }

    /* Tùy chỉnh selectbox */
    .stSelectbox>div>div>div {
        padding: 10px;
        font-size: 16px;
        border-radius: 4px;
        border: 1px solid #ccc;
    }

    /* Tùy chỉnh ô text_area */
    .stTextArea textarea {
        border: 2px solid #4CAF50; /* Thêm viền cho text_area */
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }

    /* Tùy chỉnh tiêu đề */
    h1 {
        text-align: center;
        color: #FF8C00;
        font-family: 'Georgia', serif;
    }
    </style>

    <h1> 🐝 AIC </h1>
""", unsafe_allow_html=True)

# Page title and description
st.title("📸 Image Retrieval Interface")
st.markdown("""
    Welcome to the Image Retrieval Interface. Use the form below to enter your prompt and sentence, select the retrieval method, and visualize the results.
    """)


# Suppress logging for specific libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("some_other_library").setLevel(logging.ERROR)
os.environ['TRANSFORMERS_NO_TQDM'] = '1'  # Disable TQDM progress bars for transformers


# Define the device to use 'cuda' or fallback to CPU if CUDA is not available
device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device: ",device1)
@st.cache_resource
def load_blip2_model_and_preprocessors(device):
    
    # Load the BLIP2 model checkpoint and preprocessors from the saved file
    checkpoint = torch.load(os.path.join(MODEL_DIR, "blip2_full_model.pth"), map_location=device)
    
    # Initialize the BLIP2 model structure and load the saved state
    model, vis_processors, txt_processors = load_model_and_preprocess(
        "blip2_feature_extractor", "coco", device=device, is_eval=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Use the saved visual and text processors from the checkpoint
    vis_processors = checkpoint['vis_processors']
    txt_processors = checkpoint['txt_processors']
    print("Loading BLIP2 model...: Done")
    
    return model, vis_processors, txt_processors

@st.cache_resource
def load_sentence_transformer_model(device):
    # Load the sentence transformer model and move it to the specified device
    model_emb = SentenceTransformer(os.path.join(MODEL_DIR, "sentence_transformer_model"))
    model_emb = model_emb.to(device)
    print("Loading Sentence Transformer model...: Done")
    return model_emb

@st.cache_resource
def load_translate_model(device):
    # Load the translation model (VietAI/envit5-translation) and tokenizer
    model_name = os.path.join(MODEL_DIR, "translation_model")
    tokenizer1 = AutoTokenizer.from_pretrained(model_name)
    model1 = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    print("Loading translation model...: Done")
    return tokenizer1, model1

# Load the models with caching to avoid reloading multiple times
model, vis_processors, txt_processors = load_blip2_model_and_preprocessors(device='cpu')
model_emb = load_sentence_transformer_model(device=device1)
tokenizer_trans, model_trans = load_translate_model(device=device1)



def extract_image_features(image):
    image = Image.open(image[0])
    image_tensor = vis_processors["eval"](image.convert("RGB")).to(device1)
    image_tensor = image_tensor.unsqueeze(0)
    sample = {"image": image_tensor}
    with torch.no_grad():
        features_image = model.extract_features(sample, mode="image")
        features_image = torch.mean(features_image.image_embeds, dim=1)
    return features_image

# Function to extract text features using BLIP2
def extract_text_features(text):
    text = txt_processors["eval"](text)
    sample = {"text_input": [text]}
    with torch.no_grad():        
        features_text = model.extract_features(sample, mode="text")
        features_text = torch.mean(features_text.text_embeds, dim=1).squeeze(1)
        
    return features_text


# def extract_image_features(image):
#     # Preprocess the image and move it to the device
#     image_input = vis_processors(image).unsqueeze(0).to(device1)
#     with torch.no_grad():
#         # Encode the image and normalize the features
#         image_embed = model.encode_image(image_input)
#         image_embed = F.normalize(image_embed, p=2, dim=-1)
#     return image_embed

# # Function to extract text features using BLIP2
# def extract_text_features(text):
#     # Tokenize the text and move it to the device
#     text_input = txt_processors([text]).to(device1)
#     with torch.no_grad():
#         # Encode the text and normalize the features
#         text_embed = model.encode_text(text_input)
#         text_embed = F.normalize(text_embed, p=2, dim=-1)
#     return text_embed









@st.cache_resource
def load_usearch_and_metadata(input_index_path, input_metadata_path):
    dimension = 768  # Ensure this matches the dimension used during creation
    index = Index(ndim=dimension, dtype=np.float16)
    index.load(input_index_path)
    
    # Load metadata
    metadata_df = pd.read_csv(input_metadata_path)
    
    # In ra kích thước của DataFrame và index
    print(f'metadata_df shape: {metadata_df.shape}')
    print(f'index shape: {index.size}')  # Giả sử index có phương thức `size()` để lấy số lượng vector
    
    print('load_usearch_and_metadata: Done')
    return index, metadata_df

@st.cache_resource
def load_sentence_usearch_and_metadata(input_index_path, input_metadata_path):
    dimension_sen = 384  # Example dimension; adjust according to your embedding model
    index_sen = Index(ndim=dimension_sen, dtype=np.float16)
    index_sen.load(input_index_path)
    
    # Load metadata
    metadata_df_sen = pd.read_csv(input_metadata_path)
    
    # In ra kích thước của DataFrame và index
    print(f'metadata_df_sen shape: {metadata_df_sen.shape}')
    print(f'index_sen shape: {index_sen.size}')  # Giả sử index_sen có phương thức `size()` để lấy số lượng vector
    
    print('load_sentence_usearch_and_metadata: Done')
    return index_sen, metadata_df_sen



# DOWNNNNNNNNNNNNNNNNNNNNNNNNN
from langdetect import detect
import pickle
from itertools import combinations

@st.cache_resource # Load từ vựng và embedding từ file pickle
def load_vocab_and_embeddings_obj(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)  # Load từ file pickle
    vocabulary = data['vocabulary']
    embeddings = data['embeddings']
    print("Loading embeddings_obj: ........")
    return vocabulary, embeddings

@st.cache_resource # Load label_dict từ file pickle
def load_label_dict(input_file):
    with open(input_file, 'rb') as f:
        return pickle.load(f)

vocabulary_obj, embedding_obj = load_vocab_and_embeddings_obj(object_dir + '/vocab_embeddings.pkl')
label_dict = load_label_dict(object_dir + '/label_dict.pkl')

# Bước 2: Tìm top 5 từ gần nhất dựa vào từ prompt
def find_top_5_words(prompt, vocabulary=vocabulary_obj, embeddings=embedding_obj,model=model_emb):
    prompt_embedding = model.encode([prompt])  # embedding của từ prompt
    similarities = cosine_similarity(prompt_embedding, embeddings)  # tính độ tương đồng cosine
    top_5_idx = np.argsort(similarities[0])[::-1][:5]  # lấy 5 index có độ tương đồng cao nhất
    top_5_words = [vocabulary[idx] for idx in top_5_idx]  # lấy ra từ tương ứng
    return top_5_words, top_5_words[0]  # trả về danh sách top 5 và từ có độ tương đồng cao nhất (index đầu tiên)

def object(prompt_vietnamese, model=model_emb, label_dict=label_dict):
    # Nếu prompt có nhiều object, tách thành sub_prompts
    if "/" in prompt_vietnamese:
        sub_prompts = [sub.strip() for sub in prompt_vietnamese.split('/')]
    else:
        sub_prompts = [prompt_vietnamese.strip()]

    all_dataframes = []  # List to collect DataFrames across combinations
    translated_text_all = ""
    text2print = ""  # To store top_5_words for each sub_prompt
    total_rows = 0  # Track the total rows across all merged DataFrames
    
    # Iterate over all combinations of sub_prompts (largest to smallest)
    for r in range(len(sub_prompts), 0, -1):  # From max size to size 1
        for combo in combinations(sub_prompts, r):
            combo_text = " / ".join(combo)  # For text2print
            combo_dfs = []
            combo_text2print = ""
            combo_translated_text_all = ""
            
            for prompt in combo:
                prompt_output, translated_text, response_time = translate_vietnamese_to_english(prompt)
                combo_translated_text_all += str(translated_text) + " / "

                # Find top 5 words and closest word
                top_5_words, closest_word = find_top_5_words(prompt=translated_text, model=model_emb)

                # Append top 5 words to text2print
                combo_text2print += f"Top 5 từ gần nhất cho '{translated_text}':\n" + ", ".join(top_5_words) + "\n\n"

                # Get DataFrame from label_dict for the closest word
                if closest_word in label_dict:
                    data_obj = label_dict[closest_word]  # Get filename and confidence list
                    df_obj = pd.DataFrame(data_obj, columns=['image_path', 'confidence'])  # Convert to DataFrame
                    combo_dfs.append(df_obj)
                else:
                    continue  # Skip if closest_word is not found in label_dict

            if combo_dfs:
                # Merge all DataFrames within this combination
                merged_df = combo_dfs[0]
                for df in combo_dfs[1:]:
                    merged_df = pd.merge(merged_df, df, on='image_path', suffixes=('', '_other'))

                # Calculate total confidence
                confidence_cols = [col for col in merged_df.columns if 'confidence' in col]
                merged_df['total_confidence'] = merged_df[confidence_cols].sum(axis=1)

                # Sort by total confidence
                merged_df = merged_df[['image_path', 'total_confidence']].sort_values(by='total_confidence', ascending=False)

                # Remove duplicate rows
                merged_df = merged_df.drop_duplicates(subset=['image_path'])

                # Accumulate small DataFrames if they have fewer than 20 rows
                if len(merged_df) < 20:
                    all_dataframes.append(merged_df)
                    translated_text_all += combo_translated_text_all
                    text2print += combo_text2print
                    total_rows += len(merged_df)
                else:
                    # If more than 20 rows, stop and return this result
                    return merged_df, combo_text2print

            # Break if total rows across all combinations exceed 20
            if total_rows > 20:
                break
        
        # Check if total rows across all DataFrames exceeds 20 after each round of combination
        if total_rows > 20:
            break

    # Concatenate all DataFrames if no large DataFrame was found
    if all_dataframes:
        final_df = pd.concat(all_dataframes).drop_duplicates(subset=['image_path'])
        final_df = final_df.sort_values(by='total_confidence', ascending=False)

        return final_df, text2print  # Return the combined DataFrame and the accumulated text2print
    else:
        return "Không có đối tượng nào được tìm thấy trong tất cả các sub_prompts.", text2print




def time_to_seconds(time_str):
    # Parse the time string in the format "H:M:S" or "M:S"
    parts = time_str.split(':')
    parts = [int(part) for part in parts]  # Convert parts to integers
    if len(parts) == 3:
        return timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2]).total_seconds()
    elif len(parts) == 2:
        return timedelta(minutes=parts[0], seconds=parts[1]).total_seconds()
    else:
        return float(parts[0])
def load_data():
    #with open(audio__dir+"/"+'data.pkl', 'rb') as f:
    with open(os.path.join(audio__dir,'data.pkl'), 'rb') as f:
        data = pickle.load(f)
        return data['fps_dict'], data['images_dict']

from rapidfuzz import fuzz
from datetime import timedelta
import bisect
def process_image_filename(filename):
    # Remove the file extension and split by the first period
    name_without_extension = filename.rsplit('.', 1)[0]
    # Split by the first period to separate the prefix and the number part
    prefix, number_str = name_without_extension.split('.', 1)
    
    # Convert the number part to an integer
    try:
        number = int(number_str)
    except ValueError:
        # Handle cases where the number part is not a valid integer
        raise ValueError(f"Filename '{filename}' does not have a valid numeric part after the period.")
    
    return prefix, number

# Optimized function to find the closest image using binary search
def find_closest_image_in_dict(image_path, dict_images):
    # Process the image filename to extract the prefix and number
    prefix, target_number = process_image_filename(os.path.basename(image_path))

    # If the prefix exists in dict_images, use binary search to find the closest image with a larger number
    if prefix in dict_images:
        images_with_numbers = dict_images[prefix]  # List of tuples (filename, number)
        # Extract the list of numbers
        numbers = [number for _, number in images_with_numbers]
        # Find the index where the target number would be inserted (right after the target)
        pos = bisect.bisect_right(numbers, target_number)
        
        # Check if the next available number is at least 28 greater than the target number
        while pos < len(numbers):
            if numbers[pos] > target_number + 28:
                return os.path.join(dir_img, images_with_numbers[pos][0])  # Return the closest match with a larger number
            pos += 1  # Move to the next number if the condition isn't met
    
    return None 

# Updated add_frame_before_extension function
def add_frame_before_extension(video_path, start_frame):
    # Replace '.mp4' with the format '.<frame>.jpg'
    return video_path.replace('.mp4', f".{start_frame}.jpg")

# Main function with optimized find_closest_image_in_dict and new filename format
def audio_func(input_text, csv_file='audio.csv', threshold=5, detailed_threshold=5):
    # Load data once
    dict_fps, dict_images = load_data()

    #df = pd.read_csv(audio__dir+"/"+csv_file)
    df = pd.read_csv(os.path.join(audio__dir,csv_file))

    # Step 1: Calculate similarity_fuzz for all rows
    df['similarity_fuzz'] = df['text'].apply(lambda text: fuzz.partial_ratio(input_text, text))

    # Step 2: Filter rows based on similarity_fuzz threshold
    filtered_df = df[df['similarity_fuzz'] >= threshold]

    if filtered_df.empty:
        print("No rows exceed the similarity_fuzz threshold.")
        return pd.DataFrame(), ""

    # Step 3: Find the top 2 similarity_fuzz scores
    top_fuzz_scores = filtered_df['similarity_fuzz'].drop_duplicates().nlargest(4)

    # Step 4: Filter rows with top 1 or top 2 similarity_fuzz scores
    top_filtered_df = filtered_df[filtered_df['similarity_fuzz'].isin(top_fuzz_scores)].copy()

    # Step 5: Calculate detailed similarity metrics
    top_filtered_df['similarity_token'] = top_filtered_df['text'].apply(lambda text: fuzz.token_set_ratio(input_text, text))

    # Step 6: Calculate average similarity score
    top_filtered_df['avg_similarity'] = top_filtered_df[['similarity_fuzz', 'similarity_token']].mean(axis=1)

    # Step 7: Filter based on detailed threshold
    final_results_df = top_filtered_df[top_filtered_df['avg_similarity'] >= detailed_threshold].copy()

    # Step 8: Sort the results by avg_similarity in descending order
    sorted_results = final_results_df.sort_values(by='avg_similarity', ascending=False).head(30).copy()

    # Step 9: Convert 'start' and 'end' from time strings to total seconds
    sorted_results['start_seconds'] = sorted_results['start'].apply(time_to_seconds)
    sorted_results['end_seconds'] = sorted_results['end'].apply(time_to_seconds)

    # Step 10: Create 'video_name' from 'video_path'
    sorted_results['video_name'] = sorted_results['video_path'].apply(lambda vp: os.path.splitext(os.path.basename(vp))[0])

    # Step 11: Map FPS from dict_fps to the video_path column
    sorted_results['fps'] = sorted_results['video_path'].map(dict_fps)

    # Step 12: Replace missing FPS with a default value (25)
    # sorted_results['fps'] = sorted_results['fps'].fillna(25)

    # Step 13: Calculate start_frame using FPS
    sorted_results['start_frame'] = (sorted_results['start_seconds'] * sorted_results['fps']).astype(int)

    # Step 14: Construct image_path column with frame before .jpg
    sorted_results['image_path'] = sorted_results.apply(lambda row: add_frame_before_extension(row['video_path'], row['start_frame']), axis=1)

    # Step 15: Check if image_path exists in dict_images; if not, find a replacement
    for idx, row in sorted_results.iterrows():
        image_path = row['image_path']
        if not any(image_path in v for v in dict_images.values()):
            # If image_path is not in dict_images, find the closest replacement
            replacement_image = find_closest_image_in_dict(image_path, dict_images)
            if replacement_image:
                sorted_results.at[idx, 'image_path'] = replacement_image  # Replace with the closest image

    # Step 16: Construct text2print using vectorized string concatenation
    sorted_results['text2print'] = (
        sorted_results['video_name'] + 
        ", " + (sorted_results['start_frame'] + 28).astype(str) +  # Add 100 to the start_frame
        " || start " + sorted_results['start'] + 
        " - end " + sorted_results['end'] + 
        " || text: " + sorted_results['text'] + "\n"
    )

    # Step 17: Combine all rows into a single text2print string
    text2print = "\n".join(sorted_results['text2print'].tolist())

    # Return the DataFrame with only the 'image_path' column and the combined text2print
    return sorted_results[['image_path']], text2print








def translate_vietnamese_to_english(prompt_vietnamese):
    if detect(prompt_vietnamese) == 'en':
        return "",prompt_vietnamese,""
    
    else:
        # Nếu có dấu "/", tách chuỗi thành các phần nhỏ và strip() từng phần
        if "/" in prompt_vietnamese:
            parts = [part.strip() for part in prompt_vietnamese.split("/")]
        else:
            parts = [prompt_vietnamese.strip()]

        # Thêm tiền tố "vi: " cho mỗi phần
        inputs = ["vi: " + part for part in parts]

        # Mã hóa các câu đầu vào và tạo batch đầu vào
        input_ids = tokenizer_trans(inputs, return_tensors="pt", padding=True).input_ids.to(device1)

        # Dùng mô hình để tạo bản dịch cho tất cả các câu trong batch
        output_ids = model_trans.generate(input_ids, max_length=512)

        # Giải mã bản dịch từ các token ID
        translations = tokenizer_trans.batch_decode(output_ids, skip_special_tokens=True)

        # Loại bỏ tiền tố "en: " nếu có trong mỗi bản dịch
        translations = [translation[4:] if translation.startswith("en: ") else translation for translation in translations]

        # Kết hợp các bản dịch thành một chuỗi, phân cách bằng "/"
        final_translation = "/".join(translations)

        return "",final_translation,""



# Ensure this function is defined to calculate the combined score with three components
def calculate_combined_score_with_sentence(results, weight_score=0.3, weight_count=0.2, weight_sen=0.5):
    # Normalize the scores
    results['normalized_score'] = (1 - normalize(results['score']))  # Lower scores are better
    results['normalized_count'] = normalize(results['count'])
    results['normalized_sen'] = (1 - normalize(results['sentence_dis']))  # Lower sentence_dis is better-------

    # Calculate the combined score
    results['combined_score'] = (
        weight_score * results['normalized_score'] +
        weight_count * results['normalized_count'] +
        weight_sen * results['normalized_sen']
    )

    return results.sort_values(by='combined_score', ascending=False)


def normalize(series):
    """Normalize a pandas series to a range between 0 and 1."""
    return (series - series.min()) / (series.max() - series.min())

def merge_and_rank_datasets(dataset_1, dataset_2):
    # Merge datasets using outer join
    merged_df = pd.merge(dataset_1, dataset_2, on='image_path', how='outer')

    # Filter to keep rows where 'score_text' and 'score' are not null
    filtered_df = merged_df.dropna(subset=['score_text', 'score']).copy()

    # Rename columns
    filtered_df.rename(columns={'score_text': 'text_similarity_score', 'score': 'image_similarity_score'}, inplace=True)

    # Normalize both text and image similarity scores in one step
    scaler = MinMaxScaler()
    filtered_df[['norm_text_similarity_score', 'norm_image_similarity_score']] = scaler.fit_transform(
        filtered_df[['text_similarity_score', 'image_similarity_score']]
    )

    # Calculate the new score
    filtered_df['score'] = 0.55 * filtered_df['norm_text_similarity_score'] + 0.45 * filtered_df['norm_image_similarity_score']

    # Sort by the calculated score
    ranked_df = filtered_df.sort_values(by='score', ascending=True).copy()

    return ranked_df




def calculate_combined_score(top_k_results, weight_score=0.5, weight_count=0.5):
    # Normalize the 'score' (distance) and invert it so that higher is better
    normalized_score = normalize(top_k_results['score'])
    inverted_score = 1 - normalized_score  # Inverting because lower distance is better

    # Normalize the 'count'
    normalized_count = normalize(top_k_results['count'])

    # Calculate the combined score as a weighted sum
    top_k_results['combined_score'] = (weight_score * inverted_score) + (weight_count * normalized_count)

    # Sort the DataFrame based on the combined score
    ranked_results = top_k_results.sort_values(by='combined_score', ascending=False)

    return ranked_results

def plot_ranked_images(ranked_results, images_per_row=5, max_images=100):
    # Determine the number of images to plot
    num_images = min(ranked_results.shape[0], max_images)
    
    # Calculate the number of rows needed
    num_rows = (num_images + images_per_row - 1) // images_per_row
    
    # Create a plot with the appropriate size
    plt.figure(figsize=(20, num_rows * 4))  # Adjust the size based on the number of rows

    for i, (_, row) in enumerate(ranked_results.iloc[:num_images].iterrows()):
        image_path = row["image_path"]
        combined_score = row["combined_score"]
        image = Image.open(image_path)
        
        # Plot the image in the grid
        plt.subplot(num_rows, images_per_row, i + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Score: {combined_score:.2f}")

    # Display the plot
    plt.tight_layout()
    plt.show()
    
def plot_images(image_paths, images_per_row=5):
    num_images = len(image_paths)
    
    # Calculate the number of rows needed
    num_rows = (num_images + images_per_row - 1) // images_per_row
    
    # Create a plot with the appropriate size
    plt.figure(figsize=(20, num_rows * 4))  # Adjust the size based on the number of rows

    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        
        # Plot the image in the grid
        plt.subplot(num_rows, images_per_row, i + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Image {i + 1}")

    # Display the plot
    plt.tight_layout()
    plt.show()
    
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr



# Load index và metadata với caching
index, metadata_df = load_usearch_and_metadata(input_index_path, input_metadata_path)
index_sen, metadata_df_sen = load_sentence_usearch_and_metadata(input_index_path_sen, input_metadata_path_sen)

def query_with_text_emb(sentence, top_k=100, index=index_sen, metadata_df=metadata_df_sen, model=model_emb):
    # Generate the sentence embedding
    sentence_embedding = model.encode(sentence, convert_to_tensor=True).detach().cpu().numpy().astype(np.float16)

    matches = index.search(sentence_embedding, top_k)
    # Retrieve the top indices and their corresponding scores
    top_indices = matches.keys  # Assuming matches.keys is a 1D array for a single query
    top_scores = matches.distances  # Similarly, assuming matches.distances is a 1D array
    
    # Retrieve the top results from metadata_df using the indices
    top_results = metadata_df.iloc[top_indices].copy()
    
    # Add the scores to the results dataframe
    top_results['score_text'] = top_scores

    return top_results

def query_with_text(prompt, top_k=5, index=index, metadata_df=metadata_df):
    # Extract text features
    text_features = extract_text_features(prompt)
    text_features = text_features.cpu().numpy().astype(np.float16)
    
    # Perform the search with usearch
    matches = index.search(text_features, top_k)
    
    # Retrieve the top indices and their corresponding scores
    top_indices = matches.keys  # Assuming matches.keys is a 1D array for a single query
    top_scores = matches.distances  # Similarly, assuming matches.distances is a 1D array
    
    # Retrieve the top results from metadata_df using the indices
    top_results = metadata_df.iloc[top_indices].copy()
    
    # Add the scores to the results dataframe
    top_results['score'] = top_scores

    return top_results



# 100 ảnh 
def main3(prompt_vietnamese, top_k=100, index=index, metadata_df=metadata_df, plot=True):
    # Start timing
    start_time = time.time()

    # Step 1: Translate Vietnamese prompt to English
    prompt_output, translated_text, response_time = translate_vietnamese_to_english(prompt_vietnamese)
#     print(f"Translated English text: {translated_text}")

    # Step 2: Query the top 100 results based on the translated text
    top_k_results = query_with_text(translated_text, top_k=top_k, index=index, metadata_df=metadata_df)
    
    # Extract the image paths for plotting
    image_paths = top_k_results['image_path'].tolist()

    # End timing for processing steps (excluding plotting)
    # pre_plot_end_time = time.time()
    # processing_time = pre_plot_end_time - start_time

    # if plot:
    #     # Plotting images
    #     plot_images(image_paths, images_per_row=5)

    # End timing after plotting
#     post_plot_end_time = time.time()
#     total_time_with_plotting = post_plot_end_time - start_time
#     plotting_time = post_plot_end_time - pre_plot_end_time

    # Print timing information
#     print(f"Total processing time (without plotting): {processing_time:.2f} seconds")
#     print(f"Total plotting time: {plotting_time:.2f} seconds")
#     print(f"Total execution time (with plotting): {total_time_with_plotting:.2f} seconds")

    return top_k_results, translated_text


# 5000 ảnh + 5000 text_emb --> 100 img
def main4(prompt_vietnamese, sentence, top_k=5000, flag=True, flag2=False):
    dataset_1 = query_with_text_emb(sentence, top_k=top_k)
    dataset_2, translated_text = main3(prompt_vietnamese, top_k=top_k, plot=False)
    # Merge, filter, normalize, and rank the datasets
    final_dataset = merge_and_rank_datasets(dataset_1, dataset_2)
    # Return only the top 100 rows if there are more than 100 rows
    final_dataset = final_dataset.drop_duplicates(subset='image_path', keep='first')
    return final_dataset, translated_text


def main1(prompt_vietnamese, objects, top_k=5000):
    dataset_1, translated_text = main3(prompt_vietnamese, top_k=top_k, plot=False)
    dataset_2, text2print = object(objects, model=model_emb, label_dict=label_dict)

    merged_df = pd.merge(
        dataset_1[['image_path', 'score']].astype({'score': 'float16'}),  # Use float16 for lower memory usage
        dataset_2[['image_path', 'total_confidence']].astype({'total_confidence': 'float16'}),
        on='image_path',
        how='inner'
    )

    # Scale both 'score' and 'total_confidence' columns simultaneously using NumPy for better performance
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(merged_df[['score', 'total_confidence']].values)

    # Assign scaled values directly to the DataFrame
    merged_df['score_scaled'], merged_df['total_conf_scaled'] = scaled_values[:, 0], 1 - scaled_values[:, 1]  # Invert 'total_conf_scaled'

    # Vectorized final score computation using NumPy
    merged_df['final_score'] = np.add(0.65 * merged_df['score_scaled'], 0.35 * merged_df['total_conf_scaled'])

    # In-place sorting by 'final_score' in ascending order (lower is better)
    final_df = merged_df.sort_values(by='final_score', ascending=True, inplace=False)
    
    final_df = final_df.drop_duplicates(subset='image_path', keep='first')

    # Giữ lại chỉ hai cột 'image_path' và 'final_score', sau đó đổi tên cột
    final_df = final_df[['image_path', 'final_score']].rename(columns={'final_score': 'score'})

    return final_df, translated_text + "\n\n\n" + text2print

        
def filter_df_by_frame_gap(df, frame_column, max_frame_gap=200):
    """
    Lọc các frame trong khoảng từ frame nhỏ nhất là idx đến idx + max_frame_gap
    và chỉ giữ lại frame ở vị trí trên cùng trong DataFrame ban đầu cho mỗi nhóm video.
    """
    # Sắp xếp theo frame_column để đảm bảo frame được xử lý tuần tự
    df_sorted = df.sort_values(by=[frame_column])
    df_filtered = []

    # Nhóm theo video_name
    for video_name, group in df_sorted.groupby('video_name'):
        group = group.copy()
        start_idx = 0
        while start_idx < len(group):
            # Lấy phần còn lại của nhóm bắt đầu từ start_idx
            subset_group = group.iloc[start_idx:]

            # Tìm các frame trong khoảng từ [idx : idx + max_frame_gap]
            subset = subset_group[subset_group[frame_column] <= subset_group[frame_column].iloc[0] + max_frame_gap]

            # Chọn frame ở vị trí cao nhất trong DataFrame ban đầu
            highest_row = df.loc[subset.index].iloc[0]  # Lấy hàng ở vị trí cao nhất trong df ban đầu
            df_filtered.append(highest_row)

            # Cập nhật start_idx để bỏ qua khoảng frame đã xét
            start_idx += len(subset)

    # Tạo lại DataFrame từ danh sách các frame đã lọc
    df_filtered = pd.DataFrame(df_filtered)
    df_filtered = df_filtered.drop_duplicates(subset=['image_path_0'])
    return df_filtered



def extract_video_and_frame(image_path):
    parts = image_path.split('/')[-1].split('.')
    video_name = parts[0]
    frame = int(parts[1])
    return video_name, frame



def temporal_search(dataset_subprompts, fps=25, base_gap=10):
    """
    Tìm tất cả các chuỗi frame liên tiếp trong cùng một video thỏa mãn điều kiện thời gian.
    """
    n_prompts = len(dataset_subprompts)
    
    # Tạo max_gap_seconds tự động dựa trên số lượng prompts
    max_gap_seconds = [base_gap * i for i in range(1, n_prompts)]  # max_gap_seconds cho từng frame sau frame đầu
    max_gap_frames = [gap * fps for gap in max_gap_seconds]  # max_gap_frames giữa frame_0 và các frame còn lại
    result = []
    
    # Thêm video_name và frame vào tất cả DataFrame
    for i, df in enumerate(dataset_subprompts):
        df['video_name'], df[f'frame_{i}'] = zip(*df['image_path'].apply(extract_video_and_frame))
        df.rename(columns={'image_path': f'image_path_{i}', 'index': f'index_{i}', 'score': f'score_{i}'}, inplace=True)
        dataset_subprompts[i] = df
    
    max_frame_gap = int(base_gap * fps * 0.2)
    # Bắt đầu với DataFrame đầu tiên đã được lọc
    df_filtered = filter_df_by_frame_gap(dataset_subprompts[0], f'frame_0', max_frame_gap=max_frame_gap)
    
    # Duyệt qua từng hàng trong df_filtered
    for idx, row in df_filtered.iterrows():
        current_video = row['video_name']
        frames = [row[f'frame_0']]
        image_paths = [row[f'image_path_0']]
        total_score = row[f'score_0']
    
        # Tạo danh sách các chuỗi frame tiềm năng
        sequences = [{
            'frames': frames,
            'image_paths': image_paths,
            'total_score': total_score,
            'current_frame': frames[-1]
        }]
    
        # Duyệt qua từng frame tiếp theo
        for i in range(1, n_prompts):
            new_sequences = []
            for seq in sequences:
                current_frame = seq['current_frame']
                # Lấy DataFrame tương ứng
                next_df = dataset_subprompts[i]
                # Tìm tất cả các frame thỏa mãn điều kiện
                next_frames = next_df[
                    (next_df['video_name'] == current_video) & 
                    (next_df[f'frame_{i}'] - seq['frames'][0] <= max_gap_frames[i-1]) &
                    (next_df[f'frame_{i}'] - current_frame > 0)
                ]
                if not next_frames.empty:
                    # Sắp xếp next_frames theo số frame
                    next_frames = next_frames.sort_values(by=f'frame_{i}')
                    for _, next_row in next_frames.iterrows():
                        next_frame_value = next_row[f'frame_{i}']
                        # Tạo chuỗi mới
                        new_seq = {
                            'frames': seq['frames'] + [next_frame_value],
                            'image_paths': seq['image_paths'] + [next_row[f'image_path_{i}']],
                            'total_score': seq['total_score'] + next_row[f'score_{i}'],
                            'current_frame': next_frame_value
                        }
                        new_sequences.append(new_seq)
                # Nếu không tìm thấy frame tiếp theo, không thêm chuỗi mới
            sequences = new_sequences
            # Nếu không còn chuỗi nào, thoát khỏi vòng lặp
            if not sequences:
                break
    
        # Thêm các chuỗi hoàn chỉnh vào kết quả
        for seq in sequences:
            if len(seq['frames']) == n_prompts:
                result.append({
                    "video": current_video,
                    "frames": seq['frames'],
                    "image_paths": seq['image_paths'],
                    "total_score": seq['total_score']
                })
    
    # Sắp xếp kết quả theo total_score (từ nhỏ đến lớn)
    result = sorted(result, key=lambda x: x['total_score'])

    # Loại bỏ các sequences giống nhau (nếu 1 sequences mà có (n_prompts - 1)/n_prompts giống với sequences khác thì chỉ giữ lại 1)
    unique_results = []
    for seq in result:
        is_duplicate = False
        for unique_seq in unique_results:
            shared_frames = set(seq['frames']).intersection(set(unique_seq['frames']))
            if len(shared_frames) >= n_prompts - 1:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_results.append(seq)
    result = unique_results

    return result


def temporal_search_plus(dataset_subprompts, fps=25, base_gap=10):
    """
    Tìm các frame liên tiếp trong cùng một video sử dụng tính năng merge của Pandas và xếp hạng theo tổng score.

    - dataset_subprompts: Danh sách các DataFrame chứa các subprompt.
    - fps: Số frame per second của video (mặc định là 25).
    - base_gap: Khoảng cách cơ bản giữa các frame tính bằng giây (mặc định là 10s).

    Trả về danh sách các frame liên tiếp thỏa mãn điều kiện, được xếp hạng theo tổng score.
    """
    n_prompts = len(dataset_subprompts)
    
    # Tạo max_gap_seconds tự động dựa trên số lượng prompts
    max_gap_seconds = [base_gap * i for i in range(1, n_prompts)]  # max_gap_seconds cho từng frame sau frame đầu
    max_gap_frames = [gap * fps for gap in max_gap_seconds]  # max_gap_frames giữa frame_0 và các frame còn lại
    result = []

    # Thêm video_name và frame vào tất cả DataFrame
    for i, df in enumerate(dataset_subprompts):
        df['video_name'], df[f'frame_{i}'] = zip(*df['image_path'].apply(extract_video_and_frame))
        df = df.rename(columns={'image_path': f'image_path_{i}', 'index': f'index_{i}', 'score': f'score_{i}'})
        dataset_subprompts[i] = df
    
    max_frame_gap= int(base_gap*fps*0.2)
    # Bắt đầu với DataFrame đầu tiên
    df_filtered = filter_df_by_frame_gap(dataset_subprompts[0], f'frame_0', max_frame_gap=max_frame_gap)

    # Chạy qua từng frame tiếp theo và áp dụng điều kiện max_gap_frames
    for _, row in df_filtered.iterrows():
        valid_frames = True
        current_video = row['video_name']
        current_frame = row['frame_0']
        frames = [current_frame]
        image_paths = [row[f'image_path_0']]
        total_score = row[f'score_0']  # Tổng score ban đầu

        # Duyệt qua các DataFrame tiếp theo (frame_1, frame_2, ...)
        for i in range(1, n_prompts):
            next_df = dataset_subprompts[i]
            
            # Tìm frame trong cùng video_name, cách current_frame không quá max_gap_frames[i-1]
            next_frames = next_df[(next_df['video_name'] == current_video) & 
                                  (next_df[f'frame_{i}'] - current_frame <= max_gap_frames[i-1]) &
                                  (next_df[f'frame_{i}'] - current_frame > 0)]
            
            # Sắp xếp next_frames theo số frame trước khi chọn frame đầu tiên
            if not next_frames.empty:
                next_frames = next_frames.sort_values(by=f'frame_{i}')
                next_frame_value = next_frames.iloc[0][f'frame_{i}']

                # Đảm bảo frame tiếp theo lớn hơn frame hiện tại
                if next_frame_value > frames[-1]:
                    frames.append(next_frame_value)
                    image_paths.append(next_frames.iloc[0][f'image_path_{i}'])
                    total_score += next_frames.iloc[0][f'score_{i}']
                else:
                    valid_frames = False
                    break
            else:
                valid_frames = False
                break

        if valid_frames:
            result.append({
                "video": current_video,
                "frames": frames,
                "image_paths": image_paths,
                "total_score": total_score  # Lưu lại tổng score cho mỗi tập hợp kết quả
            })

    # Sắp xếp kết quả theo total_score (từ nhỏ đến lớn)
    result = sorted(result, key=lambda x: x['total_score'])

    return result



def main9(prompt_vietnamese, sentence, top_k=500, flag=True, base_gap=10):

    if "/" in prompt_vietnamese:
        sub_prompts = [sub.strip() for sub in prompt_vietnamese.split('/')]
        prompt_vietnamese = prompt_vietnamese.replace("/", ", ")
    else:
        sub_prompts = [prompt_vietnamese.strip()]
    
    dataset_subprompts = []
    translated_text_all = ""
    
    # Nếu không có sentence hoặc sentence rỗng, áp dụng main3 cho tất cả các prompt
    if sentence is None or sentence == '':
        for prompt in sub_prompts:
            dataset_subprompt, translated_text = main3(prompt, top_k=top_k, plot=False)
            translated_text_all += str(translated_text) + "/ "
            dataset_subprompts.append(dataset_subprompt)
    else:
        # Nếu có sentence, tách thành các câu
        sub_sens = [sub.strip() for sub in sentence.split('/')]
        
        # Nếu số lượng prompts và sentences không bằng nhau, in ra thông báo
        if len(sub_prompts) != len(sub_sens):
            print(f"Warning: Number of prompts ({len(sub_prompts)}) and sentences ({len(sub_sens)}) are not equal.")
        
        # Xử lý các cặp prompt và sen
        for prompt, sen in zip(sub_prompts, sub_sens):
            if sen == '':
                # Nếu sen rỗng, sử dụng main3
                dataset_subprompt, translated_text = main3(prompt, top_k=top_k, plot=False)
            else:
                # Nếu sen không rỗng, sử dụng main4
                dataset_subprompt, translated_text = main4(prompt, sen, top_k=top_k, flag2=True)
                
            translated_text_all += str(translated_text) + "/ "
            dataset_subprompts.append(dataset_subprompt)

    # Thực hiện temporal search
    result = temporal_search(dataset_subprompts, base_gap=base_gap)
    
    # Chỉ giữ lại những item có số lượng image_paths bằng số lượng sub_prompts
    all_image_paths = []
    for item in result:
        if len(item['image_paths']) == len(sub_prompts):
            all_image_paths.extend(item['image_paths'])  # Thêm tất cả image_paths từ mỗi item vào danh sách
    
    df_image_paths = pd.DataFrame({'image_path': all_image_paths})
    
    # Trả về DataFrame chứa image_paths và chuỗi translated_text_all
    return df_image_paths, translated_text_all, len(sub_prompts)




def main10(prompt_vietnamese, sentence, top_k=500, flag=True, base_gap=10):

    if "/" in prompt_vietnamese:
        sub_prompts = [sub.strip() for sub in prompt_vietnamese.split('/')]
        prompt_vietnamese = prompt_vietnamese.replace("/", ", ")
    else:
        sub_prompts = [prompt_vietnamese.strip()]
    
    dataset_subprompts = []
    translated_text_all = ""
    
    # Nếu không có sentence hoặc sentence rỗng, áp dụng main3 cho tất cả các prompt
    if sentence is None or sentence == '':
        for prompt in sub_prompts:
            dataset_subprompt, translated_text = main3(prompt, top_k=top_k, plot=False)
            translated_text_all += str(translated_text) + "/ "
            dataset_subprompts.append(dataset_subprompt)
    else:
        # Nếu có sentence, tách thành các câu
        sub_sens = [sub.strip() for sub in sentence.split('/')]
        
        # Nếu số lượng prompts và sentences không bằng nhau, in ra thông báo
        if len(sub_prompts) != len(sub_sens):
            print(f"Warning: Number of prompts ({len(sub_prompts)}) and sentences ({len(sub_sens)}) are not equal.")
        
        # Xử lý các cặp prompt và sen
        for prompt, sen in zip(sub_prompts, sub_sens):
            if sen == '':
                # Nếu sen rỗng, sử dụng main3
                dataset_subprompt, translated_text = main3(prompt, top_k=top_k, plot=False)
            else:
                # Nếu sen không rỗng, sử dụng main4
                dataset_subprompt, translated_text = main4(prompt, sen, top_k=top_k, flag2=True)
                
            translated_text_all += str(translated_text) + "/ "
            dataset_subprompts.append(dataset_subprompt)

    # Thực hiện temporal search
    result = temporal_search_plus(dataset_subprompts, base_gap=base_gap)
    
    # Chỉ giữ lại những item có số lượng image_paths bằng số lượng sub_prompts
    all_image_paths = []
    for item in result:
        if len(item['image_paths']) == len(sub_prompts):
            all_image_paths.extend(item['image_paths'])  # Thêm tất cả image_paths từ mỗi item vào danh sách
    
    df_image_paths = pd.DataFrame({'image_path': all_image_paths})
    
    # Trả về DataFrame chứa image_paths và chuỗi translated_text_all
    return df_image_paths, translated_text_all, len(sub_prompts)
        

#---------------------------------------------------------

def main12(prompt_vietnamese, objects, top_k=500, flag=True, base_gap=10):

    if "/" in prompt_vietnamese:
        sub_prompts = [sub.strip() for sub in prompt_vietnamese.split('/')]
        prompt_vietnamese = prompt_vietnamese.replace("/", ", ")
    else:
        sub_prompts = [prompt_vietnamese.strip()]
    
    dataset_subprompts = []
    translated_text_all = ""
    
    # Nếu không có objects hoặc objects rỗng, áp dụng main3 cho tất cả các prompt
    if objects is None or objects == '':
        for prompt in sub_prompts:
            dataset_subprompt, translated_text = main3(prompt, top_k=top_k, plot=False)
            translated_text_all += str(translated_text) + "// "
            dataset_subprompts.append(dataset_subprompt)
    else:
        # Nếu có objects, tách thành các câu
        sub_sens = [sub.strip() for sub in objects.split(';')]
        
        # Nếu số lượng prompts và objects không bằng nhau, in ra thông báo
        if len(sub_prompts) != len(sub_sens):
            print(f"Warning: Number of prompts ({len(sub_prompts)}) and objects ({len(sub_sens)}) are not equal.")
        
        # Xử lý các cặp prompt và sen
        for prompt, sen in zip(sub_prompts, sub_sens):
            if sen == '':
                # Nếu sen rỗng, sử dụng main3
                dataset_subprompt, translated_text = main3(prompt, top_k=top_k, plot=False)
            else:
                # Nếu sen không rỗng, sử dụng main4
                dataset_subprompt, translated_text = main1(prompt, sen, top_k=top_k)
                
            translated_text_all += str(translated_text) + "// "
            dataset_subprompts.append(dataset_subprompt)

    # Thực hiện temporal search
    result = temporal_search_plus(dataset_subprompts, base_gap=base_gap)
    
    # Chỉ giữ lại những item có số lượng image_paths bằng số lượng sub_prompts
    all_image_paths = []
    for item in result:
        if len(item['image_paths']) == len(sub_prompts):
            all_image_paths.extend(item['image_paths'])  # Thêm tất cả image_paths từ mỗi item vào danh sách
    
    df_image_paths = pd.DataFrame({'image_path': all_image_paths})
    
    # Trả về DataFrame chứa image_paths và chuỗi translated_text_all
    return df_image_paths, translated_text_all, len(sub_prompts)


def query_img(images, top_k=500, index=index, metadata_df=metadata_df, base_gap=10):
    """
    Query the index with features extracted from multiple images, 
    inner join results on `image_path`, scale and sum scores, then sort.
    
    :param images: List of PIL Image objects
    :param top_k: Number of top results to retrieve for each image
    :param index: The search index for querying
    :param metadata_df: Metadata DataFrame to fetch results
    :return: DataFrame with `image_path` and final `score` for all images
    """
    all_results = []
    images = images.replace('"', '')
    images = images.replace("\\", "/")
    
    if "," in images:
        sub_prompts = [sub.strip() for sub in images.split(',')]
    else:
        sub_prompts = [images.strip()]
    # Iterate over each image
    for image in sub_prompts:
        if not os.path.exists(image):
            print(f"Error: File not found - {image}")
            continue 
        # Extract features for the current image
        img_features = extract_image_features([image])  # Assuming extract_image_features works on batches
        img_features = img_features.cpu().numpy().astype(np.float16)
        
        # Perform the search with usearch (simulating index search)
        matches = index.search(img_features, top_k)  # Simulate index search for current image
        
        # Retrieve the top indices and their corresponding scores
        top_indices = matches.keys  # Assuming matches.keys is a 1D array
        top_scores = matches.distances  # Similarly, assuming matches.distances is a 1D array
        
        # Retrieve the top results from metadata_df using the indices
        top_results = metadata_df.iloc[top_indices].copy()
        
        # Add the scores to the results DataFrame and tag them by image_path
        top_results['score'] = top_scores
        top_results['image_path'] = top_results['image_path']  # Ensure we have `image_path` for join
        
        # Append the current image's results to the overall results list
        all_results.append(top_results)
    
    if len(sub_prompts) > 1:
        # Thực hiện temporal search
        result = temporal_search_plus(all_results, base_gap=base_gap)
        
        # Chỉ giữ lại những item có số lượng image_paths bằng số lượng sub_prompts
        all_image_paths = []
        for item in result:
            if len(item['image_paths']) == len(sub_prompts):
                all_image_paths.extend(item['image_paths'])  # Thêm tất cả image_paths từ mỗi item vào danh sách
        
        df_image_paths = pd.DataFrame({'image_path': all_image_paths})

        return df_image_paths, "", len(sub_prompts)

    else:
        return top_results, "", 5




        
def suppress_output():
    return redirect_stdout(sys.stdout), redirect_stderr(sys.stderr)

def run_main_function(main_option, prompt_vietnamese, sentence, objects, audio, images):
    if main_option == "Only img":
        a, b = main3(prompt_vietnamese, top_k=500, plot=False)
        return a, b, None  # Return None for img_per_row
    elif main_option == "Img & Text":
        a, b = main4(prompt_vietnamese, sentence, top_k=5000)
        return a, b, None  # Return None for img_per_row
    elif main_option == "Only text":
        a = query_with_text_emb(sentence, top_k=100, index=index_sen, metadata_df=metadata_df_sen, model=model_emb)
        return a, "", None
    elif main_option == "Temporal Search":
        df_image_paths, translated_text_all, img_per_row = main9(prompt_vietnamese, sentence, top_k=top_k, base_gap=base_gap)
        if img_per_row == 1:
            img_per_row = 5
        return df_image_paths, translated_text_all, img_per_row
    elif main_option == "Temporal Search(+)":
        df_image_paths, translated_text_all, img_per_row = main10(prompt_vietnamese, sentence, top_k=top_k, base_gap=base_gap)
        if img_per_row == 1:
            img_per_row = 5
        return df_image_paths, translated_text_all, img_per_row
    elif main_option == "Object":
        a, b = object(prompt_vietnamese=objects)
        return a, b, None  # Return None for img_per_row
    elif main_option == "Audio":
        a, b = audio_func(input_text=audio)
        return a, b, None

    elif main_option == "Img & Object":
        a, b = main1(prompt_vietnamese, objects, top_k=8000)
        return a, b, None

    elif main_option == "Temporal Search(+) & Obj":
        df_image_paths, translated_text_all, img_per_row = main12(prompt_vietnamese, top_k=top_k, base_gap=base_gap, objects=objects)
        if img_per_row == 1:
            img_per_row = 5
    
    elif main_option == "Google img":
        a,b,c = query_img(images, top_k=top_k, base_gap=base_gap)
        return a, b,c  # Return None for img_per_row


# Hàm thay đổi đường dẫn của ảnh
def change_path_img(image_paths, dir_img='/kaggle/input/aic-frames/output', local=True):
    if local:
        # Thay thế phần đầu của đường dẫn bằng dir_img, giữ lại tên file
        image_paths = [os.path.join(dir_img, os.path.basename(path)) for path in image_paths]
    return image_paths
    
    
def plot_images_from_csv(df, images_per_row=5, dir_img='', local=True):
    image_paths = df['image_path'].tolist()
    image_paths = change_path_img(image_paths, dir_img, local)
    num_images = len(image_paths)
    
    # Display images in a grid using Streamlit's st.image
    for i in range(0, num_images, images_per_row):
        cols = st.columns(images_per_row)
        for j, col in enumerate(cols):
            if i + j < num_images:
                image_path = image_paths[i + j]
                try:
                    # Open the image
                    image = Image.open(image_path)
                    
                    # Extract the filename and format the caption
                    base_filename = os.path.basename(image_path)
                    # Remove the extension and split by the dot
                    name_parts = base_filename.rsplit('.', 1)[0].split('.')
                    if len(name_parts) == 2:
                        formatted_caption = f"{name_parts[0]}, {name_parts[1]}"
                    else:
                        formatted_caption = base_filename  # Fallback to the original if unexpected format
                    
                    # Display the image with the formatted caption
                    col.image(image, caption=formatted_caption, use_column_width=True)
                except Exception as e:
                    col.error(f"Error loading image: {e}")

def convert_df_to_csv(df: pd.DataFrame, text: str = "") -> str:
    # Giữ lại cột 'image_path'
    df = df[['image_path']].copy()

    # Thay thế '.' bằng ', ' và loại bỏ phần mở rộng '.jpg'
    df['image_path'] = df['image_path'].apply(lambda x: re.sub(r'\.', ', ', x.split('/')[-1].replace('.jpg', '')))

    # Append the provided text to each line in the CSV, only if text is not empty
    if text:
        df['image_path'] = df['image_path'].apply(lambda x: f"{x}, {str(text)}")
    else:
        df['image_path'] = df['image_path'].apply(lambda x: f"{x}")

    # Chuyển đổi DataFrame thành chuỗi CSV mà không có bất kỳ dấu " hoặc \ nào
    csv_data = '\n'.join(df['image_path'].tolist())
    

    return csv_data

def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    import base64
    import pickle
    import uuid
    import re

    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None
    else:
        if isinstance(object_to_download, bytes):
            pass
        elif isinstance(object_to_download, pd.DataFrame):
            # Get the text input from Streamlit
            text_to_append = st.session_state.get("text_to_append", "")
            object_to_download = convert_df_to_csv(object_to_download, text_to_append)
        else:
            object_to_download = json.dumps(object_to_download)

    # Encode the object to base64
    try:
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    # Generate a unique button ID
    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    # Define custom CSS for the download button
    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    # Create the download link using base64 encoding
    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/csv;base64,{b64}">{button_text}</a><br></br>'

    return dl_link

# Layout setup
col1, col2 = st.columns(2)
with col1:
    prompt_vietnamese = st.text_area(
        "Enter Vietnamese prompt:",
        key="prompt_vietnamese",
        height=200  # Adjust the height as needed
    )
    text_to_append = st.text_input("Enter text to append to CSV:", key="text_to_append")
with col2:
    sentence = st.text_input("Enter sentence (OCR):", key="sentence")
    objects = st.text_input("Enter objects:", key="objects")
    audio = st.text_area(
        "Enter audio:",
        key="audio",
        height=35  # Adjust the height as needed
    )


main_option = st.selectbox(
    "Choose a retrieval method:",
    # ("Only img", "Only text", "Img & Text", "Img & Object", "Temporal Search", "Temporal Search(+)","Object","Audio"),
    ("Only img", "Only text", "Img & Text", "Img & Object", "Temporal Search(+)","Object", "Google img","Audio", "Temporal Search(+) & Obj"),

    key="main_option"
)


images = st.text_area(
    "Enter path IMG(s):",
    key="images",
    height=35
)



# Display sliders only when "Temporal Search" is selected
if main_option == "Temporal Search(+) & Obj" or main_option == "Temporal Search(+)" or  main_option == "Google img":
    top_k = st.slider("Select top_k:", min_value=0, max_value=3500, value=1200, step=100, key="top_k_slider")
    base_gap = st.slider("Select base_gap:", min_value=0, max_value=350, value=35, step=1, key="base_gap_slider")
else:
    top_k = None
    base_gap = None
    


st.markdown("---")  # Horizontal rule for separation


# Process the inputs and run the selected function when the "Run Retrieval" button is clicked
if st.button("🔍 Run Retrieval"):
    start_time = time.time()

    with st.spinner("Processing..."):
        result_df, translated_text, img_per_row = run_main_function(main_option, prompt_vietnamese, sentence, objects, audio, images)
        st.session_state["result_df"] = result_df
        st.session_state["images_plotted"] = False  # Reset image plot state
        st.session_state["more_images_plotted"] = False  # Track if more images are plotted

    processing_time = time.time() - start_time

    # Ensure the result_df has at most 100 rows initially for plotting
    result_df_plot = result_df.head(100)
    st.session_state["result_df_plot"] = result_df_plot

    # Store remaining images only if result_df has more than 100 rows
    if len(result_df) > 100:
        if len(result_df) > 400:
            st.session_state["remaining_images"] = result_df.iloc[100:400]
        else:
            st.session_state["remaining_images"] = result_df.iloc[100:]
    else:
        st.session_state["remaining_images"] = None  # No remaining images

    # Display the translated text
    st.markdown("### Translated Text")
    st.write(translated_text)

    # Display the download button (limit to 100 rows)
    download_result_df = result_df.head(100)  # Only keep the first 100 rows for the CSV download
    download_button_str = download_button(download_result_df, "retrieval_results.csv", "Download Results as CSV")
    st.markdown(download_button_str, unsafe_allow_html=True)

    # Plot the first 100 retrieved images
    plot_start_time = time.time()
    st.markdown("### 🎨 Retrieved Images")
    plot_images_from_csv(result_df_plot, images_per_row=img_per_row if img_per_row else 5, dir_img=dir_img, local=local)
    st.session_state["images_plotted"] = True

    plotting_time = time.time() - plot_start_time
    total_time_with_plotting = processing_time + plotting_time

    st.write(f"**Total processing time (without plotting):** {processing_time:.2f} seconds")
    st.write(f"**Total plotting time:** {plotting_time:.2f} seconds")
    st.write(f"**Total execution time (with plotting):** {total_time_with_plotting:.2f} seconds")
    st.markdown("---")


# Button to plot more images, only show if there are remaining images
if st.session_state.get("remaining_images") is not None and st.button("🖼️ More img"):
    remaining_images = st.session_state["remaining_images"]

    if not st.session_state.get("more_images_plotted", False):
        st.markdown("### 🎨 Additional Retrieved Images")
        plot_images_from_csv(remaining_images, images_per_row= 5, dir_img=dir_img, local=local)
        st.session_state["more_images_plotted"] = True


# Ensure the images remain plotted when interacting with the download button
if "result_df" in st.session_state:
    result_df_plot = st.session_state["result_df_plot"]

    if not st.session_state.get("images_plotted", False):
        st.markdown("### 🎨 Retrieved Images")
        plot_images_from_csv(result_df_plot, images_per_row=5, dir_img=dir_img, local=local)
        st.session_state["images_plotted"] = True

    # Ensure only the first 100 rows are available for download
    download_result_df = st.session_state["result_df"].head(100)  
    download_button_str = download_button(download_result_df, "retrieval_results.csv", "Download Results as CSV")
    st.markdown(download_button_str, unsafe_allow_html=True)
