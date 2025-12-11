import streamlit as st
import os
import subprocess
import sys
import shutil
import gc
import time
from pathlib import Path
import io
import stat
import matplotlib.pyplot as plt
import tempfile
import json
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
import uuid
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(layout="wide")

# Google GenAI imports for AI recommendations with Gemini
try:
    from google import genai
    from google.genai import types
    print("✓ Google GenAI libraries found")
except ImportError:
    st.error("Google GenAI libraries not found. Please install with: pip install google-generativeai")
    st.stop()

# Whisper, Azure Custom Vision, and ffmpeg imports
try:
    import whisper
    print("✓ Whisper library found")
except ImportError:
    st.error("Whisper library not found. Please install with: pip install openai-whisper")
    st.stop()

try:
    from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
    from msrest.authentication import ApiKeyCredentials
    print("✓ Azure Custom Vision libraries found")
except ImportError as e:
    st.error(f"Azure Custom Vision libraries not found. Please install required packages.")
    st.stop()

try:
    import ffmpeg
    print("✓ ffmpeg-python library found")
except ImportError:
    st.error("ffmpeg-python library not found. Please install with: pip install ffmpeg-python")
    st.stop()

# Configuration
CATEGORY_COLORS = {
    "Visual and Audio Branding": '#1A6D35',
    "Audio Branding Only": '#00B050',
    "Visual Branding Only": '#B8E4BE',
    "No Branding Present": '#FF0000'
}

BRAND_VARIATIONS = {
    "Bounce": ["bounce", "bounces", "bouncy"],
    "Downy": ["downy", "downey", "dumming", "downie","down"],
    "Gain": ["gain", "gains", "gane", "gayne", "game"],
    "Tide": ["tide", "tied", "tyde", "tyde","todd"],
    "Unstopables": ["unstopables", "unstoppables", "unstoppable", "unstopable", "unstopabls","star bubbles"]
}

BRAND_DISPLAY_MAP = {
    "Bounce": "Bounce",
    "Downy Beads (excl. Unstopables)": "Downy",
    "Downy LFE": "Downy",
    "Downy Rinse": "Downy",
    "Downy (Total)": "Downy",
    "Gain FE": "Gain",
    "Gain LND": "Gain",
    "Gain Mega (LND + FE)": "Gain",
    "Tide": "Tide",
    "Unstopables": "Unstopables",
}

MEDIA_VEHICLES = [
    "Facebook",
    "Facebook Reels",
    "Instagram",
    "Instagram Reels",
    "Linear TV (15s)",
    "Linear TV (30s)",
    "Online Video (OLV)",
    "OTT (15s)",
    "OTT (30s)",
    "Rewarded OLV",
    "TikTok",
    "YouTube Bumper (6s)",
    "YouTube Non-Skip (15s)"
]

attention_df = pd.read_csv("attention_data.csv")

# Azure Custom Vision Configuration
PREDICTION_KEY = st.secrets["AZURE_PREDICTION_KEY"]
ENDPOINT = st.secrets["AZURE_ENDPOINT"]
PROJECT_ID = st.secrets["AZURE_PROJECT_ID"]
MODEL_NAME = st.secrets["AZURE_MODEL_NAME"]

# Google Gemini API Key Configuration
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Google Sheets IDs for AI Enhancement Logging
FEATURE_EXTRACTION_SHEET_ID = "1_yqI957bORGGTqGwN6CL8B31uXAxfnghGRYp_bzv95Y"
CRITERIA_EXTRACTION_SHEET_ID = "1Ai0SkhYrn2XwLbPgCYO9ILzbvewO_Js1Q41qPSA4DWE"
SUMMARY_BEST_PRACTICE_SHEET_ID = "1-psaUTnXRLJe7Bu7C_JPrxurvxaKsvU9Li0KT7zZy-o"
AI_RECOMMENDATION_SHEET_ID = "1Y8RMwiuIAtvKOUP8MPhT2Si3nQKh1T8UEmb4lr9BcWE"

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": """Welcome to the Project Sonar Tool!

I'll help you analyze your video and provide recommendations for media placement.

Please provide the following:
1. **Video File** (.mp4 format)
2. **Brand** (select from dropdown)
3. **Intended Media Vehicle** (where you plan to place this video)

Once you upload these, I'll analyze your video then provide AI-powered recommendations!"""
    })

# Google Sheets Connection
def get_gsheet_client():
   """Authenticate using Streamlit secrets."""
   scopes = [
       "https://www.googleapis.com/auth/spreadsheets",
       "https://www.googleapis.com/auth/drive"
   ]
   
   service_account_info = dict(st.secrets["gcp_service_account"])
   
   creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
   client = gspread.authorize(creds)
   return client

def append_to_gsheet(sheet_id, rows):
    """
    Append rows to a Google Sheet.
    - sheet_id: the Google Sheet ID (from its URL)
    - rows: list of lists, where each sublist is a row
    """
    client = get_gsheet_client()
    sheet = client.open_by_key(sheet_id).sheet1  # assumes first tab
    sheet.append_rows(rows, value_input_option="USER_ENTERED")

# Helper functions
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_video_duration(video_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
               '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = int(float(result.stdout.strip()))
        return duration
    except Exception as e:
        st.error(f"Error getting video duration: {e}")
        return None

def load_features_table():
   """Load features table from CSV file"""
   try:
       features_df = pd.read_csv("features.csv")
       return features_df
   except Exception as e:
       st.error(f"Error loading features table: {e}")
       return None
def load_creative_best_practices_table():
   """Load creative best practices table from CSV file"""
   try:
       practices_df = pd.read_csv("creative_best_practices.csv")
       return practices_df
   except Exception as e:
       st.error(f"Error loading creative best practices table: {e}")
       return None
def get_criteria_for_media_vehicle(media_vehicle, practices_df):
   """Filter creative best practices by media vehicle"""
   try:
       filtered_df = practices_df[practices_df['media_vehicle'] == media_vehicle]
       criteria_list = []
       for _, row in filtered_df.iterrows():
           criteria_list.append({
               'criteria_id': row['criteria_id'],
               'criteria_name': row['criteria_name'],
               'description': row['description']
           })
       return criteria_list
   except Exception as e:
       st.error(f"Error filtering criteria: {e}")
       return []

def upload_video_to_gemini(video_path, google_api_key):
   """
   Upload video to Gemini's File API for analysis.
   Args:
       video_path: Path to the video file
       google_api_key: Google API key
   Returns:
       uploaded_file: Uploaded file object, or None if failed
   """
   try:
       client = genai.Client(api_key=google_api_key)
       # Upload the video file (pass path as string to file parameter)
       uploaded_file = client.files.upload(file=video_path)
       # Wait for file to be processed
       while uploaded_file.state.name == "PROCESSING":
           time.sleep(2)
           uploaded_file = client.files.get(name=uploaded_file.name)
       if uploaded_file.state.name == "FAILED":
           st.error("Video upload to Gemini failed")
           return None
       return uploaded_file
   except Exception as e:
       st.error(f"Error uploading video to Gemini: {e}")
       return None
       
# Audio Extraction + Analysis
def extract_audio(video_path, output_audio_path):
    try:
        cmd = ['ffmpeg', '-i', video_path, '-y', '-vn', '-acodec', 'pcm_s16le', 
               '-ar', '16000', '-ac', '1', output_audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if os.path.exists(output_audio_path):
            return output_audio_path
        return None
    except subprocess.CalledProcessError as e:
        st.error(f"Audio extraction failed: {e}")
        return None

def transcribe_with_whisper(audio_path, model_size="small"):
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path, word_timestamps=True)
        return result
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None

def analyze_audio_branding(transcription_result, brand_name, brand_variations, duration):
    audio_results = {}
    for second in range(1, duration + 1):
        audio_results[second] = {
            'detected': 0,
            'mentions': [],
            'status': 'NO BRAND'
        }
    
    variations_lower = [var.lower() for var in brand_variations]
    
    for segment in transcription_result["segments"]:
        if "words" in segment:
            for word_info in segment["words"]:
                word = word_info["word"].strip().lower().replace(",", "").replace(".", "").replace("!", "").replace("?", "")
                
                word_matches = False
                matched_variation = None
                
                if word in variations_lower:
                    word_matches = True
                    matched_variation = word
                else:
                    for variation in variations_lower:
                        if (variation in word and len(variation) >= 3) or (word in variation and len(word) >= 3):
                            word_matches = True
                            matched_variation = variation
                            break
                
                if word_matches:
                    start_time = word_info.get("start", 0)
                    second = int(start_time) + 1
                    
                    if 1 <= second <= duration:
                        audio_results[second]['detected'] = 1
                        audio_results[second]['status'] = 'BRAND DETECTED'
                        audio_results[second]['mentions'].append({
                            'word': word_info["word"],
                            'matched_variation': matched_variation,
                            'timestamp': start_time
                        })
    
    return audio_results

def handle_remove_readonly(func, path, exc):
    if os.path.exists(path):
        os.chmod(path, stat.S_IWRITE)
        func(path)

def safe_remove_directory(directory_path, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            if os.path.exists(directory_path):
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.chmod(file_path, stat.S_IWRITE)
                        except:
                            pass
                gc.collect()
                time.sleep(0.5)
                shutil.rmtree(directory_path, onerror=handle_remove_readonly)
                return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                return False
    return False

# Visual Extraction + Analysis
def extract_frames(video_path, output_dir):
    try:
        if os.path.exists(output_dir):
            if not safe_remove_directory(output_dir):
                import random
                output_dir = f"temp_frames_{random.randint(1000, 9999)}"
        
        os.makedirs(output_dir)
        
        cmd = ['ffmpeg', '-i', video_path, '-y', '-vf', 'fps=1', 
               os.path.join(output_dir, 'frame_%04d.jpg')]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        frame_count = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
        return output_dir, frame_count
        
    except subprocess.CalledProcessError as e:
        st.error(f"Frame extraction failed: {e}")
        return None, 0

def analyze_visual_branding(frames_dir, brand_tag, duration, prediction_key, endpoint, project_id, model_name):
    visual_results = {}
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(endpoint, credentials)
    
    for i in range(1, duration + 1):
        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.jpg')
        
        if not os.path.exists(frame_path):
            visual_results[i] = {
                'detected': 0, 
                'confidence': 0.0, 
                'status': 'Frame not found',
                'predictions': []
            }
            continue

        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                with open(frame_path, 'rb') as image_file:
                    image_data = image_file.read()
                
                predictions = predictor.detect_image(project_id, model_name, image_data)
                
                brand_detected = False
                max_confidence = 0.0
                all_predictions = []
                
                for prediction in predictions.predictions:
                    pred_info = {
                        'tag': prediction.tag_name,
                        'confidence': prediction.probability,
                        'bounding_box': {
                            'left': prediction.bounding_box.left,
                            'top': prediction.bounding_box.top,
                            'width': prediction.bounding_box.width,
                            'height': prediction.bounding_box.height
                        }
                    }
                    all_predictions.append(pred_info)
                    
                    if prediction.tag_name.lower() == brand_tag.lower():
                        if prediction.probability > 0.05:
                            brand_detected = True
                            max_confidence = max(max_confidence, prediction.probability)
                
                detected = 1 if brand_detected else 0
                status = "BRAND DETECTED" if detected else "NO BRAND"
                
                visual_results[i] = {
                    'detected': detected,
                    'confidence': max_confidence,
                    'status': status,
                    'predictions': all_predictions
                }
                
                break
                
            except Exception as e:
                error_message = str(e)
                if "Too Many Requests" in error_message or "429" in error_message:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 1.5
                        continue
                    else:
                        visual_results[i] = {
                            'detected': 0, 
                            'confidence': 0.0, 
                            'status': 'Rate limit exceeded',
                            'predictions': []
                        }
                        break
                else:
                    visual_results[i] = {
                        'detected': 0, 
                        'confidence': 0.0, 
                        'status': f'Error: {e}',
                        'predictions': []
                    }
                    break
            finally:
                gc.collect()
                time.sleep(1.2)
        
    return visual_results

# Combine audio_results and visual_results
def categorize_branding(audio_results, visual_results, duration):
    final_categories = {}
    
    for second in range(1, duration + 1):
        audio_detected = audio_results.get(second, {}).get('detected', 0)
        visual_detected = visual_results.get(second, {}).get('detected', 0)
        
        if audio_detected == 1 and visual_detected == 1:
            category = "Visual and Audio Branding"
        elif audio_detected == 1 and visual_detected == 0:
            category = "Audio Branding Only"
        elif audio_detected == 0 and visual_detected == 1:
            category = "Visual Branding Only"
        else:
            category = "No Branding Present"
        
        final_categories[second] = category
    
    return final_categories

# Create Visualizations
def create_timeline_visualization(final_categories, duration):
    fig_width = max(12, duration * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, 3))
    fig.patch.set_facecolor('white')
    
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Branding Analysis Timeline")
    
    if duration > 60:
        ax.set_xticks(range(0, duration + 1, 5))
    elif duration > 30:
        ax.set_xticks(range(0, duration + 1, 2))
    else:
        ax.set_xticks(range(0, duration + 1, 1))
    
    for second, category in final_categories.items():
        color = CATEGORY_COLORS[category]
        rect = plt.Rectangle((second - 1, 0), 1, 1, 
                           facecolor=color, 
                           edgecolor='black', 
                           linewidth=0.5)
        ax.add_patch(rect)
    
    ax.grid(True, axis='x', alpha=0.3, linewidth=0.5)
    
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', label=category) 
                      for category, color in CATEGORY_COLORS.items()]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    return fig

def create_summary_visualization(final_categories, duration):
    """Create summary bar chart showing percentage of time in each category"""
    print("Creating summary visualization...")
    
    category_counts = {cat: 0 for cat in CATEGORY_COLORS}
    for category in final_categories.values():
        category_counts[category] += 1
    
    percentages = {cat: (count / duration) * 100 if duration > 0 else 0 
                  for cat, count in category_counts.items()}
    
    fig_summary, ax_summary = plt.subplots(figsize=(12, 6))
    fig_summary.patch.set_facecolor('#f0f0f0')
    
    labels = list(percentages.keys())
    values = list(percentages.values())
    bar_colors = [CATEGORY_COLORS[label] for label in labels]
    
    bars = ax_summary.bar(labels, values, color=bar_colors)
    ax_summary.set_ylabel("Percentage of Total Seconds (%)")
    ax_summary.set_title("Branding Analysis Distribution")
    ax_summary.set_ylim(0, 100)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax_summary.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{round(value)}%', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax_summary.get_xticklabels(), rotation=15, ha="right")
    fig_summary.tight_layout()
    
    return fig_summary

# Incorporate Attention Data
def classify_branding_with_attention(visual_detected, audio_detected, is_attentive):
    """
    Classify branding based on visual/audio detection and attention data.
    
    Rules:
    - If attentive:
        Visual & Audio → Visual & Audio Branding
        Visual only → Visual Branding Only
        Audio only → Audio Branding Only
        None → No Branding Present
    - If NOT attentive:
        Visual & Audio → Audio Branding Only (visual missed, audio persists)
        Visual only → Not counted (visual missed)
        Audio only → Audio Branding Only
        None → Not counted
    
    Returns: branding category string or None if not counted
    """
    if is_attentive:
        # Attentive second - both visual and audio can be perceived
        if visual_detected and audio_detected:
            return "Visual and Audio Branding"
        elif visual_detected and not audio_detected:
            return "Visual Branding Only"
        elif audio_detected and not visual_detected:
            return "Audio Branding Only"
        else:
            return "No Branding Present"
    else:
        # Non-attentive second - only audio can be perceived
        if audio_detected:
            return "Audio Branding Only"
        else:
            # Visual without audio is missed, no branding is missed
            return None  # Not counted as an attentive second

# Get Attention Data Metrics
def get_attention_metrics(media_vehicle, attention_df):
    """
    Retrieve attention metrics for a given media vehicle from the attention dataframe.
    
    Args:
        media_vehicle: string, the media vehicle name
        attention_df: pandas DataFrame with columns:
            - media_vehicle
            - media_form ("Short Form or Social" or "Long Form")
            - rounded_average_attentive_seconds
            - rounded_average_watch_time (only for Short Form)
    
    Returns:
        dict with 'media_form', 'avg_attentive_seconds', and optionally 'rounded_avg_watch_time'
    """
    vehicle_data = attention_df[attention_df['media_vehicle'] == media_vehicle]
    
    if vehicle_data.empty:
        return None
    
    row = vehicle_data.iloc[0]
    
    result = {
        'media_form': row['media_form'],
        'avg_attentive_seconds': row['rounded_average_attentive_seconds']
    }
    
    # Add watch time if it exists (Short Form only)
    if 'rounded_average_watch_time' in row and pd.notna(row['rounded_average_watch_time']):
        result['rounded_avg_watch_time'] = row['rounded_average_watch_time']
    
    return result

# Analyze "Short Form or Social" Attention and Branding
def analyze_short_form_with_watch_time(final_categories, audio_results, visual_results, 
                                        avg_attentive_seconds, rounded_avg_watch_time, duration):
    """
    Analyze branding for short-form media using rounded_average_watch_time.
    
    Logic:
    - Attentive seconds = first avg_attentive_seconds (eyes on screen)
    - Watch time seconds = first rounded_avg_watch_time (total viewing)
    - Non-attentive but within watch time: Audio branding can still count
    - Beyond watch time: All seconds are non-attentive and don't count
    
    Attentive branding score = branded seconds / rounded_avg_watch_time
    """
    attentive_seconds = min(int(round(avg_attentive_seconds)), duration)
    watch_time_seconds = min(int(round(rounded_avg_watch_time)), duration)
    
    # Initialize counters
    branding_counts = {
        "Visual and Audio Branding": 0,
        "Visual Branding Only": 0,
        "Audio Branding Only": 0,
        "No Branding Present": 0
    }
    
    branded_seconds_in_watch_time = 0
    
    # Process each second up to watch time
    for second in range(1, watch_time_seconds + 1):
        is_attentive = (second <= attentive_seconds)
        
        visual_detected = visual_results.get(second, {}).get('detected', 0) == 1
        audio_detected = audio_results.get(second, {}).get('detected', 0) == 1
        
        if is_attentive:
            # Eyes on screen - both visual and audio can be perceived
            if visual_detected and audio_detected:
                branding_counts["Visual and Audio Branding"] += 1
                branded_seconds_in_watch_time += 1
            elif visual_detected and not audio_detected:
                branding_counts["Visual Branding Only"] += 1
                branded_seconds_in_watch_time += 1
            elif audio_detected and not visual_detected:
                branding_counts["Audio Branding Only"] += 1
                branded_seconds_in_watch_time += 1
            else:
                branding_counts["No Branding Present"] += 1
        else:
            # Non-attentive but within watch time - only audio counts
            if audio_detected:
                # Visual+Audio or Audio Only both become Audio Only
                branding_counts["Audio Branding Only"] += 1
                branded_seconds_in_watch_time += 1
            else:
                # No audio = no branding counted
                branding_counts["No Branding Present"] += 1
    
    # Calculate attentive branding score using watch time as denominator
    attentive_branding_score = (branded_seconds_in_watch_time / watch_time_seconds * 100 
                                if watch_time_seconds > 0 else 0)
    
    # Calculate distribution percentages based on watch time
    attentive_branding_distribution = {
        cat: (count / watch_time_seconds * 100 if watch_time_seconds > 0 else 0)
        for cat, count in branding_counts.items()
    }
    
    return {
        'attentive_branding_score': attentive_branding_score,
        'attentive_branding_distribution': attentive_branding_distribution,
        'branding_counts': branding_counts,
        'total_classified_seconds': watch_time_seconds,
        'attentive_seconds': attentive_seconds,
        'watch_time_seconds': watch_time_seconds,
        'media_form': 'Short Form or Social',
        'audio_results': audio_results,
        'visual_results': visual_results  
    }

# Analyze "Long Form" Attention and Branding
def analyze_long_form_simplified(final_categories, audio_results, visual_results, 
                                 avg_attentive_seconds, duration):
    """
    Analyze branding for long-form media using simplified calculation.
    
    Formula: attentive_branding_score = (branding_percentage * rounded_avg_attentive_seconds) / rounded_avg_attentive_seconds
    
    No Monte Carlo simulations for the score itself.
    """
    # Calculate overall branding percentage
    branding_present_count = sum(1 for cat in final_categories.values() 
                                 if cat != "No Branding Present")
    branding_percentage = (branding_present_count / duration * 100) if duration > 0 else 0
    
    # Apply the simplified formula
    attentive_branding_score = (branding_percentage * avg_attentive_seconds) / avg_attentive_seconds
    
    # For distribution, we'll use the overall video distribution as reference
    category_counts = {cat: 0 for cat in CATEGORY_COLORS}
    for category in final_categories.values():
        category_counts[category] += 1
    
    attentive_branding_distribution = {
        cat: (count / duration * 100 if duration > 0 else 0)
        for cat, count in category_counts.items()
    }
    
    return {
        'attentive_branding_score': attentive_branding_score,
        'attentive_branding_distribution': attentive_branding_distribution,
        'branding_percentage': branding_percentage,
        'attentive_seconds': int(round(avg_attentive_seconds)),
        'media_form': 'Long Form'
    }

# Monte Carlo Simulations for 3 Simulated Viewing Experiences (Long Form)
def generate_long_form_viewing_simulations(audio_results, visual_results, 
                                           avg_attentive_seconds, duration, n_simulations=3):
    """
    Generate 3 Monte Carlo simulations of viewing experiences for long-form content.
    
    Each simulation randomly selects avg_attentive_seconds and creates a timeline
    showing attentive/non-attentive seconds with branding.
    
    Returns: list of 3 simulation results, each containing second-by-second categorization
    """
    np.random.seed(42)  # For reproducibility
    
    simulations = []
    attentive_count = min(int(round(avg_attentive_seconds)), duration)
    all_seconds = list(range(1, duration + 1))
    
    for sim_num in range(n_simulations):
        # Randomly select attentive seconds (without replacement)
        attentive_second_indices = set(np.random.choice(all_seconds, 
                                                        size=attentive_count, 
                                                        replace=False))
        
        # Categorize each second for this simulation
        sim_categories = {}
        
        for second in range(1, duration + 1):
            is_attentive = (second in attentive_second_indices)
            
            visual_detected = visual_results.get(second, {}).get('detected', 0) == 1
            audio_detected = audio_results.get(second, {}).get('detected', 0) == 1
            
            if is_attentive:
                # Eyes on screen
                if visual_detected and audio_detected:
                    sim_categories[second] = "Visual and Audio Branding"
                elif visual_detected and not audio_detected:
                    sim_categories[second] = "Visual Branding Only"
                elif audio_detected and not visual_detected:
                    sim_categories[second] = "Audio Branding Only"
                else:
                    sim_categories[second] = "No Branding Present"
            else:
                # Non-attentive - only audio counts
                if audio_detected:
                    sim_categories[second] = "Audio Branding Only (Non-Attentive)"
                else:
                    sim_categories[second] = "Non-Attentive"
        
        simulations.append({
            'simulation_number': sim_num + 1,
            'categories': sim_categories,
            'attentive_second_indices': attentive_second_indices
        })
    
    return simulations

def create_stacked_long_form_simulation_timelines(simulations, duration):
    """
    Create ONE visualization that stacks the 3 long-form simulation timelines
    vertically with a single shared legend in the top-right *outside* the plots.
    """
    n_sims = len(simulations)

    fig_width = max(12, duration * 0.25)   # slightly narrower to allow right margin
    fig, axes = plt.subplots(n_sims, 1, figsize=(fig_width, 3 * n_sims), sharex=True)

    if n_sims == 1:
        axes = [axes]

    non_attentive_color = '#000000'
    audio_non_attentive_color = '#00B050'

    for ax, simulation in zip(axes, simulations):
        categories = simulation['categories']

        ax.set_xlim(0, duration)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(f"Viewing Simulation #{simulation['simulation_number']}")

        # Restore second labels for each subplot
        ax.set_xticks(range(0, duration + 1, 1))
        ax.set_xticklabels([str(x) for x in range(0, duration + 1, 1)])
        ax.tick_params(axis='x', which='both', labelbottom=True)

        # Draw each 1-second block
        for second, category in categories.items():
            if category == "Non-Attentive":
                color = non_attentive_color
            elif category == "Audio Branding Only (Non-Attentive)":
                color = audio_non_attentive_color
            else:
                color = CATEGORY_COLORS[category]

            ax.add_patch(
                plt.Rectangle(
                    (second - 1, 0), 1, 1,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=0.5
                )
            )

        ax.grid(True, axis='x', alpha=0.3, linewidth=0.5)

    # --- SINGLE LEGEND (top-right, outside plot area) ---
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS["Visual and Audio Branding"],
                      edgecolor='black', label="Visual and Audio Branding"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS["Audio Branding Only"],
                      edgecolor='black', label="Audio Branding Only"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS["Visual Branding Only"],
                      edgecolor='black', label="Visual Branding Only"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS["No Branding Present"],
                      edgecolor='black', label="No Branding Present"),
        plt.Rectangle((0, 0), 1, 1, facecolor=non_attentive_color,
                      edgecolor='black', label="Non-Attentive"),
    ]

    fig.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(1.2, 1),   # <-- Move legend OUTSIDE plot
        borderpad=1,
        frameon=True,
        title="Legend"
    )
        
    # Reserve space on the right for the legend
    plt.subplots_adjust(right=0.85)

    plt.tight_layout()
    return fig
    
# "Short Form or Social" Attention/Branding Timeline Visualization
def create_short_form_timeline_with_watch_time(audio_results, visual_results, 
                                                avg_attentive_seconds, rounded_avg_watch_time, 
                                                duration):
    """
    Create timeline visualization for Short Form showing:
    - Attentive seconds (eyes on screen)
    - Non-attentive within watch time (greyed with audio branding if present)
    - Watch time cutoff indicator
    - Non-attentive beyond watch time (greyed out)
    """
    fig_width = max(12, duration * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, 3))
    fig.patch.set_facecolor('white')
    
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Attentive Branding Timeline (with Average Watch Time)")
    
    if duration > 60:
        ax.set_xticks(range(0, duration + 1, 5))
    elif duration > 30:
        ax.set_xticks(range(0, duration + 1, 2))
    else:
        ax.set_xticks(range(0, duration + 1, 1))
    
    attentive_seconds = min(int(round(avg_attentive_seconds)), duration)
    watch_time_seconds = min(int(round(rounded_avg_watch_time)), duration)
    
    # Color scheme
    non_attentive_color = '#000000'  # Grey
    audio_non_attentive_color = '#00B050'  # Audio branding green
    
    # Draw each second
    for second in range(1, duration + 1):
        is_attentive = (second <= attentive_seconds)
        within_watch_time = (second <= watch_time_seconds)
        
        visual_detected = visual_results.get(second, {}).get('detected', 0) == 1
        audio_detected = audio_results.get(second, {}).get('detected', 0) == 1
        
        if is_attentive:
            # Attentive - use standard branding colors
            if visual_detected and audio_detected:
                color = CATEGORY_COLORS["Visual and Audio Branding"]
            elif visual_detected:
                color = CATEGORY_COLORS["Visual Branding Only"]
            elif audio_detected:
                color = CATEGORY_COLORS["Audio Branding Only"]
            else:
                color = CATEGORY_COLORS["No Branding Present"]
        elif within_watch_time:
            # Non-attentive but within watch time
            if audio_detected:
                color = audio_non_attentive_color  # Audio branding color
            else:
                color = non_attentive_color  # Grey
        else:
            # Beyond watch time - always grey
            color = non_attentive_color
        
        rect = plt.Rectangle((second - 1, 0), 1, 1, 
                           facecolor=color, 
                           edgecolor='black', 
                           linewidth=0.5)
        ax.add_patch(rect)
    
    # Add vertical line to show watch time cutoff
    ax.axvline(x=watch_time_seconds, color='blue', linestyle='--', linewidth=2, 
               label=f'Avg Watch Time ({watch_time_seconds}s)')
    
    ax.grid(True, axis='x', alpha=0.3, linewidth=0.5)
    
    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS["Visual and Audio Branding"], 
                     edgecolor='black', label="Visual and Audio Branding"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS["Audio Branding Only"], 
                     edgecolor='black', label="Audio Branding Only"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS["Visual Branding Only"], 
                     edgecolor='black', label="Visual Branding Only"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS["No Branding Present"], 
                     edgecolor='black', label="No Branding Present"),
        plt.Rectangle((0, 0), 1, 1, facecolor=non_attentive_color, 
                     edgecolor='black', label="Non-Attentive"),
        plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2, 
                  label=f'Avg Watch Time ({watch_time_seconds}s)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    return fig

# "Long Form" Attention/Branding Timeline Visualization(s) (3)
def create_long_form_simulation_timeline(simulation_result, duration):
    """
    Create timeline visualization for a single Long Form viewing simulation.
    
    Shows attentive vs non-attentive seconds with branding categories.
    """
    fig_width = max(12, duration * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, 3))
    fig.patch.set_facecolor('white')
    
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Viewing Simulation #{simulation_result['simulation_number']}")
    
    if duration > 60:
        ax.set_xticks(range(0, duration + 1, 5))
    elif duration > 30:
        ax.set_xticks(range(0, duration + 1, 2))
    else:
        ax.set_xticks(range(0, duration + 1, 1))
    
    # Color scheme
    non_attentive_color = '#000000'  # Grey
    audio_non_attentive_color = '#00B050'  # Audio branding green
    
    categories = simulation_result['categories']
    
    # Draw each second
    for second, category in categories.items():
        if category == "Non-Attentive":
            color = non_attentive_color
        elif category == "Audio Branding Only (Non-Attentive)":
            color = audio_non_attentive_color
        else:
            color = CATEGORY_COLORS[category]
        
        rect = plt.Rectangle((second - 1, 0), 1, 1, 
                           facecolor=color, 
                           edgecolor='black', 
                           linewidth=0.5)
        ax.add_patch(rect)
    
    ax.grid(True, axis='x', alpha=0.3, linewidth=0.5)
    
    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS["Visual and Audio Branding"], 
                     edgecolor='black', label="Visual and Audio Branding"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS["Audio Branding Only"], 
                     edgecolor='black', label="Audio Branding Only"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS["Visual Branding Only"], 
                     edgecolor='black', label="Visual Branding Only"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS["No Branding Present"], 
                     edgecolor='black', label="No Branding Present"),
        plt.Rectangle((0, 0), 1, 1, facecolor=non_attentive_color, 
                     edgecolor='black', label="Non-Attentive")
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    return fig

# Attention Analysis
def perform_attention_analysis(final_categories, audio_results, visual_results, 
                                duration, media_vehicle, attention_df, video_path):
    """
    Main function to perform attention-based branding analysis.
    
    Updated for Phase 2:
    - Short Form: Uses rounded_average_watch_time
    - Long Form: Simplified calculation + 3 Monte Carlo simulations for visualization
    """
    attention_metrics = get_attention_metrics(media_vehicle, attention_df)
    
    if attention_metrics is None:
        return None
    
    media_form = attention_metrics['media_form']
    avg_attentive_seconds = attention_metrics['avg_attentive_seconds']
    
    if media_form == "Short Form or Social":
        # Check if rounded_average_watch_time exists
        if 'rounded_avg_watch_time' not in attention_metrics:
            return None
        
        rounded_avg_watch_time = attention_metrics['rounded_avg_watch_time']
        
        # Use new Short Form logic
        results = analyze_short_form_with_watch_time(
            final_categories, audio_results, visual_results, 
            avg_attentive_seconds, rounded_avg_watch_time, duration
        )
        
        # Create Short Form timeline visualization
        results['timeline_viz'] = create_short_form_timeline_with_watch_time(
            audio_results, visual_results, avg_attentive_seconds, 
            rounded_avg_watch_time, duration
        )

        # Create edited video for Short Form
        status_text = st.empty()
        status_text.text("Generating edited video...")
        
        # Generate unique identifier
        import time
        unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        
        edited_video = create_edited_video_short_form(
           video_path, results, duration, unique_id
        )
        
        if edited_video:
           results['edited_video'] = edited_video
        
        status_text.empty()
                                    
    else:  # Long Form
        # Use simplified calculation (no Monte Carlo for score)
        results = analyze_long_form_simplified(
            final_categories, audio_results, visual_results, 
            avg_attentive_seconds, duration
        )
        
        # Generate 3 Monte Carlo simulations for visualization
        simulations = generate_long_form_viewing_simulations(
            audio_results, visual_results, avg_attentive_seconds, duration, n_simulations=3
        )
        
        # Create timeline visualizations for each simulation
        results['stacked_simulation_fig'] = create_stacked_long_form_simulation_timelines(
        simulations, duration
        )
        
        results['simulations'] = simulations

                # Create edited videos for Long Form (3 simulations)
        status_text = st.empty()
       
        # Generate unique identifier
        import time
        unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        
        # Show progress for video generation
        for i in range(1, 4):
            status_text.text(f"Generating edited video {i} of 3...")
            time.sleep(0.1)  # Brief delay so user can see progress
        edited_videos = create_edited_videos_long_form(
            video_path, simulations, duration, unique_id
        )
        
        if edited_videos:
            results['edited_videos'] = edited_videos
        status_text.empty()
    
    return results

def create_attention_visualization(attention_results):
    """
    Create visualization for attention-based branding analysis.
    
    Shows the distribution of branding types during attentive moments.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#f0f0f0')
    
    distribution = attention_results['attentive_branding_distribution']
    
    labels = list(distribution.keys())
    values = list(distribution.values())
    bar_colors = [CATEGORY_COLORS[label] for label in labels]
    
    bars = ax.bar(labels, values, color=bar_colors)
    ax.set_ylabel("Percentage of Attentive Time (%)")
    ax.set_title(f"Branding Distribution During Attentive Moments")
    #ax.set_title(f"Branding Distribution During Attentive Moments ({attention_results['media_form']})")
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{value:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    
    return fig

def get_attentive_branding_rating(score, media_form):
    """
    Determine the attentive branding rating category based on score and media form.
    
    Args:
        score: float, the attentive branding score percentage
        media_form: string, either "Short Form or Social" or "Long Form"
    
    Returns:
        dict with 'rating' (category name) and 'color' (hex color code)
    """
    if media_form == "Short Form or Social":
        if score > 50:
            return {'rating': 'Exceptional', 'color': '#00B050'}
        elif score >= 50:
            return {'rating': 'Partial', 'color': '#FFFF00'}
        else:
            return {'rating': 'Insufficient', 'color': '#FF0000'}
    else:  # Long Form
        if score >= 75:
            return {'rating': 'Exceptional', 'color': '#00B050'}
        elif score >= 50:
            return {'rating': 'Strong', 'color': '#92D050'}
        elif score >= 25:
            return {'rating': 'Partial', 'color': '#FFFF00'}
        else:
            return {'rating': 'Insufficient', 'color': '#FF0000'}

# Attentive Branding Rating HTML
def display_attentive_branding_rating(score, media_form, media_vehicle):
    """
    Display the attentive branding rating using Streamlit components.
    
    Args:
        score: float, the attentive branding score percentage
        media_form: string, either "Short Form or Social" or "Long Form"
    """
    # Get the rating for this score
    rating_info = get_attentive_branding_rating(score, media_form)
    
    # Define ranges based on media form
    if media_form == "Short Form or Social":
        ranges = [
            {'label': 'Exceptional', 'range': '≥ 50%', 'color': '#00B050'},
            {'label': 'Partial', 'range': '1-50%', 'color': '#FFFF00'},
            {'label': 'Insufficient', 'range': '0%', 'color': '#FF0000'}
        ]
    else:  # Long Form
        ranges = [
            {'label': 'Exceptional', 'range': '≥ 75%', 'color': '#00B050'},
            {'label': 'Strong', 'range': '50-75%', 'color': '#92D050'},
            {'label': 'Partial', 'range': '25-50%', 'color': '#FFFF00'},
            {'label': 'Insufficient', 'range': '0-25%', 'color': '#FF0000'}
        ]
    
    # Display the example sentence with highlighted rating
    st.markdown(
        f"##### The probability of your brand being seen or heard on {media_vehicle} is **{score:.0f}%** ("
        f"<span style='background-color: {rating_info['color']}; padding: 2px 8px; "
        f"border-radius: 3px; font-weight: bold; color: #000;'>{rating_info['rating']}</span>).",
        unsafe_allow_html=True
    )
    
#    st.markdown("Attentive Branding Rating:")
    st.markdown("<p style='font-size:10pt;'>Probability Brand Seen/Heard Rating:</p>",unsafe_allow_html=True)

    # Create columns for the rating boxes
    cols = st.columns([1, 1, 1, 1])
    
    for idx, r in enumerate(ranges):
        with cols[idx]:
            # Create a colored box using HTML
            box_html = f"""
            <div style="padding: 4px; background-color: {r['color']}; 
                        border: 2px solid #000; border-radius: 5px; text-align: center;
                        min-height: 50px; display: flex; flex-direction: column; 
                        justify-content: center;">
                <div style="font-weight: bold; font-size: 14px; color: #000;">{r['label']}</div>
                <div style="font-style: italic; font-size: 12px; color: #000; margin-top: 0px;line-height: 1.2;">{r['range']}</div>
            </div>
            """
            st.markdown(box_html, unsafe_allow_html=True)
    
   # Add spacing
    st.write("")

def calculate_short_form_caption_stats(audio_results, visual_results, watch_time_seconds):
    """
    Calculate statistics for Short Form caption.
    
    Returns dict with counts of each branding type within watch time.
    """
    visual_and_audio_count = 0
    audio_only_count = 0
    visual_only_count = 0
    
    for second in range(1, watch_time_seconds + 1):
        visual_detected = visual_results.get(second, {}).get('detected', 0) == 1
        audio_detected = audio_results.get(second, {}).get('detected', 0) == 1
        
        if visual_detected and audio_detected:
            visual_and_audio_count += 1
        elif audio_detected and not visual_detected:
            audio_only_count += 1
        elif visual_detected and not audio_detected:
            visual_only_count += 1
    
    total_branding_seconds = visual_and_audio_count + audio_only_count + visual_only_count
    
    return {
        'total_branding_seconds': total_branding_seconds,
        'visual_and_audio_count': visual_and_audio_count,
        'audio_only_count': audio_only_count,
        'visual_only_count': visual_only_count
    }


def calculate_long_form_simulation_caption_stats(simulation_result, duration):
    """
    Calculate statistics for Long Form simulation caption.
    
    Returns dict with counts and percentages for the simulation.
    """
    categories = simulation_result['categories']
    
    visual_and_audio_count = 0
    audio_only_count = 0
    visual_only_count = 0
    
    for second, category in categories.items():
        if category == "Visual and Audio Branding":
            visual_and_audio_count += 1
        elif category == "Audio Branding Only":
            audio_only_count += 1
        elif category == "Audio Branding Only (Non-Attentive)":
            audio_only_count += 1  # Count non-attentive audio branding too
        elif category == "Visual Branding Only":
            visual_only_count += 1
    
    total_branding_seconds = visual_and_audio_count + audio_only_count + visual_only_count
    
    # Calculate percentages
    branding_percentage = (total_branding_seconds / duration * 100) if duration > 0 else 0
    no_branding_percentage = 100 - branding_percentage
    
    return {
        'total_branding_seconds': total_branding_seconds,
        'visual_and_audio_count': visual_and_audio_count,
        'audio_only_count': audio_only_count,
        'visual_only_count': visual_only_count,
        'branding_percentage': branding_percentage,
        'no_branding_percentage': no_branding_percentage
    }


def format_caption_with_plurals(total, va_count, audio_count, visual_count):
    """
    Helper function to format caption text with correct singular/plural forms.
    
    Returns formatted string with proper "second" vs "seconds" usage.
    """
    # Determine plural forms
    total_word = "second" if total == 1 else "seconds"
    va_word = "second" if va_count == 1 else "seconds"
    audio_word = "second" if audio_count == 1 else "seconds"
    visual_word = "second" if visual_count == 1 else "seconds"
    
    return (f"On average, {total} {total_word} of branding seen or heard "
            f"({va_count} {va_word} of Visual and Audio Branding + "
            f"{audio_count} {audio_word} of Audio Branding Only + "
            f"{visual_count} {visual_word} of Visual Branding Only).")


def format_long_form_caption_with_plurals(sim_num, total, va_count, audio_count, 
                                          visual_count, branding_pct, no_branding_pct):
    """
    Helper function to format Long Form simulation caption with correct singular/plural forms.
    
    Returns formatted string with proper "second" vs "seconds" usage.
    """
    # Determine plural forms
    total_word = "second" if total == 1 else "seconds"
    va_word = "second" if va_count == 1 else "seconds"
    audio_word = "second" if audio_count == 1 else "seconds"
    visual_word = "second" if visual_count == 1 else "seconds"
    
    return (f"In Viewing Simulation #{sim_num}, {total} {total_word} of branding seen or heard "
            f"({va_count} {va_word} of Visual and Audio Branding + "
            f"{audio_count} {audio_word} of Audio Branding Only + "
            f"{visual_count} {visual_word} of Visual Branding Only). "
            f"Branding was seen or heard in {branding_pct:.0f}% of the viewing experience, "
            f"while the remaining {no_branding_pct:.0f}% of seconds had No Branding Present "
            f"or were non-attentive.")

def generate_ffmpeg_filter_complex_for_video_editing(non_attentive_seconds, duration, mute_audio=False):
    """
    Generate ffmpeg filter_complex string for blacking out non-attentive seconds
    and adding "Consumer not watching" text.
    
    Args:
        non_attentive_seconds: List of second numbers that should be blacked out
        duration: Total video duration in seconds
        mute_audio: If True, mute audio during non-attentive seconds
        
    Returns:
        tuple: (video_filter_string, audio_filter_string)
    """
    if not non_attentive_seconds:
        # No edits needed
        return None, None
    
    # Sort seconds to process in order
    non_attentive_seconds = sorted(non_attentive_seconds)
    
    # Build video filter for black overlays and text
    video_filters = []
    
    for second in non_attentive_seconds:
        # Convert second to time range (zero-based indexing)
        # Second 1 = 0:00.00 to 0:00.99
        start_time = second - 1  # Convert to 0-based
        end_time = second - 0.01  # End just before next second
        
        # Black overlay filter
        black_filter = f"drawbox=enable='between(t,{start_time},{end_time})':color=black:t=fill"
        video_filters.append(black_filter)
        
        # Text overlay filter - white text at bottom center
        text_filter = (f"drawtext=enable='between(t,{start_time},{end_time})':"
                      f"text='Consumer not watching':"
                      f"fontsize=48:"
                      f"fontcolor=white:"
                      f"x=(w-text_w)/2:"
                      f"y=(h-text_h)/2")
        video_filters.append(text_filter)
    
    # Combine all video filters
    video_filter_string = ",".join(video_filters) if video_filters else None
    
    # Build audio filter for muting if needed
    audio_filter_string = None
    if mute_audio and non_attentive_seconds:
        audio_filters = []
        for second in non_attentive_seconds:
            start_time = second - 1
            end_time = second - 0.01
            audio_filters.append(f"volume=enable='between(t,{start_time},{end_time})':volume=0")
        
        audio_filter_string = ",".join(audio_filters) if audio_filters else None
    
    return video_filter_string, audio_filter_string

def create_edited_video_short_form(video_path, attention_results, duration, unique_id):
   """
   Create edited video for Short Form/Social showing consumer attention.
   Blacks out screen and mutes audio during non-attentive seconds.
   """
   try:
       # Create output path with unique identifier - USE TEMP DIR FOR PUBLIC DEPLOYMENT
       output_dir = tempfile.gettempdir()
       output_filename = f"edited_video_short_form_{unique_id}.mp4"
       output_path = os.path.join(output_dir, output_filename)
       # Get audio and visual results from attention_results
       audio_results = attention_results.get('audio_results', {})
       visual_results = attention_results.get('visual_results', {})
       attentive_seconds = attention_results.get('attentive_seconds', 0)
       # Identify non-attentive seconds
       non_attentive_seconds = []
       for second in range(1, duration + 1):
           if second > attentive_seconds:
               non_attentive_seconds.append(second)
       # If no non-attentive seconds, just copy the original video
       if not non_attentive_seconds:
           import shutil
           shutil.copy2(video_path, output_path)
           return output_path
       # Generate filter strings
       video_filter, audio_filter = generate_ffmpeg_filter_complex_for_video_editing(
           non_attentive_seconds, duration, mute_audio=True
       )
       # Build ffmpeg command
       cmd = ['ffmpeg', '-i', video_path, '-y']
       # Add video filters
       if video_filter:
           cmd.extend(['-vf', video_filter])
       # Add audio filters
       if audio_filter:
           cmd.extend(['-af', audio_filter])
       # Output settings
       cmd.extend([
           '-c:v', 'libx264',
           '-preset', 'medium',
           '-crf', '23',
           '-c:a', 'aac',
           '-b:a', '128k',
           output_path
       ])
       # Run ffmpeg
       result = subprocess.run(cmd, capture_output=True, text=True, check=True)
       if os.path.exists(output_path):
           return output_path
       else:
           return None
   except subprocess.CalledProcessError as e:
       st.error(f"Video editing failed: {e.stderr}")
       return None
   except Exception as e:
       st.error(f"Video editing error: {e}")
       return None

def create_edited_videos_long_form(video_path, simulations, duration, unique_id):
   """
   Create 3 edited videos for Long Form showing different viewing simulations.
   Blacks out screen during non-attentive seconds but keeps audio on.
   """
   try:
       edited_videos = []
       for sim_num, simulation in enumerate(simulations, start=1):
           # Create output path with unique identifier - USE TEMP DIR FOR PUBLIC DEPLOYMENT
           output_dir = tempfile.gettempdir()
           output_filename = f"viewing_simulation_{sim_num}_{unique_id}.mp4"
           output_path = os.path.join(output_dir, output_filename)
           # Get categories for this simulation
           categories = simulation.get('categories', {})
           # Identify non-attentive seconds
           non_attentive_seconds = []
           for second, category in categories.items():
               if category == "Non-Attentive" or category == "Audio Branding Only (Non-Attentive)":
                   non_attentive_seconds.append(second)
           # If no non-attentive seconds, just copy the original video
           if not non_attentive_seconds:
               import shutil
               temp_output = output_path
               shutil.copy2(video_path, temp_output)
               edited_videos.append(temp_output)
               continue
           # Generate filter strings (no audio muting for long form)
           video_filter, _ = generate_ffmpeg_filter_complex_for_video_editing(
               non_attentive_seconds, duration, mute_audio=False
           )
           # Build ffmpeg command
           cmd = ['ffmpeg', '-i', video_path, '-y']
           # Add video filters
           if video_filter:
               cmd.extend(['-vf', video_filter])
           # Output settings
           cmd.extend([
               '-c:v', 'libx264',
               '-preset', 'medium',
               '-crf', '23',
               '-c:a', 'copy',
               output_path
           ])
           # Run ffmpeg
           result = subprocess.run(cmd, capture_output=True, text=True, check=True)
           if os.path.exists(output_path):
               edited_videos.append(output_path)
           else:
               st.error(f"Failed to create simulation {sim_num} video")
               return None
       return edited_videos if len(edited_videos) == 3 else None
   except subprocess.CalledProcessError as e:
       st.error(f"Video editing failed: {e.stderr}")
       return None
   except Exception as e:
       st.error(f"Video editing error: {e}")
       return None
       
#AI Recommendation Retry Logic
def call_gemini_with_retry(client, model, system_instruction, user_prompt, max_retries=4, base_delay=2):
    """
    Retry logic helpfer function 

    Call Gemini API with retry logic for handling overload errors.
    
    Args:
        client: Gemini client instance
        model: Model name (e.g., "gemini-2.5-flash")
        system_instruction: System instruction text
        user_prompt: User prompt text or content list
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        
    Returns:
        response.text or error message
    """
    # --- RETRY LOGIC ---
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                ),
                contents=user_prompt
            )
            # If successful, return immediately
            return response.text
            
        except Exception as e:
            # Check if the error is a 503 Service Unavailable (Overloaded)
            if "503" in str(e) or "overloaded" in str(e).lower():
                if attempt < max_retries - 1:
                    # Exponential backoff: Wait 2s, then 4s, then 8s...
                    sleep_time = base_delay * (2 ** attempt)
                    time.sleep(sleep_time)
                    continue # Try the loop again
                else:
                    return "Google AI is currently overloaded. Please try again in a few minutes."
            else:
                # If it's a completely different error (e.g. Invalid API Key), fail immediately
                raise e
    # --- RETRY LOGIC ---

    return "Failed to get response after maximum retries."

#AI Recommendation API Calls (Short Form/Social = 4 calls, Long Form = 3 calls)
def extract_features_from_video(video_file_obj, analysis_id, timestamp, google_api_key):
   """
   SHORT FORM CALL #1: Extract features from video using AI.
   Args:
       video_file_obj: Gemini uploaded file object
       analysis_id: Unique analysis identifier
       timestamp: Timestamp of analysis
       google_api_key: Google API key
   Returns:
       feature_output_response: List of dicts with feature_id, feature_name, response
   """
   try:
       # Load features table
       features_df = load_features_table()
       if features_df is None:
           return None
       # Prepare feature list
       feature_list = []
       for _, row in features_df.iterrows():
           feature_list.append({
               'feature_id': str(row['feature_id']),
               'feature_name': row['feature_name'],
               'description': row['description']
           })
       # Prepare combined prompt
       combined_prompts = "\n".join([
           f"{feature['feature_id']} - {feature['feature_name']}: {feature['description']}"
           for feature in feature_list
       ])
       # Construct query
       query_text = f"""Analyze the following features and provide a concise description for each feature without unnecessary words or restating the question.
Please provide the output in the format 'FeatureID - FeatureName: Response':
{combined_prompts}"""
       # System instruction
       system_instruction = "You are a helpful assistant that likes to help people. Please provide answers without unnecessary phrases or restating the question."
       # Call Gemini with retry logic - pass file object directly
       client = genai.Client(api_key=google_api_key)
       response_text = call_gemini_with_retry(
           client=client,
           model="gemini-2.5-flash",
           system_instruction=system_instruction,
           user_prompt=[query_text, video_file_obj]  # ← FIXED: Pass file object directly in list
       )
       # Parse response using regex
       import re
       pattern = r"(\d+)\s*-\s*([^:]+):\s*(.+?)(?=\n\d+\s*-|\Z)"
       matches = re.findall(pattern, response_text, re.DOTALL)
       # Create feature_output_response
       feature_output_response = []
       for match in matches:
           feature_id = match[0].strip()
           feature_name = match[1].strip()
           response = match[2].strip()
           feature_output_response.append({
               'feature_id': feature_id,
               'feature_name': feature_name,
               'response': response
           })
       # Log to Google Sheets
       log_feature_extraction(
           FEATURE_EXTRACTION_SHEET_ID,
           analysis_id,
           timestamp,
           feature_output_response
       )
       return feature_output_response
   except Exception as e:
       st.error(f"Feature extraction failed: {e}")
       return None

def compare_features_to_criteria(feature_output_response, media_vehicle, analysis_id, timestamp, google_api_key):
    """
    SHORT FORM CALL #2: Compare feature extraction to creative best practices criteria.
    
    Args:
        feature_output_response: Output from call #1
        media_vehicle: Selected media vehicle
        analysis_id: Unique analysis identifier
        timestamp: Timestamp of analysis
        google_api_key: Google API key
        
    Returns:
        criteria_output_response: List of dicts with criteria_id, criteria_name, response
    """
    try:
        # Load creative best practices and filter by media vehicle
        practices_df = load_creative_best_practices_table()
        if practices_df is None:
            return None
        
        criteria_list = get_criteria_for_media_vehicle(media_vehicle, practices_df)
        if not criteria_list:
            st.warning(f"No criteria found for {media_vehicle}")
            return None
        
        # Format feature extraction results
        feature_text = "\n".join([
            f"{f['feature_id']} - {f['feature_name']}: {f['response']}"
            for f in feature_output_response
        ])
        
        # Format criteria
        criteria_text = "\n".join([
            f"{c['criteria_id']} - {c['criteria_name']}: {c['description']}"
            for c in criteria_list
        ])
        
        # Construct query
        query_text = f"""Here are the extracted features from the video:

{feature_text}

Now compare these features against the following creative best practice criteria for {media_vehicle}:

{criteria_text}

Determine if each criterion has been met. Provide specific examples and descriptions for each criterion that explains why it has or has not been met.

Please provide the output in the format 'CriteriaID - CriteriaName: Response':"""
        
        # System instruction
        system_instruction = "You are a helpful assistant that likes to help people. Please provide answers without unnecessary phrases or restating the question."
        
        # Call Gemini with retry logic
        client = genai.Client(api_key=google_api_key)
        response_text = call_gemini_with_retry(
            client=client,
            model="gemini-2.5-flash",
            system_instruction=system_instruction,
            user_prompt=query_text
        )
        
        # Parse response using regex
        import re
        pattern = r"(\d+)\s*-\s*([^:]+):\s*(.+?)(?=\n\d+\s*-|\Z)"
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        # Create criteria_output_response
        criteria_output_response = []
        for match in matches:
            criteria_id = match[0].strip()
            criteria_name = match[1].strip()
            response = match[2].strip()
            
            criteria_output_response.append({
                'criteria_id': criteria_id,
                'criteria_name': criteria_name,
                'response': response
            })
        
        # Log to Google Sheets
        log_criteria_extraction(
            CRITERIA_EXTRACTION_SHEET_ID,
            analysis_id,
            timestamp,
            criteria_output_response
        )
        
        return criteria_output_response
        
    except Exception as e:
        st.error(f"Criteria comparison failed: {e}")
        return None

def summarize_criteria_and_suggest_improvements(criteria_output_response, media_vehicle, analysis_id, timestamp, google_api_key):
    """
    SHORT FORM CALL #3 / LONG FORM CALL #2: Summarize criteria output and provide improvement suggestions.
    
    Args:
        criteria_output_response: Output from criteria assessment
        media_vehicle: Selected media vehicle
        analysis_id: Unique analysis identifier
        timestamp: Timestamp of analysis
        google_api_key: Google API key
        
    Returns:
        creative_best_practice_summary: Summary text from AI
    """
    try:
        # Format criteria responses
        criteria_text = "\n".join([
            f"{c['criteria_id']} - {c['criteria_name']}: {c['response']}"
            for c in criteria_output_response
        ])
        
        # Construct query
        query_text = f"""Please analyze the following criteria assessment results:

{criteria_text}

Summarize whether the creative aligns with creative best practices on {media_vehicle}. Also provide suggestions for improvement to the creative that will make it better suited for the media vehicle based on what the criteria is.

Structure your response as:
Summary: [Your summary here]
Suggestions: [Your suggestions here]"""
        
        # System instruction
        system_instruction = "You are a helpful assistant that likes to help people. Please provide answers without unnecessary phrases or restating the question."
        
        # Call Gemini with retry logic
        client = genai.Client(api_key=google_api_key)
        creative_best_practice_summary = call_gemini_with_retry(
            client=client,
            model="gemini-2.5-flash",
            system_instruction=system_instruction,
            user_prompt=query_text
        )
        
        # Log to Google Sheets
        log_creative_best_practice_summary(
            SUMMARY_BEST_PRACTICE_SHEET_ID,
            analysis_id,
            timestamp,
            creative_best_practice_summary
        )
        
        return creative_best_practice_summary
        
    except Exception as e:
        st.error(f"Criteria summary failed: {e}")
        return None

def assess_criteria_from_video_longform(video_file_obj, media_vehicle, analysis_id, timestamp, google_api_key):
   """
   LONG FORM CALL #1: Assess criteria directly from video (no feature extraction step).
   Args:
       video_file_obj: Gemini uploaded file object
       media_vehicle: Selected media vehicle
       analysis_id: Unique analysis identifier
       timestamp: Timestamp of analysis
       google_api_key: Google API key
   Returns:
       criteria_output_response: List of dicts with criteria_id, criteria_name, response
   """
   try:
       # Load creative best practices and filter by media vehicle
       practices_df = load_creative_best_practices_table()
       if practices_df is None:
           return None
       criteria_list = get_criteria_for_media_vehicle(media_vehicle, practices_df)
       if not criteria_list:
           st.warning(f"No criteria found for {media_vehicle}")
           return None
       # Format criteria
       combined_prompts = "\n".join([
           f"{c['criteria_id']} - {c['criteria_name']}: {c['description']}"
           for c in criteria_list
       ])
       # Construct query
       query_text = f"""Analyze the video against the following creative best practice criteria for {media_vehicle}:
{combined_prompts}
Provide a concise description for each criterion without unnecessary words or restating the question. Provide specific examples or descriptions for each criterion that explains why it has or has not been met.
Please provide the output in the format 'CriteriaID - CriteriaName: Response':"""
       # System instruction
       system_instruction = "You are a helpful assistant that likes to help people. Please provide answers without unnecessary phrases or restating the question."
       # Call Gemini with retry logic - pass file object directly
       client = genai.Client(api_key=google_api_key)
       response_text = call_gemini_with_retry(
           client=client,
           model="gemini-2.5-flash",
           system_instruction=system_instruction,
           user_prompt=[query_text, video_file_obj]  # ← FIXED: Pass file object directly in list
       )
       # Parse response using regex
       import re
       pattern = r"(\d+)\s*-\s*([^:]+):\s*(.+?)(?=\n\d+\s*-|\Z)"
       matches = re.findall(pattern, response_text, re.DOTALL)
       # Create criteria_output_response
       criteria_output_response = []
       for match in matches:
           criteria_id = match[0].strip()
           criteria_name = match[1].strip()
           response = match[2].strip()
           criteria_output_response.append({
               'criteria_id': criteria_id,
               'criteria_name': criteria_name,
               'response': response
           })
       # Log to Google Sheets (same sheet as Short Form)
       log_criteria_extraction(
           CRITERIA_EXTRACTION_SHEET_ID,
           analysis_id,
           timestamp,
           criteria_output_response
       )
       return criteria_output_response
   except Exception as e:
       st.error(f"Long form criteria assessment failed: {e}")
       return None
    
def get_ai_recommendation(analysis_summary, brand_name, media_vehicle, google_api_key, 
                          criteria_output_response=None, creative_best_practice_summary=None,
                          analysis_id=None, timestamp=None):
    """
    SHORT FORM CALL #4 / LONG FORM CALL #3: Get AI recommendation using enhanced context.
    
    This is the final call that combines all previous analysis results.
    """
    try:
        # Configure the API key
        client = genai.Client(api_key=google_api_key)

        sys_instruct = """You are an expert media strategist and advertising analyst.
        Your job is to provide recommendations about whether a video advertisement should be placed on a specific media vehicle based on branding and attention analysis results as well as results of creative best practices comparisons. Avoid strong language like "unacceptable" or "extremely" in your response.
        
        """

        user_prompt = f"""
        DATA TO ANALYZE:
        Brand: {brand_name}
        Intended Media Vehicle: {media_vehicle}

        Branding and Attention Analysis Results:
        {analysis_summary}
        """
        
        # Add criteria output if available
        if criteria_output_response:
            criteria_text = "\n".join([
                f"{c['criteria_id']} - {c['criteria_name']}: {c['response']}"
                for c in criteria_output_response
            ])
            user_prompt += f"""
        
        Creative Best Practice Criteria Assessment:
        {criteria_text}
        """
        
        # Add summary if available
        if creative_best_practice_summary:
            user_prompt += f"""
        
        Creative Best Practice Summary and Suggestions:
        {creative_best_practice_summary}
        """
        
        user_prompt += f"""

        Based on this analysis, should this video be placed on {media_vehicle}?
        
        Format your response in this exact structures with each section on a new line, each bullet point on its own line, and a line break between sections :
        **Recommendation:** [Recommended / Not Recommended]
        
        **Key Reasoning:**
        Explain the key reasons in 2-3 sentences using specific data from the analysis (branding %, attention metrics, criteria assessment).
        
        **What's Working:**
        [Strength 1 - be specific]
        [Strength 2 - be specific]
        [Strength 3 - be specific]
        
        **What Can Improve:**
        [Concrete, actionable change 1]
        [Concrete, actionable change 2]
        [Concrete, actionable change 3] 
        
        **Bottom Line:**
        One sentence - should this run as-is, or does it need changes first?
        """
        
        # --- RETRY LOGIC ---
        ai_recommendation = call_gemini_with_retry(
            client=client,
            model="gemini-2.5-flash",
            system_instruction=sys_instruct,
            user_prompt=user_prompt
        )
        
        # Log to Google Sheets if analysis_id and timestamp provided
        if analysis_id and timestamp:
            log_ai_recommendation(
                AI_RECOMMENDATION_SHEET_ID,
                analysis_id,
                timestamp,
                ai_recommendation
            )
        
        return ai_recommendation
        
    except Exception as e:
        return f"Error generating AI recommendation: {e}. Please check your Google API key and try again."

def log_analysis_results(results, video_file_name, brand_name, brand_display_name, media_vehicle,
                       final_categories, audio_results, visual_results):
   """
   Log analysis results to Google Sheets for tracking and benchmarking.
   Creates two Google Sheets:
   1. Second-by-second log (detailed second-by-second branding analysis)
   2. Summary log (high-level summary metrics for each analysis)
   """

   try:
       # --- Google Sheet IDs ---
       SECOND_LOG_SHEET_ID = "1h7U-l_GQGFejELIHzhsR2nZzoa0LzwpbUiUEFE9IWgk"
       SUMMARY_LOG_SHEET_ID = "1tjWzAmAdkkKbrtdg0vifvv-ikCU4wLWF8AeAEK-lqWg"
       
       # Generate unique analysis ID
       unique_code = str(uuid.uuid4())[:8]
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       analysis_id = f"{timestamp}_{unique_code}"
       
       # Extract key result info
       duration = results['duration']
       category_counts = results['category_counts']
       branding_percentage = results['branding_percentage']
       attention_results = results.get('attention_results')
       
       # Determine media form
       media_form = attention_results['media_form'] if attention_results else "Unknown"
       
       # --- FILE 1: Second-by-Second Log ---
       second_data = []
       for second in range(1, duration + 1):
           branding_category = final_categories.get(second, "Unknown")
           second_data.append([
               analysis_id,
               timestamp,
               video_file_name,
               brand_name,
               media_vehicle,
               media_form,
               duration,
               second,
               branding_category,
               brand_display_name
           ])
       
       # --- FILE 2: Summary Log ---
       visual_and_audio_pct = (category_counts['Visual and Audio Branding'] / duration * 100) if duration > 0 else 0
       visual_only_pct = (category_counts['Visual Branding Only'] / duration * 100) if duration > 0 else 0
       audio_only_pct = (category_counts['Audio Branding Only'] / duration * 100) if duration > 0 else 0
       no_branding_pct = (category_counts['No Branding Present'] / duration * 100) if duration > 0 else 0
       
       attentive_branding_score = None
       if attention_results:
           attentive_branding_score = attention_results.get('attentive_branding_score')
       
       summary_data = [[
           analysis_id,
           timestamp,
           video_file_name,
           brand_name,
           media_vehicle,
           media_form,
           duration,
           round(visual_and_audio_pct, 2),
           round(visual_only_pct, 2),
           round(audio_only_pct, 2),
           round(no_branding_pct, 2),
           round(branding_percentage, 2),
           round(attentive_branding_score, 2) if attentive_branding_score is not None else None,
           brand_display_name
       ]]
       
       # --- Upload to Google Sheets (DATA ONLY, NO HEADERS) ---
       append_to_gsheet(SECOND_LOG_SHEET_ID, second_data)
       append_to_gsheet(SUMMARY_LOG_SHEET_ID, summary_data)
       
       return analysis_id, timestamp, None
   
   except Exception as e:
       return None, None, f"Logging failed: {str(e)}"

def log_feature_extraction(sheet_id, analysis_id, timestamp, feature_responses):
   """
   Log feature extraction results to Google Sheets.
   Args:
       sheet_id: Google Sheet ID for social_feature_extraction_log
       analysis_id: Unique analysis identifier
       timestamp: Timestamp of analysis
       feature_responses: List of dicts with feature_id, feature_name, response
   """
   try:
       rows = []
       for feature in feature_responses:
           rows.append([
               analysis_id,
               timestamp,
               feature['feature_id'],
               feature['feature_name'],
               feature['response']
           ])
       append_to_gsheet(sheet_id, rows)
       st.success(f"✓ Feature extraction logged: {len(rows)} features")
   except Exception as e:
       st.error(f"Feature extraction logging failed: {str(e)}")
       import traceback
       st.error(f"Full error: {traceback.format_exc()}")

def log_criteria_extraction(sheet_id, analysis_id, timestamp, criteria_responses):
   """
   Log criteria extraction results to Google Sheets.
   Args:
       sheet_id: Google Sheet ID for criteria_extraction_log
       analysis_id: Unique analysis identifier
       timestamp: Timestamp of analysis
       criteria_responses: List of dicts with criteria_id, criteria_name, response
   """
   try:
       rows = []
       for criterion in criteria_responses:
           rows.append([
               analysis_id,
               timestamp,
               criterion['criteria_id'],
               criterion['criteria_name'],
               criterion['response']
           ])
       append_to_gsheet(sheet_id, rows)
       st.success(f"✓ Criteria extraction logged: {len(rows)} criteria")
   except Exception as e:
       st.error(f"Criteria extraction logging failed: {str(e)}")
       import traceback
       st.error(f"Full error: {traceback.format_exc()}")

def log_creative_best_practice_summary(sheet_id, analysis_id, timestamp, summary_text):
   """
   Log creative best practice summary to Google Sheets.
   Args:
       sheet_id: Google Sheet ID for summary_creative_best_practice_log
       analysis_id: Unique analysis identifier
       timestamp: Timestamp of analysis
       summary_text: Summary text from AI
   """
   try:
       rows = [[analysis_id, timestamp, summary_text]]
       append_to_gsheet(sheet_id, rows)
       st.success(f"✓ Summary logged")
   except Exception as e:
       st.error(f"Summary logging failed: {str(e)}")
       import traceback
       st.error(f"Full error: {traceback.format_exc()}")

def log_ai_recommendation(sheet_id, analysis_id, timestamp, recommendation_text):
   """
   Log AI recommendation to Google Sheets.
   Args:
       sheet_id: Google Sheet ID for ai_recommendation_log
       analysis_id: Unique analysis identifier
       timestamp: Timestamp of analysis
       recommendation_text: AI recommendation text
   """
   try:
       rows = [[analysis_id, timestamp, recommendation_text]]
       append_to_gsheet(sheet_id, rows)
       st.success(f"✓ AI recommendation logged")
   except Exception as e:
       st.error(f"AI recommendation logging failed: {str(e)}")
       import traceback
       st.error(f"Full error: {traceback.format_exc()}")

def cleanup_temp_files(audio_file, frames_dir):
    try:
        if os.path.exists(audio_file):
            os.remove(audio_file)
        if os.path.exists(frames_dir):
            safe_remove_directory(frames_dir)
    except Exception as e:
        st.warning(f"Cleanup warning: {e}")

# Main Process Function
def process_video_analysis(video_file, brand_name, brand_display_name, media_vehicle, google_api_key, attention_df):
    """Main processing function"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name
        
        status_text.text("Analyzing video duration...")
        progress_bar.progress(10)
        
        duration = get_video_duration(video_path)
        if duration is None:
            return None, "Failed to get video duration"
        
        # Prepare temporary files
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        temp_frames_dir = tempfile.mkdtemp()
        
        # Audio Analysis
        status_text.text("Extracting and analyzing audio...")
        progress_bar.progress(25)
        
        audio_path = extract_audio(video_path, temp_audio_file)
        if audio_path is None:
            return None, "Audio extraction failed"
        
        transcription_result = transcribe_with_whisper(audio_path, "small")  # Using small model for speed
        if transcription_result is None:
            return None, "Audio transcription failed"
        
        brand_variations = BRAND_VARIATIONS.get(brand_name, [brand_name.lower()])
        audio_results = analyze_audio_branding(transcription_result, brand_name, brand_variations, duration)
        
        # Visual Analysis
        status_text.text("Extracting and analyzing frames...")
        progress_bar.progress(50)
        
        frames_dir, frame_count = extract_frames(video_path, temp_frames_dir)
        if frames_dir is None:
            return None, "Frame extraction failed"
        
        status_text.text("Running visual brand detection...")
        progress_bar.progress(70)
        
        visual_results = analyze_visual_branding(
            frames_dir, brand_name, duration, PREDICTION_KEY, ENDPOINT, PROJECT_ID, MODEL_NAME
        )
        
        # Categorize results
        status_text.text("Categorizing branding results...")
        progress_bar.progress(85)
        
        final_categories = categorize_branding(audio_results, visual_results, duration)
        
        # Calculate summary stats
        category_counts = {cat: 0 for cat in CATEGORY_COLORS}
        for category in final_categories.values():
            category_counts[category] += 1
        
        total_branding_seconds = (category_counts["Visual and Audio Branding"] + 
                                category_counts["Audio Branding Only"] + 
                                category_counts["Visual Branding Only"])
        
        branding_percentage = (total_branding_seconds / duration) * 100 if duration > 0 else 0
        
        attention_results = None
        attention_viz = None

        if attention_df is not None:
            attention_results = perform_attention_analysis(
                final_categories, audio_results, visual_results, 
                duration, media_vehicle, attention_df, video_path
            )
            
            if attention_results and attention_results['media_form'] == 'Short Form or Social':
                attention_viz = create_attention_visualization(attention_results)
        
        # Total Branding Coverage
        total_branding_coverage = f"""
{round(branding_percentage)}% of seconds in video contain branding
"""
        
        # Create base analysis summary
        analysis_summary = f"""
Video Duration: {duration} seconds
Brand: {brand_name}

Overall Branding Presence Analysis:
- Visual and Audio Branding: {category_counts['Visual and Audio Branding']} seconds ({round((category_counts['Visual and Audio Branding']/duration)*100)}%)
- Audio Branding Only: {category_counts['Audio Branding Only']} seconds ({round((category_counts['Audio Branding Only']/duration)*100)}%)  
- Visual Branding Only: {category_counts['Visual Branding Only']} seconds ({round((category_counts['Visual Branding Only']/duration)*100)}%)
- No Branding Present: {category_counts['No Branding Present']} seconds ({round((category_counts['No Branding Present']/duration)*100)}%)

Percentage of Branding Presence: {round(branding_percentage)}% of video duration
"""
        
        # Add attention analysis to summary if available
        if attention_results:
            analysis_summary += f"""

        - Attention Norms Based Branding Analysis ({attention_results['media_form']}):
        - Attentive Seconds Analyzed: {attention_results['attentive_seconds']} seconds
        - Probablity brand seen/heard given what we know about attention norms on platform: {attention_results['attentive_branding_score']:.0f}%"""
        
        # Create visualizations
        timeline_fig = create_timeline_visualization(final_categories, duration)
        summary_fig = create_summary_visualization(final_categories, duration)
        
        # Log results to Google Sheets
        analysis_id, timestamp_str, log_error = log_analysis_results(
            {
                'duration': duration,
                'category_counts': category_counts,
                'branding_percentage': branding_percentage,
                'attention_results': attention_results
            },
            video_file.name,
            brand_name,
            brand_display_name,
            media_vehicle,
            final_categories,
            audio_results,
            visual_results
        )
        
        if log_error:
            st.warning(f"Results were not logged: {log_error}")
        else:
            st.success(f"Analysis logged with ID: {analysis_id}")
        
        # ============================================================================
        # ENHANCED AI RECOMMENDATION WITH MEDIA FORM LOGIC
        # ============================================================================
        
        # Upload video to Gemini File API for AI analysis
        status_text.text("Uploading video for AI analysis...")
        progress_bar.progress(88)
        
        gemini_video_file = upload_video_to_gemini(video_path, google_api_key)
        if gemini_video_file is None:
            st.warning("Video upload to Gemini failed. AI recommendation will use basic analysis only.")
            criteria_output_response = None
            creative_best_practice_summary = None
        else:
            # Determine media form from attention results
            if attention_results:
                media_form = attention_results['media_form']
            else:
                # Fallback: determine from attention_df
                attention_metrics = get_attention_metrics(media_vehicle, attention_df)
                media_form = attention_metrics['media_form'] if attention_metrics else "Unknown"
            
            # ============================================================================
            # SHORT FORM OR SOCIAL FLOW (4 API CALLS)
            # ============================================================================
            if media_form == "Short Form or Social":
                status_text.text("Extracting features from video (1/4)...")
                progress_bar.progress(90)
                
                # Call #1: Feature Extraction
                feature_output_response = extract_features_from_video(
                    gemini_video_file, analysis_id, timestamp_str, google_api_key
                )
                
                if feature_output_response:
                    status_text.text("Comparing features to criteria (2/4)...")
                    progress_bar.progress(92)
                    
                    # Call #2: Compare Features to Criteria
                    criteria_output_response = compare_features_to_criteria(
                        feature_output_response, media_vehicle, analysis_id, timestamp_str, google_api_key
                    )
                    
                    if criteria_output_response:
                        status_text.text("Summarizing criteria assessment (3/4)...")
                        progress_bar.progress(94)
                        
                        # Call #3: Summarize Criteria
                        creative_best_practice_summary = summarize_criteria_and_suggest_improvements(
                            criteria_output_response, media_vehicle, analysis_id, timestamp_str, google_api_key
                        )
                    else:
                        creative_best_practice_summary = None
                else:
                    criteria_output_response = None
                    creative_best_practice_summary = None
            
            # ============================================================================
            # LONG FORM FLOW (3 API CALLS)
            # ============================================================================
            elif media_form == "Long Form":
                status_text.text("Assessing criteria from video (1/3)...")
                progress_bar.progress(90)
                
                # Call #1: Direct Criteria Assessment
                criteria_output_response = assess_criteria_from_video_longform(
                    gemini_video_file, media_vehicle, analysis_id, timestamp_str, google_api_key
                )
                
                if criteria_output_response:
                    status_text.text("Summarizing criteria assessment (2/3)...")
                    progress_bar.progress(93)
                    
                    # Call #2: Summarize Criteria (same function as Short Form #3)
                    creative_best_practice_summary = summarize_criteria_and_suggest_improvements(
                        criteria_output_response, media_vehicle, analysis_id, timestamp_str, google_api_key
                    )
                else:
                    creative_best_practice_summary = None
            
            else:
                # Unknown media form - skip enhanced AI
                criteria_output_response = None
                creative_best_practice_summary = None
        
        # Get AI recommendation (final call for both flows)
        status_text.text("Generating final AI recommendation...")
        progress_bar.progress(96)
        
        # Call #4 (Short Form) / Call #3 (Long Form): Final Recommendation
        ai_recommendation = get_ai_recommendation(
            analysis_summary, 
            brand_name, 
            media_vehicle, 
            google_api_key,
            criteria_output_response=criteria_output_response,
            creative_best_practice_summary=creative_best_practice_summary,
            analysis_id=analysis_id,
            timestamp=timestamp_str
        )

        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Cleanup
        cleanup_temp_files(temp_audio_file, frames_dir)
        os.unlink(video_path)
        
        return {
            'analysis_summary': analysis_summary,
            'total_branding_coverage': total_branding_coverage,
            'ai_recommendation': ai_recommendation,
            'timeline_fig': timeline_fig,
            'summary_fig': summary_fig,
            'category_counts': category_counts,
            'duration': duration,
            'branding_percentage': branding_percentage,
            'attention_results': attention_results,
            'attention_viz': attention_viz,
            'media_vehicle': media_vehicle
        }, None
        
    except Exception as e:
        return None, f"Analysis failed: {str(e)}"

# Streamlit UI
def main():
    st.title("🎬 AI Branding Analysis Chatbot")

    # --- Initialize a reset counter ---
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0
        
    # Check FFmpeg availability
    if not check_ffmpeg():
        st.error("FFmpeg not found! Please install FFmpeg to process videos.")
        st.stop()
    
    # Validate all required secrets
    required_secrets = {
        "GOOGLE_API_KEY": "Google Gemini API Key",
        "AZURE_PREDICTION_KEY": "Azure Prediction Key",
        "AZURE_ENDPOINT": "Azure Endpoint",
        "AZURE_PROJECT_ID": "Azure Project ID",
        "AZURE_MODEL_NAME": "Azure Model Name"
    }
    missing_secrets = []
    for secret_key, secret_name in required_secrets.items():
        try:
            if not st.secrets.get(secret_key):
                missing_secrets.append(secret_name)
        except:
            missing_secrets.append(secret_name)
    
    # Check for GCP service account
    try:
        if "gcp_service_account" not in st.secrets:
            missing_secrets.append("GCP Service Account")
    except:
        missing_secrets.append("GCP Service Account")
    if missing_secrets:
        st.error(f"⚠️ Missing required secrets: {', '.join(missing_secrets)}")
        st.info("Please configure all required secrets in Streamlit Cloud Settings → Secrets")
        st.stop()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "results" in message:
                # Display previous results
                results = message["results"]
                
                st.markdown("<h3 style='text-decoration: underline;'>Branding Summary</h3>", unsafe_allow_html=True)
                
                st.markdown("#### Branding Timeline:")
                st.pyplot(results['timeline_fig'])

                st.markdown("#### Branding Distribution:")
                st.metric("Percentage of Branding Presence", 
                            f"{results['branding_percentage']:.0f}%")
                st.pyplot(results['summary_fig'])

                st.divider()
                
                if results.get('attention_results'):
                    st.markdown("<h3 style='text-decoration: underline;'>Branding Attention Analysis: Average Consumer Viewing Experience</h3>",unsafe_allow_html=True)   

                    attn = results['attention_results']

                    media_vehicle = results.get('media_vehicle', 'Unknown Vehicle')
                    
                    # Display the attentive branding rating
                    display_attentive_branding_rating(
                        attn['attentive_branding_score'], 
                        attn['media_form'],
                        media_vehicle
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Probability Brand Seen/Heard", 
                                f"{attn['attentive_branding_score']:.0f}%")
                        
                    with col2:
                        st.metric(f"Average Attentive Seconds on {media_vehicle}",
                                f"{attn['attentive_seconds']} seconds")
                        
                        # Show watch time for Short Form
#                        if attn['media_form'] == 'Short Form or Social' and 'watch_time_seconds' in attn:
#                            st.metric("Average Watch Time", 
#                                    f"{attn['watch_time_seconds']} seconds")
                        
                        # Show branding percentage for Long Form
#                        if attn['media_form'] == 'Long Form' and 'branding_percentage' in attn:
#                            st.metric("Overall Branding %", 
#                                    f"{attn['branding_percentage']:.0f}%")
                    
                    # Display timeline visualization for Short Form
                    if 'timeline_viz' in attn:
                        st.subheader("Attentive Branding")
                        st.pyplot(attn['timeline_viz'])
                        
                        # Add caption for Short Form
                        #caption_stats = calculate_short_form_caption_stats(
                            #message['results']['attention_results']['audio_results'],
                            #message['results']['attention_results']['visual_results'],
                            #attn['watch_time_seconds']
                        #)
                        #caption_text = format_caption_with_plurals(
                            #caption_stats['total_branding_seconds'],
                            #caption_stats['visual_and_audio_count'],
                            #caption_stats['audio_only_count'],
                            #caption_stats['visual_only_count']
                        #)
                        #st.markdown(caption_text)
                    
                    # Display simulation timelines for Long Form
                    if 'stacked_simulation_fig' in attn:
                        st.markdown("#### Viewing Experience Simulations:")
                        st.pyplot(attn['stacked_simulation_fig'])

                    # Display edited videos for Long Form
                    if 'edited_videos' in attn and len(attn['edited_videos']) == 3:
                        st.markdown("#### Edited Videos: Viewing Simulations")
                            
                        col1, col2, col3 = st.columns(3)
                            
                        with col1:
                            st.markdown("**Viewing Simulation #1**")
                            st.video(attn['edited_videos'][0])
                            
                        with col2:
                            st.markdown("**Viewing Simulation #2**")
                            st.video(attn['edited_videos'][1])
                            
                        with col3:
                            st.markdown("**Viewing Simulation #3**")
                            st.video(attn['edited_videos'][2])
                    
                            # Add caption for each simulation
                            #simulation = attn['simulations'][i-1]
                            #caption_stats = calculate_long_form_simulation_caption_stats(
                                #simulation,
                                #message['results']['duration']
                            #)
                            #caption_text = format_long_form_caption_with_plurals(
                                #i,
                                #caption_stats['total_branding_seconds'],
                                #caption_stats['visual_and_audio_count'],
                                #caption_stats['audio_only_count'],
                                #caption_stats['visual_only_count'],
                                #caption_stats['branding_percentage'],
                                #caption_stats['no_branding_percentage']
                            #)
#                            st.caption(caption_text)
                            #a, b, c = st.columns(3)
                            #with a:
                                #st.metric("Visual and Audio Branding", 
                                        #f"{caption_stats['visual_and_audio_count']} sec")
                                
                            #with b:
                                #st.metric("Visual Branding Seen", 
                                        #f"{caption_stats['visual_only_count']} sec")
                                
                           # with c:
                                #st.metric("Audio Branding Heard", 
                                        #f"{caption_stats['audio_only_count']} sec")
                    
                    # Keep the original distribution chart if it exists
                    if results.get('attention_viz'):
                        st.pyplot(results['attention_viz'])

                    # Display edited video for Short Form
                    if attn['media_form'] == 'Short Form or Social' and 'edited_video' in attn:
                        st.subheader("Edited Video: Average Consumer Experience")
                        col1, col2 = st.columns([1, 5])
                        with col1:
                            st.video(attn['edited_video'])

                st.divider()

                st.markdown(f"<h3 style='text-decoration: underline;'>Is the creative best suited for {media_vehicle}?</h3>", unsafe_allow_html=True)
                st.caption("The following recommendations were generated with help of AI. The insights are for informational purposes only and should be reviewed with human judgment.")
                st.markdown("<p style='font-size:10pt;'><i>This section uses AI to provide recommendations about whether a creative should be placed on the intended media vehicle specified as input. " \
                "The recommendation is based on the branding and attention analysis results.</i></p>", unsafe_allow_html=True)
                st.markdown(results['ai_recommendation'])
        
    # Input section
    st.subheader("Provide Your Inputs")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload a video file", 
            type=['mp4'], 
            key=f"video_uploader_{st.session_state.input_key}" 
        )
    
    with col2:
        selected_display_brand = st.selectbox(
            "Select Brand", 
            list(BRAND_DISPLAY_MAP.keys()),
            key=f"brand_select_{st.session_state.input_key}"
        )
        
        selected_brand = BRAND_DISPLAY_MAP[selected_display_brand]
    
    with col3:
        selected_media_vehicle = st.selectbox(
            "Select Intended Media Vehicle", 
            MEDIA_VEHICLES,
            key=f"vehicle_select_{st.session_state.input_key}"
        )
    
    # Process button
    if st.button("Analyze Video", type="primary"):
        if not uploaded_file:
            st.error("Please upload a video file")
        elif not selected_brand:
            st.error("Please select a brand")
        elif not selected_media_vehicle:
            st.error("Please select a media vehicle")
        else:
            # Add user message
            user_message = f"Analyzing video: {uploaded_file.name}\nBrand: {selected_brand}\nMedia Vehicle: {selected_media_vehicle}"
            st.session_state.messages.append({
                "role": "user",
                "content": user_message
            })
            
            with st.chat_message("user"):
                st.write(user_message)
            
            # Process the video using hardcoded API key
            with st.chat_message("assistant"):
                st.write("Processing your video... This may take a few minutes depending on video length.")
                
                results, error = process_video_analysis(
                    uploaded_file, selected_brand, selected_display_brand, selected_media_vehicle, GOOGLE_API_KEY, attention_df
                )
                
                if error:
                    st.error(f"Analysis failed: {error}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Analysis failed: {error}"
                    })
                else:
                    # Display results
                    results['media_vehicle'] = selected_media_vehicle
                    
                    # Add assistant message with results
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "**Analysis Complete!**\n\nI've analyzed your video and generated recommendations. Check the detailed results below!",
                        "results": results
                    })

                    st.session_state.input_key += 1
            
            st.rerun()

if __name__ == "__main__":
    main()
