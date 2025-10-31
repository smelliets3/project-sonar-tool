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

# LangChain imports for AI recommendations with Gemini
#try:
#    from langchain_google_genai import ChatGoogleGenerativeAI
#    from langchain.schema import HumanMessage, SystemMessage
#    print("✓ LangChain Google GenAI libraries found")
#except ImportError:
#    st.error("LangChain Google GenAI libraries not found. Please install with: pip install langchain-google-genai")
#    st.stop()

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
    "Downy": ["downy", "downey", "dumming", "downie"],
    "Gain": ["gain", "gains", "gane", "gayne", "game"],
    "Tide": ["tide", "tied", "tyde", "tyde"],
    "Unstopables": ["unstopables", "unstoppables", "unstoppable", "unstopable", "unstopabls"]
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

def transcribe_with_whisper(audio_path, model_size="large"):
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
    non_attentive_color = '#999999'  # Grey
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
    non_attentive_color = '#999999'  # Grey
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
                                duration, media_vehicle, attention_df):
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
        results['simulation_timelines'] = [
            create_long_form_simulation_timeline(sim, duration) 
            for sim in simulations
        ]
        results['simulations'] = simulations
    
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
def display_attentive_branding_rating(score, media_form):
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
        f"##### Your attentive branding score of **{score:.0f}%** is "
        f"<span style='background-color: {rating_info['color']}; padding: 2px 8px; "
        f"border-radius: 3px; font-weight: bold; color: #000;'>{rating_info['rating']}</span>.",
        unsafe_allow_html=True
    )
    
#    st.markdown("Attentive Branding Rating:")
    st.markdown("<p style='font-size:10pt;'>Attentive Branding Rating:</p>",unsafe_allow_html=True)

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
                <div style="font-weight: bold; font-size: 12px; color: #000;">{r['label']}</div>
                <div style="font-style: italic; font-size: 10px; color: #000; margin-top: 0px;line-height: 1.2;">{r['range']}</div>
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

def log_analysis_results(results, video_file_name, brand_name, media_vehicle,
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
               branding_category
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
           round(attentive_branding_score, 2) if attentive_branding_score is not None else None
       ]]
       
       # --- Upload to Google Sheets (DATA ONLY, NO HEADERS) ---
       append_to_gsheet(SECOND_LOG_SHEET_ID, second_data)
       append_to_gsheet(SUMMARY_LOG_SHEET_ID, summary_data)
       
       return analysis_id, None
   
   except Exception as e:
       return None, f"Logging failed: {str(e)}"
    
# AI Recommendation
def get_ai_recommendation(analysis_summary, brand_name, media_vehicle, google_api_key):
    """Get AI recommendation using LangChain and Google Gemini"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=google_api_key
        )
        
        system_prompt = """You are an expert media strategist and advertising analyst. Your job is to provide recommendations about whether a video advertisement should be placed on a specific media vehicle based on branding and attention analysis results.

Consider these factors in your analysis:
1. Brand presence strength (audio + visual)
2. Media vehicle requirements and best practices
3. Audience attention patterns for the chosen media vehicle
4. Consumer experience on the chosen media vehicle
5. Brand recall and recognition effectiveness

Provide a clear recommendation with supporting reasoning."""

        human_prompt = f"""
Brand: {brand_name}
Intended Media Vehicle: {media_vehicle}

Branding and Attention Analysis Results:
{analysis_summary}

Based on this analysis, should this video be placed on {media_vehicle}? 

Please provide:
1. Clear recommendation (Highly Recommended / Recommended / Conditional / Not Recommended)
2. Key reasoning points
3. Specific suggestions for optimization if needed 

Keep your response concise but comprehensive."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        return f"Error generating AI recommendation: {e}. Please check your Google API key and try again."

def cleanup_temp_files(audio_file, frames_dir):
    try:
        if os.path.exists(audio_file):
            os.remove(audio_file)
        if os.path.exists(frames_dir):
            safe_remove_directory(frames_dir)
    except Exception as e:
        st.warning(f"Cleanup warning: {e}")

# Main Process Function
def process_video_analysis(video_file, brand_name, media_vehicle, google_api_key, attention_df):
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
        
        transcription_result = transcribe_with_whisper(audio_path, "base")  # Using base model for speed
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
                duration, media_vehicle, attention_df
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

Total Branding Coverage: {round(branding_percentage)}% of video duration
"""
        
        # Add attention analysis to summary if available
        if attention_results:
            analysis_summary += f"""

#        Attention-Based Branding Analysis ({attention_results['media_form']}):
#        - Attentive Seconds Analyzed: {attention_results['attentive_seconds']} seconds
#        - Attentive Branding Score: {attention_results['attentive_branding_score']:.0f}%"""
            
            # Add Short Form specific context
#            if attention_results['media_form'] == 'Short Form or Social':
#                if 'watch_time_seconds' in attention_results:
#                   analysis_summary += f"""
#        - Average Watch Time: {attention_results['watch_time_seconds']} seconds
#        - Total Seconds Classified: {attention_results['total_classified_seconds']} seconds"""
            
#            # Add Long Form specific context
#            elif attention_results['media_form'] == 'Long Form':
#                if 'branding_percentage' in attention_results:
#                    analysis_summary += f"""
#        - Overall Video Branding Percentage: {attention_results['branding_percentage']:.0f}%"""
            
#            analysis_summary += f"""

#Branding Distribution During Attentive Moments:
#- Visual and Audio Branding: {attention_results['attentive_branding_distribution']['Visual and Audio Branding']:.0f}%
#- Audio Branding Only: {attention_results['attentive_branding_distribution']['Audio Branding Only']:.0f}%
#- Visual Branding Only: {attention_results['attentive_branding_distribution']['Visual Branding Only']:.0f}%
#- No Branding Present: {attention_results['attentive_branding_distribution']['No Branding Present']:.0f}%
#"""
        # Get AI recommendation
        status_text.text("Generating AI recommendation...")
        progress_bar.progress(95)
        
        ai_recommendation = get_ai_recommendation(analysis_summary, brand_name, media_vehicle, google_api_key)
        
        # Create visualizations
        timeline_fig = create_timeline_visualization(final_categories, duration)
        summary_fig = create_summary_visualization(final_categories, duration)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Log results to Google Sheets
        analysis_id, log_error = log_analysis_results(
            {
                'duration': duration,
                'category_counts': category_counts,
                'branding_percentage': branding_percentage,
                'attention_results': attention_results
            },
            video_file.name,
            brand_name,
            media_vehicle,
            final_categories,
            audio_results,
            visual_results
        )
        
        if log_error:
            st.warning(f"Results were not logged: {log_error}")
        else:
            st.success(f"Analysis logged with ID: {analysis_id}")
        
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
            'attention_viz': attention_viz
        }, None
        
    except Exception as e:
        return None, f"Analysis failed: {str(e)}"

# Streamlit UI
def main():
    st.title("🎬 AI Branding Analysis Chatbot")
    
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

                if results.get('attention_results'):
#                    st.subheader(f"Branding Attention Analysis: Average Consumer Viewing Experience")
                    st.markdown("<h3 style='text-decoration: underline;'>Branding Attention Analysis: Average Consumer Viewing Experience</h3>",unsafe_allow_html=True)   

                    attn = results['attention_results']
                    
                    # Display the attentive branding rating
                    display_attentive_branding_rating(
                        attn['attentive_branding_score'], 
                        attn['media_form']
                    )

                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Attentive Branding Score", 
                                f"{attn['attentive_branding_score']:.0f}%")
                        
                    with col2:
                        st.metric("Average Attentive Seconds", 
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
                        caption_stats = calculate_short_form_caption_stats(
                            message['results']['attention_results']['audio_results'],
                            message['results']['attention_results']['visual_results'],
                            attn['watch_time_seconds']
                        )
                        caption_text = format_caption_with_plurals(
                            caption_stats['total_branding_seconds'],
                            caption_stats['visual_and_audio_count'],
                            caption_stats['audio_only_count'],
                            caption_stats['visual_only_count']
                        )
                        st.markdown(caption_text)
                    
                    # Display simulation timelines for Long Form
                    if 'simulation_timelines' in attn:
                        st.markdown("#### Viewing Experience Simulations:")
#                        st.subheader("Viewing Experience Simulations")
                        for i, timeline_fig in enumerate(attn['simulation_timelines'], 1):
                            st.pyplot(timeline_fig)
                            
                            # Add caption for each simulation
                            simulation = attn['simulations'][i-1]
                            caption_stats = calculate_long_form_simulation_caption_stats(
                                simulation,
                                message['results']['duration']
                            )
                            caption_text = format_long_form_caption_with_plurals(
                                i,
                                caption_stats['total_branding_seconds'],
                                caption_stats['visual_and_audio_count'],
                                caption_stats['audio_only_count'],
                                caption_stats['visual_only_count'],
                                caption_stats['branding_percentage'],
                                caption_stats['no_branding_percentage']
                            )
#                            st.caption(caption_text)
                            a, b, c = st.columns(3)
                            with a:
                                st.metric("Visual and Audio Branding Seen/Heard", 
                                        f"{caption_stats['visual_and_audio_count']} sec")
                                
                            with b:
                                st.metric("Visual Branding Seen", 
                                        f"{caption_stats['visual_only_count']} sec")
                                
                            with c:
                                st.metric("Audio Branding Heard", 
                                        f"{caption_stats['audio_only_count']} sec")
                    
                    # Keep the original distribution chart if it exists
                    if results.get('attention_viz'):
                        st.pyplot(results['attention_viz'])

#                st.subheader("Branding Analysis Summary")
#                st.text(results['analysis_summary'])

                st.divider()

#                st.subheader("Branding Summary")
                st.markdown("<h3 style='text-decoration: underline;'>Branding Summary</h3>", unsafe_allow_html=True)
                
                st.markdown("#### Branding Timeline:")
#                st.subheader("Branding Timeline")
                st.pyplot(results['timeline_fig'])

                st.markdown("#### Branding Distribution:")
                st.text(f"{results['total_branding_coverage']}")
#                st.subheader("Branding Distribution")
                st.pyplot(results['summary_fig'])

                st.divider()

#                st.subheader("AI Recommendation")
                st.markdown(f"<h3 style='text-decoration: underline;'>Is the creative best suited for the intended media vehicle?</h3>", unsafe_allow_html=True)
                st.caption("The following recommendations were generated with help of AI. The insights are for informational purposes only and should be reviewed with human judgment.")
                st.markdown("<p style='font-size:10pt;'><i>This section uses AI to provide recommendations about whether a creative should be placed on the intended media vehicle specified as input. " \
                "The recommendation of Highly Recommended, Recommended, Conditional, or Not Recommended is based on the branding and attention analysis results.</i></p>", unsafe_allow_html=True)
                st.write(results['ai_recommendation'])
        
    # Input section
    st.subheader("Provide Your Inputs")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        uploaded_file = st.file_uploader("Upload a video file", type=['mp4'])
    
    with col2:
        selected_brand = st.selectbox("Select Brand", list(BRAND_VARIATIONS.keys()))
    
    with col3:
        selected_media_vehicle = st.selectbox("Select Intended Media Vehicle", MEDIA_VEHICLES)
    
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
                    uploaded_file, selected_brand, selected_media_vehicle, GOOGLE_API_KEY, attention_df
                )
                
                if error:
                    st.error(f"Analysis failed: {error}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Analysis failed: {error}"
                    })
                else:
                    # Display results
                    st.subheader("Analysis Summary")
                    st.text(results['analysis_summary'])

                    st.subheader("Branding Timeline")
                    st.pyplot(results['timeline_fig'])

                    st.subheader("Branding Distribution")
                    st.pyplot(results['summary_fig'])
        
                    st.subheader("AI Recommendation")
                    st.write(results['ai_recommendation'])
                    
                    # Add assistant message with results
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "**Analysis Complete!**\n\nI've analyzed your video and generated recommendations. Check the detailed results below!",
                        "results": results
                    })
            
            st.rerun()

if __name__ == "__main__":
    main()
