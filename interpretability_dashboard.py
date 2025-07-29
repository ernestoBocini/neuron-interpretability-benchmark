import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
import json
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="CLIP Neuron Analysis Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS combining both apps' styling
st.markdown("""
<style>
    /* Main container styles */
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #333;
        font-weight: 500;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #4CAF50 0%, #2196F3 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
        background-color: #fff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Cards and containers */
    .feature-vis {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        background-color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
    }
    
    /* Neuron grid styling */
    .neuron-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
        grid-gap: 15px;
    }
    .neuron-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 8px;
        text-align: center;
        background-color: white;
        transition: transform 0.2s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .neuron-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f1f1f1;
        border-radius: 8px 8px 0 0;
        padding: 5px 5px 0 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e0e0e0;
        border-radius: 8px 8px 0 0;
        gap: 1px;
        padding: 10px 15px;
        font-weight: 500;
        color: #333;
        border: none;
        transition: background-color 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #8a2be2;
        color: white;
    }
    
    /* Component legend styling */
    .component-legend {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: #f5f5f5;
        border-radius: 20px;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background-color: #2c3e50;
        color: #ecf0f1;
    }
    
    .stSidebar label, .stSidebar span, .stSidebar div {
        color: #ecf0f1 !important;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #8a2be2 !important;
        border-bottom: 1px solid #4a6278;
        padding-bottom: 5px;
        margin-top: 20px;
        font-weight: 600;
    }
    
    /* Hide default Streamlit components */
    footer {
        visibility: hidden;
    }
    #MainMenu {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# BENCHMARK DATA AND FUNCTIONS
# ============================================================================

@st.cache_data
def load_benchmark_data():
    """Load the interpretability benchmark data"""
    data = {
        'Rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Concept': ['Fire', 'Mushrooms', 'Droplets', 'Sailboat', 'Raised hand', 
                   'Trump', 'Fashion Model', 'Arabic Alphabet', 'Australia', 'Puppies'],
        'Neuron': [297, 1157, 967, 363, 1116, 89, 1424, 479, 513, 355],
        'S': [1.000, 0.999, 0.998, 0.999, 0.990, 1.000, 0.993, 1.000, 0.991, 0.997],
        'C': [0.249, 0.053, 0.143, 0.146, 0.161, 0.291, 0.371, 0.000, 0.139, 0.281],
        'R': [0.610, 0.861, 0.644, 0.452, 0.421, 0.409, 0.285, 0.667, 0.477, 0.250],
        'H': [0.880, 0.813, 0.758, 0.863, 0.868, 0.723, 0.562, 0.411, 0.430, 0.376],
        'InterpScore': [0.685, 0.682, 0.636, 0.615, 0.610, 0.606, 0.552, 0.520, 0.509, 0.476]
    }
    return pd.DataFrame(data)

def score_to_color(score, min_score, max_score):
    """Convert score to color on yellow-to-green scale"""
    normalized = (score - min_score) / (max_score - min_score) if max_score > min_score else 0.5
    red = int(255 - (255 - 76) * normalized)
    green = int(235 + (175 - 235) * normalized)
    blue = int(59 - 59 * normalized)
    return f"rgb({red}, {green}, {blue})"

def create_styled_benchmark_dataframe(df, sort_by='InterpScore'):
    """Create a styled dataframe with color coding"""
    df_display = df.copy()
    ascending = True if sort_by in ['Rank'] else False
    df_display = df_display.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    df_display['Rank'] = range(1, len(df_display) + 1)
    
    metrics = ['S', 'C', 'R', 'H', 'InterpScore']
    color_ranges = {}
    for metric in metrics:
        color_ranges[metric] = (df_display[metric].min(), df_display[metric].max())
    
    return df_display, color_ranges

# ============================================================================
# MICROSCOPE FUNCTIONS
# ============================================================================

@st.cache_resource
def load_activations(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        result = {}
        for key in data.files:
            result[key] = data[key]
        return result
    except Exception as e:
        st.error(f"Error loading activations file: {e}")
        return None

@st.cache_resource
def get_generated_images(folder_path):
    gen_images = {}
    try:
        for file_path in glob.glob(os.path.join(folder_path, "*.png")):
            filename = os.path.basename(file_path)
            if "neuron" in filename:
                parts = filename.split("neuron")[1].split("_")[0]
                neuron_idx = int(parts)
                gen_images[neuron_idx] = file_path
    except Exception as e:
        st.error(f"Error loading generated images: {e}")
    return gen_images

def display_top_images(activations, paths, neuron_idx, num_images=200):
    neuron_activations = activations[:, neuron_idx]
    top_indices = np.argsort(neuron_activations)[-num_images:][::-1]
    top_paths = [paths[i] for i in top_indices]
    top_activations = [neuron_activations[i] for i in top_indices]
    return top_paths, top_activations

def create_activation_heatmap(activations, neuron_idx, num_top_images=100):
    neuron_activations = activations[:, neuron_idx]
    top_indices = np.argsort(neuron_activations)[-num_top_images:][::-1]
    top_activations = neuron_activations[top_indices]
    
    grid_size = int(np.ceil(np.sqrt(len(top_activations))))
    heatmap_data = np.zeros((grid_size, grid_size))
    
    for i, val in enumerate(top_activations):
        if i < grid_size * grid_size:
            row = i // grid_size
            col = i % grid_size
            heatmap_data[row, col] = val
    
    colors = [(1, 1, 1), (0.54, 0.17, 0.89)]
    cmap = LinearSegmentedColormap.from_list("white_to_purple", colors)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(heatmap_data, cmap=cmap)
    plt.colorbar(im, ax=ax, label='Activation Value')
    ax.set_title(f'Top {num_top_images} Activations for Neuron {neuron_idx}', fontsize=12)
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 CLIP Neuron Analysis Platform</h1>
        <p>Comprehensive neuron interpretability benchmark and microscope visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create main navigation tabs
    main_tab1, main_tab2 = st.tabs(["📊 Interpretability Benchmark", "🔬 Neuron Microscope"])
    
    # ========================================================================
    # TAB 1: INTERPRETABILITY BENCHMARK
    # ========================================================================
    with main_tab1:
        st.header("Neuron Interpretability Benchmark")
        
        # Load benchmark data
        df = load_benchmark_data()
        
        # Sidebar for benchmark
        with st.sidebar:
            st.header("📊 Benchmark Controls")
            
            sort_options = {
                'Overall Score': 'InterpScore',
                'Selectivity (S)': 'S', 
                'Causality (C)': 'C',
                'Robustness (R)': 'R',
                'Human Consistency (H)': 'H',
                'Neuron ID': 'Neuron'
            }
            
            sort_by = st.selectbox(
                "Sort by Metric:",
                options=list(sort_options.keys()),
                index=0
            )
            
            st.markdown("""
            ### 🔍 Component Meanings
            
            **S - Selectivity**: How well the neuron discriminates between target and non-target concepts
            
            **C - Causality**: How much the neuron causally influences the final model output
            
            **R - Robustness**: How consistent the neuron's behavior is across image perturbations
            
            **H - Human Consistency**: How well the neuron's high activations align with human recognition
            """)
        
        # Main benchmark content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🏆 Leaderboard")
            
            # Component legend
            st.markdown("""
            <div class="component-legend">
                <div class="legend-item">
                    <div style="width: 20px; height: 20px; background: #4CAF50; border-radius: 50%;"></div>
                    <span>S - Selectivity</span>
                </div>
                <div class="legend-item">
                    <div style="width: 20px; height: 20px; background: #2196F3; border-radius: 50%;"></div>
                    <span>C - Causality</span>
                </div>
                <div class="legend-item">
                    <div style="width: 20px; height: 20px; background: #FF9800; border-radius: 50%;"></div>
                    <span>R - Robustness</span>
                </div>
                <div class="legend-item">
                    <div style="width: 20px; height: 20px; background: #9C27B0; border-radius: 50%;"></div>
                    <span>H - Human Consistency</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sort and display dataframe
            df_sorted, color_ranges = create_styled_benchmark_dataframe(df, sort_options[sort_by])
            
            def highlight_scores(val, column, color_ranges):
                min_val, max_val = color_ranges[column]
                color = score_to_color(val, min_val, max_val)
                return f'background-color: {color}; color: black; font-weight: bold'
            
            styled_df = df_sorted.style.format({
                'S': '{:.3f}',
                'C': '{:.3f}', 
                'R': '{:.3f}',
                'H': '{:.3f}',
                'InterpScore': '{:.3f}'
            })
            
            for col in ['S', 'C', 'R', 'H', 'InterpScore']:
                styled_df = styled_df.applymap(
                    lambda x: highlight_scores(x, col, color_ranges),
                    subset=[col]
                )
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Concept": st.column_config.TextColumn("Concept", width="medium"),
                    "Neuron": st.column_config.NumberColumn("Neuron", width="small"),
                    "S": st.column_config.NumberColumn("S", width="small"),
                    "C": st.column_config.NumberColumn("C", width="small"),
                    "R": st.column_config.NumberColumn("R", width="small"),
                    "H": st.column_config.NumberColumn("H", width="small"),
                    "InterpScore": st.column_config.NumberColumn("InterpScore", width="medium")
                }
            )
            
            # Add button to explore neuron in microscope
            selected_row = st.selectbox(
                "Select a neuron to explore in microscope:",
                options=range(len(df_sorted)),
                format_func=lambda x: f"#{df_sorted.iloc[x]['Neuron']}: {df_sorted.iloc[x]['Concept']} (Score: {df_sorted.iloc[x]['InterpScore']:.3f})"
            )
            
            if st.button("🔬 Explore in Microscope"):
                selected_neuron = df_sorted.iloc[selected_row]['Neuron']
                st.session_state['microscope_neuron'] = selected_neuron
                st.success(f"Navigate to the Microscope tab to explore Neuron {selected_neuron}")
        
        with col2:
            st.subheader("📈 Statistics")
            
            mean_score = df['InterpScore'].mean()
            std_score = df['InterpScore'].std()
            max_score = df['InterpScore'].max()
            min_score = df['InterpScore'].min()
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Overall Statistics</h4>
                <p><strong>Mean Score:</strong> {mean_score:.3f} ± {std_score:.3f}</p>
                <p><strong>Range:</strong> {min_score:.3f} - {max_score:.3f}</p>
                <p><strong>Total Neurons:</strong> {len(df)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Top performers
            top_3 = df.nlargest(3, 'InterpScore')
            st.markdown("""
            <div class="metric-card">
                <h4>🏆 Top Performers</h4>
            """, unsafe_allow_html=True)
            
            for i, (_, row) in enumerate(top_3.iterrows()):
                st.markdown(f"""
                <p><strong>{i+1}. {row['Concept']}</strong> (Neuron {row['Neuron']})<br>
                Score: {row['InterpScore']:.3f}</p>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 2: NEURON MICROSCOPE
    # ========================================================================
    with main_tab2:
        st.header("CLIP Neuron Microscope")
        
        # Check if coming from benchmark
        initial_neuron = 1
        if 'microscope_neuron' in st.session_state:
            initial_neuron = st.session_state['microscope_neuron']
        
        # Microscope sidebar configuration
        with st.sidebar:
            st.header("🔬 Microscope Controls")
            
            # Data split selection
            split_option = st.radio(
                "Data Split",
                ["Train", "Val"],
                index=1,
                horizontal=True
            )
            
            # Set file paths based on split
            if split_option == "Train":
                npz_path = "../clip_imagenet_responses/clip_neuron_activations_train.npz"
                image_base_dir = "/media/data_12tb/data_backup/Datasets/ImageNet2012/train"
            else:
                npz_path = "../clip_imagenet_responses/clip_neuron_activations_val.npz"
                image_base_dir = "/media/data_12tb/data_backup/Datasets/ImageNet2012/val"
            
            gen_imgs_folder = st.text_input(
                "Generated Images Folder", 
                value="../gen_imgs"
            )
            
            # Model configuration
            st.markdown("### Model Configuration")
            st.markdown("**CLIP RN50x4**")
            st.markdown("OpenAI's CLIP model with ResNet50x4 backbone")
            
            # Neuron selection
            st.markdown("### Neuron Selection")
            selected_neuron = st.number_input(
                "Enter Neuron Index (0-2559)", 
                min_value=0, 
                max_value=2559,
                value=initial_neuron
            )
            
            # Neuron suggestions from benchmark
            st.markdown("### 🏆 Benchmark Neurons")
            benchmark_df = load_benchmark_data()
            
            for _, row in benchmark_df.head(5).iterrows():
                if st.button(f"#{row['Neuron']}: {row['Concept']}", key=f"bench_{row['Neuron']}"):
                    st.session_state['microscope_neuron'] = row['Neuron']
                    st.experimental_rerun()
        
        # Load microscope data
        activations_data = load_activations(npz_path)
        
        if activations_data is None:
            st.error(f"Failed to load activations from {npz_path}")
            return
        
        activations = activations_data['activations']
        if 'paths' in activations_data:
            paths = activations_data['paths']
        elif 'filenames' in activations_data:
            paths = activations_data['filenames']
        else:
            st.error("Could not find image paths in the NPZ file")
            return
        
        if isinstance(paths, np.ndarray):
            paths = paths.tolist()
        
        gen_images = get_generated_images(gen_imgs_folder)
        
        # Display neuron information
        st.markdown(f"### Neuron {selected_neuron} Analysis")
        
        # Check if this neuron is in the benchmark
        neuron_in_benchmark = benchmark_df[benchmark_df['Neuron'] == selected_neuron]
        if not neuron_in_benchmark.empty:
            concept = neuron_in_benchmark.iloc[0]['Concept']
            score = neuron_in_benchmark.iloc[0]['InterpScore']
            st.info(f"🏆 This neuron is in the benchmark! Concept: **{concept}** | Score: **{score:.3f}**")
        
        # Create microscope tabs
        micro_tab1, micro_tab2, micro_tab3 = st.tabs(["🎨 Feature Visualization", "📸 Top Activations", "🌡️ Activation Heatmap"])
        
        with micro_tab1:
            st.markdown("#### Feature Visualization")
            
            if selected_neuron in gen_images:
                try:
                    img_path = gen_images[selected_neuron]
                    img = Image.open(img_path)
                    
                    neuron_activations = activations[:, selected_neuron]
                    max_activation = np.max(neuron_activations)
                    
                    draw_img = img.copy()
                    draw = ImageDraw.Draw(draw_img)
                    
                    try:
                        font = ImageFont.truetype("Arial.ttf", 20)
                    except IOError:
                        font = ImageFont.load_default()
                    
                    text = f"Max Activation: {max_activation:.4f}"
                    
                    try:
                        text_width, text_height = draw.textsize(text, font=font)
                    except AttributeError:
                        text_width, text_height = font.getsize(text)
                        
                    position = ((img.width - text_width) // 2, 10)
                    text_bg = [(position[0] - 5, position[1] - 5), 
                            (position[0] + text_width + 5, position[1] + text_height + 5)]
                    draw.rectangle(text_bg, fill=(0, 0, 0, 128))
                    draw.text(position, text, font=font, fill=(255, 255, 255, 255))
                    
                    st.image(draw_img, caption=f"Generated visualization for neuron {selected_neuron}", width=300)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            else:
                st.info(f"No generated visualization found for neuron {selected_neuron}")
        
        with micro_tab2:
            st.markdown("#### Top Activating Images")
            
            top_paths, top_activations = display_top_images(activations, paths, selected_neuron, num_images=200)
            full_paths = [os.path.join(image_base_dir, p) for p in top_paths]
            
            num_display = st.slider("Number of top images to display", 5, 50, 20)
            
            if len(full_paths) > 0:
                total_rows = (num_display + 4) // 5
                
                for row in range(total_rows):
                    cols = st.columns(5)
                    
                    for col_idx in range(5):
                        img_idx = row * 5 + col_idx
                        
                        if img_idx < num_display and img_idx < len(full_paths):
                            with cols[col_idx]:
                                try:
                                    img_path = full_paths[img_idx]
                                    if os.path.exists(img_path):
                                        img = Image.open(img_path)
                                        
                                        draw_img = img.copy()
                                        draw = ImageDraw.Draw(draw_img)
                                        
                                        try:
                                            font = ImageFont.truetype("Arial.ttf", 16)
                                        except IOError:
                                            font = ImageFont.load_default()
                                        
                                        activation_value = top_activations[img_idx]
                                        text = f"Act: {activation_value:.3f}"
                                        
                                        try:
                                            text_width, text_height = draw.textsize(text, font=font)
                                        except AttributeError:
                                            text_width, text_height = font.getsize(text)
                                        
                                        position = (10, 10)
                                        text_bg = [(position[0] - 2, position[1] - 2), 
                                                (position[0] + text_width + 2, position[1] + text_height + 2)]
                                        draw.rectangle(text_bg, fill=(0, 0, 0, 160))
                                        draw.text(position, text, font=font, fill=(255, 255, 255, 255))
                                        
                                        st.image(draw_img, use_column_width=True)
                                    else:
                                        st.markdown(f"<div style='border:1px solid #ddd; padding:10px; text-align:center;'>"
                                                f"<p>Image not found</p>"
                                                f"<p>Value: {top_activations[img_idx]:.3f}</p></div>", 
                                                unsafe_allow_html=True)
                                except Exception as e:
                                    st.markdown(f"Error: {e}")
            else:
                st.warning("No images available for this neuron.")
        
        with micro_tab3:
            st.markdown("#### Activation Heatmap")
            heatmap_buf = create_activation_heatmap(activations, selected_neuron)
            st.image(heatmap_buf, caption=f"Activation Heatmap for Neuron {selected_neuron}")

if __name__ == "__main__":
    main()