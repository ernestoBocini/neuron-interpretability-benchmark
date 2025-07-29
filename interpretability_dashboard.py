import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Neuron Interpretability Benchmark",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Brain-Score inspired styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50 0%, #2196F3 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
    }
    
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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Create sample data matching your benchmark results"""
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
    # Normalize score between 0 and 1
    normalized = (score - min_score) / (max_score - min_score) if max_score > min_score else 0.5
    
    # Create yellow to green gradient
    # Yellow: #FFEB3B, Green: #4CAF50
    red = int(255 - (255 - 76) * normalized)
    green = int(235 + (175 - 235) * normalized)
    blue = int(59 - 59 * normalized)
    
    return f"rgb({red}, {green}, {blue})"

def create_styled_dataframe(df, sort_by='InterpScore'):
    """Create a styled dataframe with color coding"""
    df_display = df.copy()
    
    # Sort by selected metric
    ascending = True if sort_by in ['Rank'] else False
    df_display = df_display.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    df_display['Rank'] = range(1, len(df_display) + 1)
    
    # Get min/max values for each metric for color scaling
    metrics = ['S', 'C', 'R', 'H', 'InterpScore']
    color_ranges = {}
    for metric in metrics:
        color_ranges[metric] = (df_display[metric].min(), df_display[metric].max())
    
    return df_display, color_ranges

def create_component_chart(df):
    """Create component breakdown chart"""
    components = ['S', 'C', 'R', 'H']
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Selectivity (S)', 'Causality (C)', 'Robustness (R)', 'Human Consistency (H)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for i, (component, color, pos) in enumerate(zip(components, colors, positions)):
        fig.add_trace(
            go.Bar(
                x=df['Concept'],
                y=df[component],
                name=component,
                marker_color=color,
                showlegend=False
            ),
            row=pos[0], col=pos[1]
        )
    
    fig.update_layout(
        height=600,
        title_text="Component Scores Breakdown",
        title_x=0.5,
        showlegend=False
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Neuron Interpretability Benchmark</h1>
        <p>Navigate our dashboard to view neuron interpretability metrics and rankings.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_sample_data()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        
        # Sort by selector
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
        
        # Component legend
        st.markdown("""
        ### 📋 Color Legend
        
        **🟢 High Performance** (Green)  
        Top scores in each metric
        
        **🟡 Low Performance** (Yellow)  
        Bottom scores in each metric
        
        *Colors are scaled relative to the range of each metric*
        """)
        
        # Component explanations
        st.markdown("""
        ### 🔍 Component Meanings
        
        **S - Selectivity**: How well the neuron discriminates between target and non-target concepts
        
        **C - Causality**: How much the neuron causally influences the final model output
        
        **R - Robustness**: How consistent the neuron's behavior is across image perturbations
        
        **H - Human Consistency**: How well the neuron's high activations align with human recognition
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🏆 Leaderboard")
        
        # Create component legend
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
        
        # Sort and display dataframe with colors
        df_sorted, color_ranges = create_styled_dataframe(df, sort_options[sort_by])
        
        # Create a function to apply background colors
        def highlight_scores(val, column, color_ranges):
            min_val, max_val = color_ranges[column]
            color = score_to_color(val, min_val, max_val)
            return f'background-color: {color}; color: black; font-weight: bold'
        
        # Apply styling to the dataframe
        styled_df = df_sorted.style.format({
            'S': '{:.3f}',
            'C': '{:.3f}', 
            'R': '{:.3f}',
            'H': '{:.3f}',
            'InterpScore': '{:.3f}'
        })
        
        # Apply background colors for each metric column
        for col in ['S', 'C', 'R', 'H', 'InterpScore']:
            styled_df = styled_df.applymap(
                lambda x: highlight_scores(x, col, color_ranges),
                subset=[col]
            )
        
        # Display the styled dataframe
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
        
        # Download button
        csv = df_sorted.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="neuron_interpretability_benchmark.csv",
            mime="text/csv"
        )
    
    with col2:
        st.header("📈 How to Interpret")
        
        # Statistics
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
        
        st.markdown("<br>", unsafe_allow_html=True)
        
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
        
        # Component analysis
        st.markdown("<br>", unsafe_allow_html=True)
        component_means = {
            'S': df['S'].mean(),
            'C': df['C'].mean(), 
            'R': df['R'].mean(),
            'H': df['H'].mean()
        }
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Component Averages</h4>
            <p><strong>Selectivity (S):</strong> {component_means['S']:.3f}</p>
            <p><strong>Causality (C):</strong> {component_means['C']:.3f}</p>
            <p><strong>Robustness (R):</strong> {component_means['R']:.3f}</p>
            <p><strong>Human Consistency (H):</strong> {component_means['H']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Component breakdown charts
    st.header("📊 Component Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Component Breakdown", "🎯 Scatter Analysis", "📈 Distribution"])
    
    with tab1:
        fig_components = create_component_chart(df)
        st.plotly_chart(fig_components, use_container_width=True)
    
    with tab2:
        # Scatter plot: Selectivity vs Human Consistency
        fig_scatter = px.scatter(
            df, x='S', y='H', 
            color='InterpScore',
            size='C',
            hover_data=['Concept', 'Neuron'],
            color_continuous_scale='Viridis',
            title="Selectivity vs Human Consistency (Size = Causality, Color = Overall Score)"
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        # Distribution of scores
        fig_dist = make_subplots(
            rows=1, cols=4,
            subplot_titles=['Selectivity', 'Causality', 'Robustness', 'Human Consistency']
        )
        
        components = ['S', 'C', 'R', 'H']
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
        
        for i, (comp, color) in enumerate(zip(components, colors)):
            fig_dist.add_trace(
                go.Histogram(x=df[comp], name=comp, marker_color=color, showlegend=False),
                row=1, col=i+1
            )
        
        fig_dist.update_layout(height=400, title_text="Distribution of Component Scores")
        st.plotly_chart(fig_dist, use_container_width=True)

if __name__ == "__main__":
    main()