"""
UI Styles for Causal AI Platform
Comprehensive CSS with academic micro-interactions and professional design
"""

import streamlit as st

def apply_custom_styles():
    """Apply all custom CSS styles to the Streamlit app"""
    st.markdown("""
<style>
    /* Page fade-in animation */
    .main {
        animation: pageLoad 0.8s ease-out;
    }
    
    @keyframes pageLoad {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Enhanced Hero Section with Micro-Interactions */
    .hero-section {
        text-align: center; 
        padding: 1.5rem 2rem; 
        background: linear-gradient(135deg, #004c6d 0%, #427aa1 50%, #6ca0dc 100%); 
        border-radius: 20px; 
        margin-bottom: 3rem; 
        color: white;
        box-shadow: 0 15px 40px rgba(0, 76, 109, 0.4), 0 5px 15px rgba(0, 76, 109, 0.2);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    .hero-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 50px rgba(0, 76, 109, 0.5), 0 8px 20px rgba(0, 76, 109, 0.3);
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 30% 20%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 70% 80%, rgba(108, 160, 220, 0.2) 0%, transparent 50%);
        pointer-events: none;
        transition: opacity 0.3s ease;
    }
    
    .hero-section:hover::before {
        opacity: 0.8;
    }
    
    .hero-section::after {
        content: '';
        position: absolute;
        top: -2px;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent 20%, #6ca0dc 50%, transparent 80%);
        opacity: 0.6;
        border-radius: 20px 20px 0 0;
        z-index: 3;
    }
    
    .hero-title {
        font-size: 2.8rem; 
        font-weight: 800; 
        margin: 0 auto 1rem auto; 
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.03em;
        color: white;
        line-height: 1.1;
        position: relative;
        z-index: 2;
        background: linear-gradient(45deg, #ffffff 0%, #e3f2fd 50%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: titleShine 3s ease-in-out infinite;
        transition: transform 0.3s ease;
        text-align: center !important;
        width: 100%;
        display: block;
        margin-left: auto;
        margin-right: auto;
        box-sizing: border-box;
    }
    
    .hero-section:hover .hero-title {
        transform: scale(1.02);
    }
    
    @keyframes titleShine {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.9; }
    }
    
    .hero-subtitle {
        font-size: 1.3rem; 
        font-weight: 400; 
        margin: 0 auto 1.2rem auto; 
        opacity: 0.95; 
        line-height: 1.6;
        max-width: 90%;
        padding: 0 2rem;
        color: rgba(255, 255, 255, 0.95);
        text-align: center !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
        position: relative;
        z-index: 2;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        transition: opacity 0.3s ease;
        width: 100%;
        display: block;
        box-sizing: border-box;
        margin-left: auto;
        margin-right: auto;
    }
    
    .hero-section:hover .hero-subtitle {
        opacity: 1;
    }
    
    .hero-subtitle strong {
        color: #e3f2fd;
        font-weight: 600;
    }
    
    /* Enhanced Step Headers with Completion States */
    .step-header {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
        padding: 1.3rem 1.8rem;
        border-left: 5px solid #3478bd;
        border-radius: 0 12px 12px 0;
        margin: 2.5rem 0 1.8rem 0;
        box-shadow: 0 6px 20px rgba(52, 120, 189, 0.25);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        cursor: pointer;
        border-top: 1px solid rgba(52, 120, 189, 0.2);
        border-bottom: 1px solid rgba(52, 120, 189, 0.2);
    }
    
    .step-header:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 30px rgba(52, 120, 189, 0.35);
        background: linear-gradient(135deg, #e8effc 0%, #d5e4f7 100%);
        border-left-color: #2563eb;
    }
    
    .step-header.step-completed {
        background: linear-gradient(135deg, #d1f2eb 0%, #a3e3d0 100%);
        border-left-color: #16a085;
        box-shadow: 0 6px 20px rgba(22, 160, 133, 0.3);
    }
    
    .step-header.step-completed:hover {
        background: linear-gradient(135deg, #d8f5ee 0%, #b0e7d5 100%);
        border-left-color: #0e8f7a;
        box-shadow: 0 8px 30px rgba(22, 160, 133, 0.4);
    }
    
    .step-header.step-completed::after {
        content: '✓';
        position: absolute;
        top: 50%;
        right: 1.8rem;
        transform: translateY(-50%);
        font-size: 1.8rem;
        color: #16a085;
        font-weight: bold;
        text-shadow: 0 2px 4px rgba(22, 160, 133, 0.3);
    }
    
    .step-header h2 {
        margin: 0;
        color: #2c3e50;
        font-weight: 700;
        font-size: 1.6rem;
        transition: color 0.3s ease;
        position: relative;
        z-index: 2;
    }
    
    .step-header:hover h2 {
        color: #1a252f;
    }
    
    .step-header.step-completed h2 {
        color: #0d5f4e;
    }

    /* Enhanced Cards and Interactive Elements */
    .professional-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .professional-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(108, 160, 220, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .professional-card:hover::before {
        left: 100%;
    }
    
    .professional-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: #6ca0dc;
    }
    
    /* Enhanced Buttons with Minimal Micro-Interactions */
    .stButton > button {
        background: linear-gradient(135deg, #4f86f7 0%, #6bcff6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(79, 134, 247, 0.2);
        text-transform: none;
        letter-spacing: 0.3px;
        border: 1px solid transparent;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(79, 134, 247, 0.3);
        background: linear-gradient(135deg, #5f96ff 0%, #7bdcff 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
        transition: all 0.15s ease;
        box-shadow: 0 4px 12px rgba(79, 134, 247, 0.3);
    }
    
    .stButton > button:disabled {
        background: linear-gradient(135deg, #94a3b8 0%, #cbd5e1 100%);
        color: #64748b;
        cursor: not-allowed;
        transform: none;
        box-shadow: 0 2px 8px rgba(148, 163, 184, 0.2);
    }
    
    .stButton > button:disabled:hover {
        transform: none;
        box-shadow: 0 2px 8px rgba(148, 163, 184, 0.2);
        background: linear-gradient(135deg, #94a3b8 0%, #cbd5e1 100%);
        border-color: transparent;
    }
    
    /* Enhanced Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8faff 0%, #ffffff 100%);
    }
    
    /* Enhanced File Uploader */
    .stFileUploader > div > div {
        transition: all 0.3s ease;
        border-radius: 10px;
    }
    
    .stFileUploader > div > div:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    }

    /* Enhanced Selectbox and Text Area */
    .stSelectbox > div > div {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .stSelectbox > div > div:hover {
        box-shadow: 0 4px 12px rgba(79, 134, 247, 0.2);
        transform: translateY(-1px);
    }
    
    .stTextArea > div > div > textarea {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .stTextArea > div > div > textarea:focus {
        box-shadow: 0 6px 20px rgba(79, 134, 247, 0.3);
        transform: scale(1.01);
    }

    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        background-color: rgba(79, 134, 247, 0.1);
    }

    /* Enhanced Metrics */
    .stMetric {
        transition: all 0.3s ease;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stMetric:hover {
        background-color: rgba(79, 134, 247, 0.05);
        transform: scale(1.02);
    }

    /* Enhanced Expanders */
    .streamlit-expanderHeader {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(79, 134, 247, 0.1);
        transform: translateX(5px);
    }
    
    /* Column hover effects */
    .stColumn {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .stColumn:hover {
        background-color: rgba(79, 134, 247, 0.02);
    }

    /* Info/Warning/Error box enhancements */
    .stAlert {
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stAlert:hover {
        transform: scale(1.01);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    /* Dataframe styling */
    .stDataFrame {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .stDataFrame:hover {
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)
