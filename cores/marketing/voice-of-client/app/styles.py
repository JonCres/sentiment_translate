import streamlit as st

def apply_custom_styles():
    st.markdown(
        """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* ===== MAIN LAYOUT ===== */
        .main {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            color: #1a1a1a;
            font-family: 'Inter', sans-serif;
        }
        
        /* ===== TYPOGRAPHY WITH GRADIENTS ===== */
        h1 {
            color: #00658f !important;
            font-weight: 800 !important;
            font-size: 2.75rem !important;
            margin-bottom: 0.5rem !important;
            letter-spacing: -0.02em !important;
        }
        
        h2 {
            color: #00658f !important;
            font-weight: 700 !important;
            font-size: 1.9rem !important;
            margin-top: 2rem !important;
            letter-spacing: -0.01em !important;
        }
        
        h3 {
            color: #0077aa !important;
            font-weight: 600 !important;
            font-size: 1.4rem !important;
        }
        
        /* Ensure all headings are visible */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            opacity: 1 !important;
            visibility: visible !important;
        }
        
        /* ===== ENHANCED METRICS ===== */
        .stMetricLabel {
            color: #495057 !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .stMetricValue {
            color: #0d1b2a !important;
            font-weight: 800 !important;
            font-size: 2.25rem !important;
        }
        
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 28px 24px;
            border-radius: 20px;
            border: 2px solid transparent;
            background-image: 
                linear-gradient(white, white),
                linear-gradient(135deg, #00a8cc 0%, #0077aa 100%);
            background-origin: border-box;
            background-clip: padding-box, border-box;
            box-shadow: 
                0 4px 12px rgba(0, 0, 0, 0.05),
                0 1px 3px rgba(0, 0, 0, 0.08);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        div[data-testid="metric-container"]:hover {
            transform: translateY(-4px);
            box-shadow: 
                0 12px 24px rgba(0, 168, 204, 0.15),
                0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* ===== INSIGHT CARDS ===== */
        .insight-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafb 100%);
            padding: 24px;
            border-radius: 16px;
            border-left: 5px solid #00a8cc;
            margin: 20px 0;
            box-shadow: 
                0 2px 8px rgba(0, 0, 0, 0.04),
                0 1px 3px rgba(0, 0, 0, 0.06);
            color: #2c3e50;
            transition: all 0.3s ease;
        }
        
        .insight-card:hover {
            transform: translateX(4px);
            box-shadow: 
                0 4px 16px rgba(0, 168, 204, 0.12),
                0 2px 6px rgba(0, 0, 0, 0.08);
        }
        
        /* ===== ANIMATED ALERTS ===== */
        .alert-critical {
            color: #c62828;
            font-weight: 600;
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            padding: 16px 20px;
            border-radius: 12px;
            border-left: 5px solid #c62828;
            box-shadow: 0 2px 8px rgba(198, 40, 40, 0.15);
            animation: pulse-critical 2s infinite;
        }
        
        @keyframes pulse-critical {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.85; }
        }
        
        .alert-warning {
            color: #ef6c00;
            font-weight: 600;
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            padding: 16px 20px;
            border-radius: 12px;
            border-left: 5px solid #ef6c00;
            box-shadow: 0 2px 8px rgba(239, 108, 0, 0.15);
        }
        
        .alert-success {
            color: #2e7d32;
            font-weight: 600;
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            padding: 16px 20px;
            border-radius: 12px;
            border-left: 5px solid #2e7d32;
            box-shadow: 0 2px 8px rgba(46, 125, 50, 0.15);
        }
        
        /* ===== CUSTOMER QUOTES WITH DECORATION ===== */
        .customer-quote {
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border-left: 5px solid #00a8cc;
            padding: 20px 24px;
            margin: 16px 0;
            border-radius: 12px;
            font-style: italic;
            color: #2c3e50;
            box-shadow: 
                0 2px 6px rgba(0, 0, 0, 0.04),
                0 1px 3px rgba(0, 0, 0, 0.06);
            position: relative;
            overflow: hidden;
        }
        
        .customer-quote::before {
            content: '"';
            position: absolute;
            top: -10px;
            left: 10px;
            font-size: 80px;
            color: rgba(0, 168, 204, 0.1);
            font-family: Georgia, serif;
        }
        
        /* ===== ENHANCED SIDEBAR ===== */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
            box-shadow: 2px 0 12px rgba(0, 0, 0, 0.08);
        }
        
        [data-testid="stSidebar"] * {
            color: #2c3e50 !important;
        }
        
        /* ===== INTERACTIVE TABS ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background: transparent;
            padding: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border-radius: 12px 12px 0 0;
            padding: 14px 28px;
            font-weight: 600;
            color: #495057;
            border: 2px solid #e9ecef;
            border-bottom: none;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: #f8f9fa;
            transform: translateY(-2px);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #00a8cc 0%, #0077aa 100%);
            color: white !important;
            border-color: #00a8cc;
            box-shadow: 0 4px 12px rgba(0, 168, 204, 0.3);
        }
        
        /* ===== ENHANCED EXPANDERS ===== */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border: 2px solid #e9ecef;
            border-radius: 12px;
            color: #2c3e50 !important;
            font-weight: 600;
            padding: 16px 20px !important;
            transition: all 0.3s ease;
        }
        
        .streamlit-expanderHeader:hover {
            border-color: #00a8cc;
            box-shadow: 0 2px 8px rgba(0, 168, 204, 0.15);
        }
        
        /* ===== ENHANCED BUTTONS ===== */
        .stButton button {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            color: #00a8cc;
            border: 2px solid #00a8cc;
            font-weight: 600;
            padding: 12px 24px;
            border-radius: 10px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        .stButton button:hover {
            background: linear-gradient(135deg, #00a8cc 0%, #0077aa 100%);
            color: white;
            border-color: #0077aa;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 168, 204, 0.3);
        }
        
        /* ===== ENHANCED INPUT FIELDS ===== */
        .stTextInput input, .stSelectbox select, .stMultiSelect {
            background: #ffffff !important;
            color: #2c3e50 !important;
            border: 2px solid #e9ecef !important;
            border-radius: 10px !important;
            padding: 12px 16px !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextInput input:focus, .stSelectbox select:focus {
            border-color: #00a8cc !important;
            box-shadow: 0 0 0 3px rgba(0, 168, 204, 0.1) !important;
        }
        
        /* ===== CUSTOM SCROLLBAR ===== */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #00a8cc 0%, #0077aa 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #0077aa 0%, #005580 100%);
        }
        
        /* ===== LOADING SPINNER ===== */
        .stSpinner > div {
            border-top-color: #00a8cc !important;
        }
        
        /* ===== PROGRESS BAR ===== */
        .stProgress > div > div {
            background-color: #00a8cc !important;
        }
        
        /* ===== INFO/WARNING/SUCCESS BOXES ===== */
        .stAlert {
            border-radius: 12px !important;
            border-left-width: 5px !important;
        }
        
        /* ===== TOOLTIPS ===== */
        [data-baseweb="tooltip"] {
            background: #1a1a1a !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        }
        
        /* Emergency visibility fix */
        h1, h2, h3, h4, h5, h6 {
            color: #00658f !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        
        [data-testid="stMarkdownContainer"] h1 {
            color: #00658f !important;
        }
        
        [data-testid="stMarkdownContainer"] h2 {
            color: #00658f !important;
        }
        
        [data-testid="stMarkdownContainer"] h3 {
            color: #0077aa !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
