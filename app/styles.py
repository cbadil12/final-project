# app/styles.py

import base64
from pathlib import Path

def set_styles(path: Path) -> str | None:
    if not path.exists():
        pic = None
    else:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        pic =f"data:{mime};base64,{b64}"

    STYLE=f"""
        <style>
        .main {{ background-color: #0F1015; color: #e5e7eb; }}
        .stMetric {{ background-color: #0F1015 !important; }}
        
        /* Portada ("Home") */
        .hero {{
        min-height: 72vh; padding: 4rem 2rem; border-radius: 16px;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background-color: #0F1015; color: #e5e7eb; position: relative; overflow: hidden;
        }}
        .hero::before{{
        content:""; position:absolute; inset:0; background-repeat: repeat;
        background-size: 220px; background-position: 0 0; opacity: 0.08;
        filter: grayscale(100%); background-image: url('{pic if pic else ""}');
        }}
        .hero h1{{ margin:0 0 .5rem 0; font-size: clamp(2.2rem, 4vw, 3.4rem); z-index:1; }}
        .hero .tagline{{ color:#94a3b8; font-size: 1.05rem; text-align:center; max-width: 900px; z-index:1; }}
        .hero .cta{{ margin-top: 1.5rem; z-index:1; }}

        /* Disclaimer message (footer global) */
        .disclaimer {{
        margin-top: 0.75rem;
        color: #94a3b8;
        font-size: 0.78rem;
        opacity: 0.90;
        text-align: center;
        line-height: 1.25;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] > div {{
            display: flex; flex-direction: column; height: 100vh;
        }}
        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {{
            display: flex; flex-direction: column; height: 100%;
        }}
        .sidebar-footer {{
            margin-top: auto; padding-top: 12px; border-top: 1px solid #2a2f3a;
            color: #94a3b8; font-size: 0.78rem; opacity: 0.90; line-height: 1.25;
        }}

        .sidebar-footer .meta {{ margin-bottom: 6px; opacity: 0.95; }}
        .sidebar-footer .disclaimer {{
            margin-top: 6px; font-size: 0.74rem; opacity: 0.85; text-align: left;
        }}
        </style>
        """
    return STYLE

ICONS = """
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    """