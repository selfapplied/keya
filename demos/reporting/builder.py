import base64
from pathlib import Path
import importlib
from .registry import DEMO_REGISTRY, DemoInfo, Artifact

# --- Configuration ---
OUTPUT_DIR = Path(__file__).parent.parent.parent / ".out"
REPORT_FILE = OUTPUT_DIR / "report.html"
REPORT_TITLE = "Kéya Project Findings: A Comprehensive Overview"

# --- Demo Discovery ---

def discover_and_load_demos():
    """
    Finds and imports all demo modules to trigger their registration.
    """
    # Ensure we start with a clean registry
    DEMO_REGISTRY.clear()
    
    demos_path = Path(__file__).parent.parent
    for file_path in demos_path.glob("*.py"):
        if file_path.name.startswith("_") or file_path.name == "report.py":
            continue
        
        module_name = f"demos.{file_path.stem}"
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            print(f"Warning: Could not import demo module {module_name}: {e}")

# --- HTML Generation ---

def get_image_as_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        return ""

def generate_artifact_html(artifact: Artifact) -> str:
    """Generates the HTML for a single demo artifact."""
    # Ensure artifacts are saved in the output directory
    artifact_path = OUTPUT_DIR / Path(artifact.filename).name
    base64_img = get_image_as_base64(str(artifact_path))

    if not base64_img:
        return f"""
        <div class="artifact-missing">
            <p><strong>Artifact not found:</strong> {artifact_path}</p>
        </div>
        """
    return f"""
    <div class="artifact">
        <img src="data:image/png;base64,{base64_img}" alt="{artifact.caption}">
        <p class="caption"><em>{artifact.caption}</em></p>
    </div>
    """

def generate_demo_section_html(demo: DemoInfo) -> str:
    """Generates the HTML for a single demo section."""
    artifacts_html = "".join(generate_artifact_html(a) for a in demo.artifacts)
    claims_html = "".join(f"<li>{claim}</li>" for claim in demo.claims)
    
    return f"""
    <div class="demo-section">
        <h2>{demo.title}</h2>
        <p>{demo.description}</p>
        
        <h3>Claims</h3>
        <ul>{claims_html}</ul>
        
        <h3>Findings</h3>
        <p>{demo.findings}</p>
        
        <h3>Artifacts</h3>
        <div class="artifacts-container">
            {artifacts_html}
        </div>
    </div>
    """

def generate_report():
    """Generates the full HTML report from the demo registry."""
    discover_and_load_demos() # This populates the registry
    
    # Ensure the main output directory exists before demos run
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run demos which might generate artifacts
    for demo in DEMO_REGISTRY:
        if demo.func:
            print(f"Running demo: {demo.title}...")
            # Demos are expected to save artifacts to OUTPUT_DIR
            demo.func()

    if not DEMO_REGISTRY:
        print("No demos were found or registered. Report will be empty.")
        return
        
    demo_sections_html = "".join(generate_demo_section_html(d) for d in DEMO_REGISTRY)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{REPORT_TITLE}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6; margin: 0; padding: 20px;
            background-color: var(--bg-color);
            color: var(--text-color);
            transition: background-color 0.3s, color 0.3s;
        }}
        :root {{
            --bg-color: #f8f9fa;
            --text-color: #343a40;
            --container-bg: #ffffff;
            --header-color: #2c3e50;
            --border-color: #e9ecef;
            --link-color: #3498db;
            --code-bg: #e9ecef;
            --artifact-bg: #f8f9fa;
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-color: #1a1a1b;
                --text-color: #d7dadc;
                --container-bg: #272729;
                --header-color: #d7dadc;
                --border-color: #49494d;
                --link-color: #5db0e4;
                --code-bg: #3a3a3c;
                --artifact-bg: #2a2a2c;
            }}
            .artifact-container svg, .modal-content svg {{
                filter: invert(1) hue-rotate(180deg);
            }}
        }}
        .container {{
            max-width: 1200px; margin: 20px auto; padding: 40px; background: var(--container-bg);
            border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        }}
        h1, h2, h3, h4 {{
            color: var(--header-color); border-bottom: 3px solid var(--link-color); padding-bottom: 12px; margin-top: 40px;
        }}
        h1 {{ font-size: 2.8em; text-align: center; border: none; }}
        h2 {{ font-size: 2.2em; }}
        h3 {{ font-size: 1.6em; margin-top: 30px; border-bottom-width: 2px; }}
        h4 {{ font-size: 1.2em; margin-top: 25px; border-bottom: 1px dashed #ced4da; }}
        code {{
            background: var(--code-bg); padding: 0.2em 0.4em; margin: 0; font-size: 85%;
            border-radius: 3px; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        }}
        ul, ol {{ padding-left: 25px; }}
        li {{ margin-bottom: 10px; }}
        .demo-section {{
            margin-bottom: 60px;
            padding: 30px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: #fdfdff;
            background: var(--container-bg);
        }}
        .artifact-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }}
        .artifact-container {{
            border: 1px solid var(--border-color); border-radius: 8px; padding: 15px; background: var(--artifact-bg);
            box-shadow: 0 4px 12px rgba(0,0,0,0.05); text-align: center;
            display: flex; flex-direction: column; justify-content: space-between;
        }}
        .artifact-container img, .artifact-container svg {{
            max-width: 100%; height: auto; border-radius: 4px; margin-bottom: 15px;
            max-height: 600px;
            object-fit: contain;
            cursor: pointer;
        }}
        .artifact-container footer {{
            font-size: 0.9em; color: #6c757d; margin-top: auto;
        }}
        .modal {{
            display: none; position: fixed; z-index: 1000; left: 0; top: 0;
            width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.8);
            align-items: center; justify-content: center;
        }}
        .modal-content {{
            margin: auto; padding: 20px; max-width: 90%; max-height: 90%;
        }}
        .modal-content svg {{
             width: 100%; height: 100%;
             object-fit: contain;
        }}
        .modal.is-open {{
            display: flex;
        }}
        .close {{
            position: absolute; top: 20px; right: 35px; color: #f1f1f1;
            font-size: 40px; font-weight: bold; cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{REPORT_TITLE}</h1>
        <p>This report showcases the capabilities of the Keya engine through a series of automated demonstrations.</p>
        {demo_sections_html}
    </div>
    <div id="myModal" class="modal">
        <span class="close">&times;</span>
        <div class="modal-content" id="modal-content-host"></div>
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', () => {{
            const modal = document.getElementById('myModal');
            const modalContentHost = document.getElementById('modal-content-host');
            const closeModal = document.querySelector('.close');

            document.querySelectorAll('.artifact-container svg').forEach(svg => {{
                svg.addEventListener('click', () => {{
                    modalContentHost.innerHTML = '';
                    const clonedSvg = svg.cloneNode(true);
                    modalContentHost.appendChild(clonedSvg);
                    modal.classList.add('is-open');
                }});
            }});

            const close = () => modal.classList.remove('is-open');
            closeModal.addEventListener('click', close);
            modal.addEventListener('click', (event) => {{
                if (event.target === modal) {{
                    close();
                }}
            }});
        }});
    </script>
</body>
</html>
"""
    
    with open(REPORT_FILE, "w") as f:
        f.write(html_content)
        
    print(f"Report generated at {REPORT_FILE.resolve()}")

if __name__ == "__main__":
    generate_report()