import base64
from pathlib import Path
import importlib
import os
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
    DEMO_REGISTRY.clear()
    
    demos_path = Path(__file__).parent.parent
    for file_path in sorted(demos_path.glob("*.py")):
        # Exclude files that are not demos
        if file_path.name.startswith("_") or file_path.name in ["report.py", "kshell.py"]:
            continue
        
        module_name = f"demos.{file_path.stem}"
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            print(f"Warning: Could not import demo module {module_name}: {e}")

# --- HTML Generation ---

def generate_artifact_html(artifact: Artifact) -> str:
    """Generates the HTML for a single demo artifact."""
    artifact_path = OUTPUT_DIR / Path(artifact.filename).name
    
    if not artifact_path.exists():
        return f'<div class="artifact-container"><footer>Artifact not found: {artifact.filename}</footer></div>'

    if artifact_path.suffix.lower() == ".svg":
        with open(artifact_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        return f"""
        <div class="artifact-container">
            {svg_content}
            <footer>{artifact.caption}</footer>
        </div>
        """
    elif artifact_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif"]:
        with open(artifact_path, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode("utf-8")
        return f"""
        <div class="artifact-container">
            <img src="data:image/{artifact_path.suffix[1:]};base64,{base64_img}" alt="{artifact.caption}" class="raster-image">
            <footer>{artifact.caption}</footer>
        </div>
        """
    else: # Handle text artifacts
        with open(artifact_path, "r", encoding="utf-8") as f:
            text_content = f.read()
        return f"""
        <div class="artifact-container text-artifact">
            <pre><code>{text_content}</code></pre>
            <footer>{artifact.caption}</footer>
        </div>
        """

def generate_demo_section_html(demo: DemoInfo) -> str:
    """Generates the HTML for a single demo section."""
    artifacts_html = "".join(generate_artifact_html(a) for a in demo.artifacts)
    claims_html = "".join(f"<li>{claim}</li>" for claim in demo.claims)
    
    return f"""
    <section class="demo-section" id="{demo.title.replace(' ', '-').lower()}">
        <h2>{demo.title}</h2>
        <p>{demo.description}</p>
        
        <h4>Claims</h4>
        <ul>{claims_html}</ul>
        
        <h4>Findings</h4>
        <p>{demo.findings}</p>
        
        <h4>Artifacts</h4>
        <div class="artifact-grid">
            {artifacts_html}
        </div>
    </section>
    """

def generate_report_html(demos_html: str, toc_html: str) -> str:
    """Generates the final HTML content with CSS and JS for interactivity."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{REPORT_TITLE}</title>
    <style>
        :root {{
            --bg-color: #ffffff; --text-color: #333333; --container-bg: #f8f9fa;
            --header-color: #2c3e50; --border-color: #e9ecef; --link-color: #007bff;
            --code-bg: #e9ecef; --artifact-bg: #ffffff; --svg-fill: #333; --svg-stroke: #555;
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-color: #121212; --text-color: #e0e0e0; --container-bg: #1e1e1e;
                --header-color: #e0e0e0; --border-color: #333333; --link-color: #bb86fc;
                --code-bg: #2c2c2c; --artifact-bg: #252525; --svg-fill: #e0e0e0; --svg-stroke: #c0c0c0;
            }}
            .artifact-container .raster-image {{ filter: brightness(.8) contrast(1.2); }}
        }}
        body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.7; margin: 0; padding: 20px; background-color: var(--bg-color); color: var(--text-color);
        }}
        .container {{ max-width: 1400px; margin: 20px auto; display: grid; grid-template-columns: 250px 1fr; gap: 40px; }}
        #table-of-contents {{ position: sticky; top: 20px; align-self: start; }}
        main {{ min-width: 0; }}
        h1, h2, h3, h4 {{ color: var(--header-color); }}
        h1 {{ font-size: 2.5em; text-align: center; border: none; grid-column: 1 / -1; margin-bottom: 0; }}
        h2 {{ font-size: 2em; border-bottom: 2px solid var(--link-color); padding-bottom: 10px; }}
        h4 {{ font-size: 1.1em; margin-top: 25px; border-bottom: 1px dashed #ced4da; }}
        code {{ background: var(--code-bg); padding: .2em .4em; border-radius: 4px; }}
        ul {{ padding-left: 20px; }}
        .demo-section {{ margin-bottom: 50px; padding: 20px; border: 1px solid var(--border-color); border-radius: 8px; background: var(--container-bg); }}
        .artifact-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .artifact-container {{ border: 1px solid var(--border-color); border-radius: 8px; padding: 15px; background: var(--artifact-bg); text-align: center; display: flex; flex-direction: column; }}
        .artifact-container svg {{ max-width: 100%; height: auto; border-radius: 4px; cursor: pointer; --fill-color: var(--svg-fill); --stroke-color: var(--svg-stroke); }}
        .artifact-container footer {{ font-size: 0.9em; color: #888; margin-top: 10px; }}
        .text-artifact pre {{ background-color: var(--code-bg); padding: 10px; border-radius: 4px; text-align: left; white-space: pre-wrap; word-wrap: break-word; }}
        .modal {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.85); align-items: center; justify-content: center; }}
        .modal.is-open {{ display: flex; }}
        .modal-content {{ margin: auto; padding: 20px; max-width: 90%; max-height: 90%; }}
        .modal-content svg {{ width: 100%; height: 100%; object-fit: contain; }}
        .close {{ position: absolute; top: 20px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; cursor: pointer; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{REPORT_TITLE}</h1>
        <nav id="table-of-contents">
            <h3>Table of Contents</h3>
            <ul>{toc_html}</ul>
        </nav>
        <main>
            <p>This report showcases the capabilities of the Kéya engine through a series of automated demonstrations.</p>
            {demos_html}
        </main>
    </div>
    <div id="myModal" class="modal"><span class="close">&times;</span><div class="modal-content" id="modal-content-host"></div></div>
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
            modal.addEventListener('click', (e) => {{ if (e.target === modal) close(); }});
        }});
    </script>
</body>
</html>
"""

def generate_report():
    """Generates the full HTML report from the demo registry."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear old artifacts before running demos
    for item in OUTPUT_DIR.iterdir():
        if item.is_file():
            os.remove(item)

    discover_and_load_demos()
    
    # Run demos to generate artifacts
    for demo in DEMO_REGISTRY:
        if demo.func:
            print(f"Running demo: {demo.title}...")
            # Set the current working directory to the output directory for the demo
            os.chdir(OUTPUT_DIR)
            demo.func()
            # Change back to the original directory
            os.chdir(Path(__file__).parent.parent.parent)

    if not DEMO_REGISTRY:
        print("No demos were found. Report will be empty.")
        return
        
    toc_html = "".join(f'<li><a href="#{d.title.replace(" ", "-").lower()}">{d.title}</a></li>' for d in DEMO_REGISTRY)
    demos_html = "".join(generate_demo_section_html(d) for d in DEMO_REGISTRY)
    html_content = generate_report_html(demos_html, toc_html)
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Report generated at {REPORT_FILE.resolve()}")

if __name__ == "__main__":
    generate_report()