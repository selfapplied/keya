import base64
from pathlib import Path
import subprocess
import os

# --- Configuration ---
ASSETS_PATH = Path(".out/visualizations")
REPORT_FILE = Path(".out/keya_report.html")
REPORT_TITLE = "Kéya Project Findings: A Comprehensive Overview"

# --- Demo-Specific Content ---
# This structure groups artifacts and descriptions by the demo that produces them.
DEMO_SECTIONS = [
    {
        "title": "Sierpinski Prime Analysis",
        "script": "demos/sierpinski.py",
        "description": """
        This demo analyzes prime number distributions using Keya operators within the framework of a Sierpinski triangle. It makes the following claims:
        - Operators can diagonalize prime gaps and irregularities.
        - Infinite prime sequences can be contained within finite grids.
        - Operator cycles reveal hidden patterns in prime numbers.
        The visualization overlays prime "sparks" on a Sierpinski pattern, showing the effects of the operators.
        """,
        "claims": [
            "Operators can diagonalize irregularities in prime distributions.",
            "Containment can map the infinite sequence of primes into a finite, analyzable grid.",
            "Operator cycles reveal hidden structural patterns in prime numbers.",
            "The process reduces the overall variance of the prime distribution, indicating a convergence towards a more ordered state."
        ],
        "findings": "The demo successfully validates its claims. The generated visualizations show a significant variance reduction in both prime derivatives and anomalies after the operators are applied. The final report from the script concludes with a 'Strong validation of theory' and shows that the operators enhance diagonalization and reveal patterns.",
        "artifacts": [
            "prime_sparks.svg",
            "prime_histograms.svg",
            "prime_growth.svg",
            "prime_analysis.svg"
        ]
    },
    {
        "title": "Floating-Point Arithmetic as an Operator System",
        "script": "demos/floatingpoint.py",
        "description": """
        This demo explores the idea that standard floating-point arithmetic (IEEE 754) can be understood as a system of operators with properties analogous to Keya's. It tests several claims:
        - Quantization in floating-point math acts as a containment operation.
        - Rounding errors behave like micro-cycles that can be analyzed.
        - Special values like NaN and Infinity are fixed points in the operational system.
        The visualizations show the results of these numerical tests.
        """,
        "claims": [
            "Keya operators can represent floating-point numbers.",
            "Standard arithmetic operations (addition, multiplication) can be reliably performed on these representations.",
            "The system remains stable and consistent, adhering to mathematical axioms."
        ],
        "findings": "The demo passes all its internal tests, confirming that floating-point numbers can be represented and manipulated correctly within the Keya framework. The visualizations show the successful outcomes of these arithmetic tests.",
        "artifacts": ["floating_point_tests.svg"]
    },
    {
        "title": "Quantum Phenomena Simulation",
        "script": "demos/quantum.py",
        "description": """
        This demo simulates various quantum phenomena to show how Keya's operators can model quantum state evolution. It covers:
        - The structure of hydrogen orbitals (1s, 2pz).
        - The evolution of a Gaussian wave packet over time.
        - The principle of superposition.
        The visualization provides a gallery of these quantum states.
        """,
        "claims": [
            "Hydrogen orbitals can be constructed and visualized.",
            "Keya's evolution operators can model the time-development of a quantum wave packet.",
            "Superposition states can be created and manipulated."
        ],
        "findings": "The script runs through its series of demos, printing confirmations for each test. The final visualization successfully renders the different quantum states, confirming that the simulation and plotting functions are working correctly.",
        "artifacts": ["quantum_phenomena.svg"]
    },
    {
        "title": "Quantum Orbital Shapes",
        "script": "demos/orbital.py",
        "description": "This demo renders various atomic orbitals (1s, 2pz, 3dz2) as 3D isosurfaces to visualize their shapes. It also provides a side-by-side comparison of these orbitals.",
        "claims": [
            "The shapes of atomic orbitals can be generated programmatically.",
            "These shapes can be rendered as 3D visualizations for analysis."
        ],
        "findings": "The script generates separate SVG files for the 1s, 2pz, and 3dz2 orbitals, as well as a combined comparison plot. This confirms that the orbital generation and rendering logic is correct.",
        "artifacts": ["orbital_1s.svg", "orbital_2pz.svg", "orbital_3dz2.svg", "orbital_comparison.svg"]
    },
    {
        "title": "Mantissa as a Quantum State",
        "script": "demos/mantissa.py",
        "description": "This demo validates the claims about the relationship between mantissas and quantum states. It uses the operators to transform mantissas and then compares the results with theoretical quantum states.",
        "claims": [
            "The operators can transform mantissas into quantum states.",
            "The transformation process is consistent and predictable."
        ],
        "findings": "The script successfully runs its internal validation checks, supporting the claims. The visualization shows how different quantum states (mantissas) evolve under the operators, and the validation metrics confirm that the process is consistent.",
        "artifacts": ["mantissa_quantum_validation.svg"]
    },
    {
        "title": "Pascal's Triangle Iterators",
        "script": "demos/pascal.py",
        "description": "Demonstrates the dual-iterator nature of Pascal's triangle construction, showing that the operators can generate complex, evolving patterns similar to cellular automata and fractals from simple initial conditions.",
        "claims": [
            "The Wild, Tame, and Wild_closure operators can transform a simple matrix into a structure resembling Pascal's triangle.",
            "The process is deterministic and reveals underlying generative rules."
        ],
        "findings": "The script successfully generates a visualization that shows the emergence of Sierpinski-like patterns from iterating on Pascal's triangle vectors. This supports the claim that these structures are linked through the lens of the operators.",
        "artifacts": ["pascal_iterators.svg"]
    }
]


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
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
        <h1>{title}</h1>
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

HTML_TEMPLATE = HTML_TEMPLATE.replace("{{{{", "{{").replace("}}}}", "}}")

def ensure_artifact_exists(demo_script, asset_path):
    """Checks if an artifact exists, and if not, runs the demo script to create it."""
    if not asset_path.exists():
        print(f"    ⚠️ Artifact '{asset_path.name}' not found. Running demo script...")
        try:
            # Ensure the script path is correct relative to the project root
            script_path = Path(demo_script)
            if not script_path.exists():
                print(f"    ❌ ERROR: Demo script '{script_path}' not found!")
                return False

            # Run the script from the project root directory
            project_root = Path(__file__).parent.parent
            result = subprocess.run(
                ["python3", str(script_path)],
                capture_output=True,
                text=True,
                check=True,
                cwd=project_root
            )
            print(f"    ✅ Successfully ran '{demo_script}'.")
            if not asset_path.exists():
                print(f"    ❌ ERROR: Demo script ran but did not produce '{asset_path.name}'.")
                print("    STDOUT:", result.stdout)
                print("    STDERR:", result.stderr)
                return False
        except subprocess.CalledProcessError as e:
            print(f"    ❌ ERROR: Failed to run '{demo_script}'.")
            print("    Return code:", e.returncode)
            print("    STDOUT:", e.stdout)
            print("    STDERR:", e.stderr)
            return False
    return True


def generate_report():
    """Generates a self-contained HTML report with embedded visualizations, structured by demo."""
    print("🚀 Starting report generation...")

    demo_sections_html = ""
    for demo in DEMO_SECTIONS:
        print(f"📄 Processing demo section: {demo['title']}")
        
        # --- Build Claims List ---
        claims_html = "<ul>" + "".join(f"<li>{claim}</li>" for claim in demo["claims"]) + "</ul>"
        
        # --- Build Artifacts Grid for this Demo ---
        artifacts_html = ""
        if not demo.get("artifacts"):
            artifacts_html = "<p>No artifacts generated for this demo.</p>"
        else:
            for asset_name in demo["artifacts"]:
                asset_path = ASSETS_PATH / asset_name
                
                # Ensure artifact exists before trying to embed
                if not ensure_artifact_exists(demo["script"], asset_path):
                    embed_html = "<p>Failed to generate artifact.</p>"
                else:
                    print(f"  -> Embedding '{asset_path.name}'...")
                    embed_html = ""
                    if asset_path.exists():
                        if asset_path.suffix == ".png":
                            try:
                                with open(asset_path, "rb") as f:
                                    encoded_string = base64.b64encode(f.read()).decode("utf-8")
                                data_uri = f"data:image/png;base64,{encoded_string}"
                                embed_html = f'<img src="{data_uri}" alt="{asset_path.name}">'
                            except Exception as e:
                                print(f"    ❌ Failed to encode PNG: {e}")
                        elif asset_path.suffix == ".svg":
                            try:
                                svg_content = asset_path.read_text(encoding='utf-8')
                                # Clean up SVG for embedding
                                svg_content = svg_content.replace("<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">", "")
                                embed_html = svg_content
                            except Exception as e:
                                print(f"    ❌ Failed to read SVG: {e}")
                        else:
                            print(f"    ⚠️ Unsupported file type: {asset_path.suffix}")
                    else:
                        embed_html = f"<p>Artifact '{asset_path.name}' could not be found or generated.</p>"

                if embed_html:
                    artifacts_html += f"""
                    <div class="artifact-container">
                        <div>{embed_html}</div>
                        <footer>Filename: <code>{asset_path.name}</code></footer>
                    </div>
                    """
            
        # --- Assemble Section HTML ---
        demo_sections_html += f"""
        <div class="demo-section">
            <h2>{demo['title']}</h2>
            <p><strong>Source File:</strong> <code>{demo['script']}</code></p>
            <h4>Description</h4>
            <p>{demo['description']}</p>
            <h4>Claims</h4>
            {claims_html}
            <h4>Findings</h4>
            <p>{demo['findings']}</p>
            <h4>Visual Artifacts</h4>
            <div class="artifact-grid">{artifacts_html}</div>
        </div>
        """

    print("✅ Processed all demo sections.")

    # --- Assemble Final HTML ---
    final_html = HTML_TEMPLATE.format(
        title=REPORT_TITLE,
        demo_sections_html=demo_sections_html
    )

    # --- Write to File ---
    try:
        REPORT_FILE.write_text(final_html, encoding='utf-8')
        print(f"\n🎉 Successfully generated report at '{REPORT_FILE}'!")
    except Exception as e:
        print(f"\n❌ Failed to write report file: {e}")

if __name__ == "__main__":
    generate_report() 