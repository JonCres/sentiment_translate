#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path
import yaml


def create_ai_core(name: str, industry: str, output_dir: str = "."):
    """Create new AI Core from template"""

    # Define paths
    template_dir = Path(__file__).parent.parent
    new_core_dir = Path(output_dir) / name

    print(f"🚀 Creating AI Core: {name}")
    print(f"📁 Industry: {industry}")
    print(f"📂 Output directory: {new_core_dir}")

    # Create new directory
    new_core_dir.mkdir(parents=True, exist_ok=True)

    # Copy entire template structure
    for item in template_dir.iterdir():
        if item.name in [".git", "__pycache__", ".pytest_cache", "venv", ".env"]:
            continue

        dest = new_core_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    # Copy industry-specific config if it exists, otherwise copy the default
    industry_config_src = template_dir / f"configs/examples/{industry}_config.yaml"
    default_config_src = template_dir / "configs/project_config.yaml"
    industry_config_dest = new_core_dir / "configs/project_config.yaml"

    # Copy industry-specific config if available, otherwise use default
    if industry_config_src.exists():
        shutil.copy2(industry_config_src, industry_config_dest)
        print(f"✅ Copied {industry} configuration")
    elif default_config_src.exists():
        shutil.copy2(default_config_src, industry_config_dest)
        print(f"✅ Copied default configuration (industry-specific not found)")
    else:
        print(f"⚠️  No configuration file found")

    # Update config with new name
    if industry_config_dest.exists():
        with open(industry_config_dest, "r") as f:
            config = yaml.safe_load(f)

        # Update project name in config
        if "project" in config:
            config["project"]["name"] = name

        with open(industry_config_dest, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"✅ Updated project name in configuration")

    # Create .env from example
    env_example = new_core_dir / ".env.example"
    env_file = new_core_dir / ".env"
    if env_example.exists():
        shutil.copy2(env_example, env_file)
        print(f"✅ Created .env file")

    # Create standard data layers
    data_dir = new_core_dir / "data"
    data_layers = [
        "01_raw",
        "02_intermediate",
        "03_primary",
        "04_feature",
        "05_model_input",
        "06_models",
        "07_model_output",
        "08_reporting",
    ]

    for layer in data_layers:
        (data_dir / layer).mkdir(parents=True, exist_ok=True)
        # Create .gitkeep to ensure git tracks these empty directories
        (data_dir / layer / ".gitkeep").touch()

    print(f"✅ Created standard data layers")

    # Update README with project name using the template
    readme_template_path = new_core_dir / "ai_core_README_template.md"
    readme_dest_path = new_core_dir / "README.md"

    if readme_template_path.exists():
        with open(readme_template_path, "r") as f:
            readme_content = f.read()

        # Replace basic placeholders
        readme_content = readme_content.replace(
            "AI Core Template", f"{name.replace('-', ' ').title()}"
        )
        readme_content = readme_content.replace("ai-core-template", name)

        # Write to README.md (overwrites instructional README if any)
        with open(readme_dest_path, "w") as f:
            f.write(readme_content)

        # Remove the template file from the new core
        readme_template_path.unlink()
        print(f"✅ Created README.md from template")

    # 1. Rename src/aicore
    src_dir = new_core_dir / "src"
    old_pkg_dir = src_dir / "aicore"
    new_pkg_dir = src_dir / "ai_core"

    if old_pkg_dir.exists():
        old_pkg_dir.rename(new_pkg_dir)
        print(f"✅ Renamed Kedro package from aicore to ai_core")

    # 2. Update pyproject.toml
    pyproject_path = new_core_dir / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "r") as f:
            content = f.read()

        # [project] replacements
        content = content.replace('name = "aicore-kedro-template"', f'name = "{name}"')
        content = content.replace(
            'description = "AI Core Template"',
            f'description = "{name.replace("-", " ").title()} AI Core"',
        )

        # [tool.kedro] replacements
        content = content.replace('package_name = "aicore"', 'package_name = "ai_core"')
        content = content.replace(
            'project_name = "AI Core Template"', f'project_name = "{name.replace("-", " ").title()}"'
        )

        # Path replacements
        content = content.replace("src/aicore", f"src/ai_core")
        content = content.replace("--cov src/aicore", "--cov src/ai_core")

        with open(pyproject_path, "w") as f:
            f.write(content)
        print("✅ Updated pyproject.toml")

    # 3. Update setup.py
    setup_path = new_core_dir / "setup.py"
    if setup_path.exists():
        with open(setup_path, "r") as f:
            content = f.read()

        content = content.replace('name="ai-core-template"', f'name="{name}"')
        content = content.replace(
            'description="Reusable AI Core Template with Prefect"',
            f'description="{name.replace("-", " ").title()} AI Core"',
        )

        with open(setup_path, "w") as f:
            f.write(content)
        print("✅ Updated setup.py")

    # 4. Update pipeline_registry.py
    registry_path = new_pkg_dir / "pipeline_registry.py"
    if registry_path.exists():
        with open(registry_path, "r") as f:
            content = f.read()

        content = content.replace("from aicore.pipelines", f"from ai_core.pipelines")

        with open(registry_path, "w") as f:
            f.write(content)
        print("✅ Updated package imports")

    # 4. Update logging.yml
    logging_path = new_core_dir / "conf/base/logging.yml"
    if logging_path.exists():
        with open(logging_path, "r") as f:
            content = f.read()

        content = content.replace("aicore_template", "ai_core")
        content = content.replace("aicore", "ai_core")

        with open(logging_path, "w") as f:
            f.write(content)
        print("✅ Updated logging configuration")

    # 5. Update feature_store.yaml
    feature_store_path = new_core_dir / "feature_repo/feature_store.yaml"
    if feature_store_path.exists():
        with open(feature_store_path, "r") as f:
            content = f.read()

        content = content.replace(
            "project: ai_core_template", f"project: {name.replace('-', '_')}"
        )

        with open(feature_store_path, "w") as f:
            f.write(content)
        print("✅ Updated feature store configuration")

    # 6. Generic replacement of aicore -> ai_core in src directory
    for py_file in (new_core_dir / "src").rglob("*.py"):
        with open(py_file, "r") as f:
            content = f.read()
        if "aicore" in content:
            content = content.replace("aicore", "ai_core")
            with open(py_file, "w") as f:
                f.write(content)
    print("✅ Standardized package naming across all source files")

    # Confirm MLflow tracking utilities are present
    mlflow_tracking = new_core_dir / "src/utils/mlflow_tracking.py"
    mlflow_config = new_core_dir / "conf/local/mlflow.yml"
    if mlflow_tracking.exists() and mlflow_config.exists():
        print("✅ MLflow tracking utilities configured")

    print(f"\n✅ AI Core '{name}' created successfully!")
    print(f"\n📋 Next steps:")
    print(f"   cd {name}")
    print("   uv sync")
    print("   source .venv/bin/activate")
    print("   cp .env.example .env  # Edit with your values")
    print("\n   # Start MLflow Server (for experiment tracking)")
    print("   mlflow server --host 127.0.0.1 --port 5000")
    print("\n   # Start Prefect Server")
    print("   uv run prefect server start")
    print("   # Run Pipeline")
    print("   uv run python -m src.prefect_orchestration.data_pipeline")
    print("\n📚 Documentation: docs/ai_product_canvas.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new AI Core from template")
    parser.add_argument(
        "--name", required=True, help="Name of the new AI Core (e.g., fraud-detection)"
    )
    parser.add_argument(
        "--industry",
        choices=[
            "retail",
            "healthcare",
            "financial services & banking",
            "media&entertainment",
            "marketing",
            "travel&hospitality",
            "retail_cpg",
        ],
        help="Industry vertical",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory (default: current directory)",
    )

    args = parser.parse_args()
    create_ai_core(args.name, args.industry, args.output_dir)
