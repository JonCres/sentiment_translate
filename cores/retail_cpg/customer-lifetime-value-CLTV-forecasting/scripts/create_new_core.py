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

    # Update README with project name
    readme_path = new_core_dir / "README.md"
    if readme_path.exists():
        with open(readme_path, "r") as f:
            readme = f.read()

        readme = readme.replace("AI Core Template", f"{name.replace('-', ' ').title()}")
        readme = readme.replace("ai-core-template", name)

        with open(readme_path, "w") as f:
            f.write(readme)

    # Kedro Adaptation: Rename package and update configs
    safe_name = name.replace("-", "_")

    # 1. Rename src/aicore_template
    src_dir = new_core_dir / "src"
    old_pkg_dir = src_dir / "aicore_template"
    new_pkg_dir = src_dir / safe_name

    if old_pkg_dir.exists():
        old_pkg_dir.rename(new_pkg_dir)
        print(f"✅ Renamed Kedro package to {safe_name}")

        # 2. Update pyproject.toml
        pyproject_path = new_core_dir / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "r") as f:
                content = f.read()

            content = content.replace(
                'package_name = "aicore_template"', f'package_name = "{safe_name}"'
            )
            content = content.replace(
                'aicore_name = "AI Core Template"', f'aicore_name = "{name}"'
            )

            with open(pyproject_path, "w") as f:
                f.write(content)

        # 3. Update pipeline_registry.py
        registry_path = new_pkg_dir / "pipeline_registry.py"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                content = f.read()

            content = content.replace(
                "from aicore_template.pipelines", f"from {safe_name}.pipelines"
            )

            with open(registry_path, "w") as f:
                f.write(content)

        # 4. Update logging.yml
        logging_path = new_core_dir / "conf/base/logging.yml"
        if logging_path.exists():
            with open(logging_path, "r") as f:
                content = f.read()

            content = content.replace("aicore_template", safe_name)

            with open(logging_path, "w") as f:
                f.write(content)

    print(f"\n✅ AI Core '{name}' created successfully!")
    print(f"\n📋 Next steps:")
    print(f"   cd {name}")
    print(f"   python -m venv venv")
    print(f"   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print(f"   pip install -r requirements.txt")
    print(f"   cp .env.example .env  # Edit with your values")
    print(f"   prefect server start  # In separate terminal")
    print(f"   python -m src.prefect_orchestration.data_pipeline")
    print(f"\n📚 Documentation: docs/getting_started.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new AI Core from template")
    parser.add_argument(
        "--name", required=True, help="Name of the new AI Core (e.g., fraud-detection)"
    )
    parser.add_argument(
        "--industry",
        required=True,
        choices=[
            "retail",
            "healthcare",
            "financial services & banking",
            "media&entertainment",
            "marketing",
            "travel&hospitality",
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
