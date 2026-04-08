import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file into Python dictionary.

    Reads and parses YAML configuration files used for Prefect orchestration settings,
    project metadata, and environment-specific parameters. Uses safe_load to prevent
    arbitrary code execution from untrusted YAML.

    Args:
        config_path: Absolute or relative path to YAML configuration file.
            Common paths:
            - 'configs/project_config.yaml': Prefect orchestration config
            - 'conf/base/parameters.yml': Kedro pipeline parameters
            - 'conf/local/credentials.yml': Sensitive credentials (gitignored)

    Returns:
        Dictionary representation of YAML configuration with nested structure
        preserved. All YAML types (strings, numbers, lists, mappings) converted
        to Python equivalents.

    Raises:
        FileNotFoundError: If config_path does not exist
        yaml.YAMLError: If file contains invalid YAML syntax
        PermissionError: If file cannot be read due to permissions

    Note:
        Uses yaml.safe_load() which only constructs Python primitive types.
        Does not execute arbitrary Python code (unlike yaml.load).

    Example:
        >>> config = load_config('configs/project_config.yaml')
        >>> print(config['project']['name'])
        'Predictive CLTV Insights'
        >>> print(config['logging']['level'])
        'INFO'
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
