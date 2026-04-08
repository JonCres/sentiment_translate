"""ai_core file for ensuring the package is executable
as `ai_core` and `python -m ai_core`
"""
import sys
from pathlib import Path
from typing import Any

from kedro.framework.cli.utils import find_run_command
# from ai_core.hooks import ProjectHooks
from kedro.framework.project import configure_project


def main(*args, **kwargs) -> Any:
    package_name = Path(__file__).parent.name
    configure_project(package_name)
    run = find_run_command(package_name)
    return run(*args, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
