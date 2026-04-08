"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html."""

from .hooks import PipelineNameHook

# Register the hook to ensure pipeline_name is always present
HOOKS = (PipelineNameHook(),)