import os

# (def save_report(report_name: str, content: str):
#    """
#    Guarda un reporte en carpeta reporting
#    """
#    path = f"data/08_reporting/{report_name}.txt"
#   os.makedirs(os.path.dirname(path), exist_ok=True)
#
#    with open(path, "w") as f:
#        f.write(content)
# 
def save_report(data, filename="report.txt"):
    """
    Guarda un reporte en un archivo de texto.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(str(data))




---
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def generate_report(data: Any, metadata: Dict[str, Any] | None = None) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = []
    report_lines.append("=== REPORTE DE EJECUCIÓN ===")
    report_lines.append(f"Fecha: {timestamp}")
    report_lines.append("")

    if metadata:
        report_lines.append("=== METADATA ===")
        for key, value in metadata.items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("")

    report_lines.append("=== DATA ===")
    report_lines.append(str(data))
    report_lines.append("")

    return "\n".join(report_lines)


def save_report(report: str, filepath: str = "data/08_reporting/report.txt") -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write(report)