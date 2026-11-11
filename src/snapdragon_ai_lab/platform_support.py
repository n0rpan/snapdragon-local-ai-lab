"""Read public SoC compatibility identifiers; never probe accelerators."""
from pathlib import Path
import platform
import re
import shutil

def detect(architecture=None,compatible_path=Path('/sys/firmware/devicetree/base/compatible')):
    architecture=platform.machine() if architecture is None else architecture
    try:values=Path(compatible_path).read_bytes()[:4096].decode(errors='replace').split('\0')
    except OSError:values=[]
    socs=sorted({value for value in values if re.fullmatch(r'qcom,x[12][ep][0-9]+',value)})
    return {'architecture':architecture,'snapdragon_x_detected':architecture=='aarch64' and bool(socs),
            'soc_identifiers':socs,'omarchy_command_present':shutil.which('omarchy') is not None,
            'validation_scope':'Yoga Slim 7x tested; other Snapdragon X devices need community verification'}
