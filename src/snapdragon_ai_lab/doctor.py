"""Read hardware capabilities without loading models or changing the system."""
import os
import platform
from pathlib import Path
from . import power
from .platform_support import detect

def inspect():
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('MemTotal:','MemAvailable:')):
            memory[line.split(':',1)[0]]=int(line.split()[1])*1024
    driver_names=set()
    for driver in Path('/sys/class/drm').glob('card*/device/driver'):
        try:driver_names.add(driver.resolve().name)
        except OSError:pass
    npu=Path('/dev/fastrpc-cdsp')
    heap=Path('/dev/dma_heap/system')
    return {'schema_version':1,'architecture':platform.machine(),'logical_cpus':os.cpu_count(),
            'memory_bytes':memory,'drm_driver_names':sorted(driver_names),
            'hexagon_device_present':npu.exists(),
            'hexagon_device_accessible':os.access(npu,os.R_OK|os.W_OK),
            'system_dma_heap_present':heap.exists(),
            'system_dma_heap_readable':os.access(heap,os.R_OK),
            'power':power.snapshot(),'platform_support':detect(),'heavy_work_started':False}
