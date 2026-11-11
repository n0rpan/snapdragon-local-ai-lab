def parse_port(value):
    if not isinstance(value,str):
        raise ValueError("string required")
    value=value.strip()
    if not value.isdigit():
        raise ValueError("digits required")
    port=int(value)
    if not 1<=port<=65535:
        raise ValueError("range")
    return port
