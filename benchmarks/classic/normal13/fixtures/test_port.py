from parse_port import parse_port
for value,expected in [("1",1),("65535",65535),(" 0080 ",80)]:
    assert parse_port(value)==expected
for value in ["", "0", "65536", "-1", "+80", "１２", "٨٠", 80, None, True]:
    try:parse_port(value)
    except ValueError:pass
    else:raise AssertionError("Expected ValueError for "+repr(value))
print("ALL_TESTS_PASSED")
