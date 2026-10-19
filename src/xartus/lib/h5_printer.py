# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause


def print_item(item, offset="", items=True) -> None:
    name = f"{item.name.split('/')[-1]}/:" if hasattr(item, "name") else ""

    if "keys" in dir(item):
        offset_inc = "  |"
        prefix = "\--"
    else:
        offset_inc = "   "
        prefix = "+--"

    if items:
        print(f"{offset[:-1]}{prefix} {name} {str(item)}")
    else:
        print(f"{offset[:-1]}{prefix} {name}")

    for at in item.attrs:
        print(f"{offset}{offset_inc} @{at}: {item.attrs[at]}")


def print_group(d, offset="", max_depth=-1, items=True) -> None:
    if max_depth == 0:
        return
    offset_inc = "  |"
    print_item(d, offset, items)
    if "keys" not in dir(d):
        return
    mx = len(d.keys())
    if mx < 10:
        for k in d:
            print_group(
                d[k], offset=offset + offset_inc, max_depth=max_depth - 1, items=items
            )

    else:
        for ii in range(5):
            k = d.keys()[ii]

            print_group(
                d[k], offset=offset + offset_inc, max_depth=max_depth - 1, items=items
            )
        print(offset + "...")
        for ii in range(-5, 0):
            k = d.keys()[ii]

            print_group(
                d[k], offset=offset + offset_inc, max_depth=max_depth - 1, items=items
            )
