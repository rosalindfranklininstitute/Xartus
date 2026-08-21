# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause


def print_item(item, offset="") -> None:
    name = f"{item.name}:" if hasattr(item, "name") else ""
    parent = f" ({item.parent})" if hasattr(item, "parent") else ""

    print(f"{offset}> {name} {str(item)}{parent}")
    for at in item.attrs:
        print(f"{offset}| - @{at}: {item.attrs[at]}")


def print_group(d, offset="", max_depth=-1) -> None:
    if max_depth == 0:
        return
    print_item(d, offset)
    if "keys" not in dir(d):
        return
    mx = len(d.keys())
    if mx < 10:
        for k in d:
            print_group(d[k], offset=offset + "| ", max_depth=max_depth - 1)

    else:
        for ii in range(5):
            k = d.keys()[ii]

            print_group(d[k], offset=offset + "| ", max_depth=max_depth - 1)
        print(offset + "...")
        for ii in range(-5, 0):
            k = d.keys()[ii]

            print_group(d[k], offset=offset + "| ", max_depth=max_depth - 1)
