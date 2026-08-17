# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause

from ..api import nexus_slice


def nxslice() -> None:
    partial_args = nexus_slice.ProcessArgs.parse_config("nxslice")
    process_args = nexus_slice.ProcessArgs.parse_interactive(
        "nxslice",
        exclude=["config"],
        args=partial_args.remaining_args,
    )

    nexus_slice.process(process_args, {})
