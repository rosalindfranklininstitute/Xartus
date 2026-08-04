# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
class NoDataError(Exception):
    pass


class InnerDataNotContainedError(Exception):
    pass


class UnsupportedDataError(RuntimeError):
    pass
