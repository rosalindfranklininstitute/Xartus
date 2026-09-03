@ECHO OFF

REM SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
REM 
REM SPDX-License-Identifier: LicenseRef-Sphinx

set /p PUBLISH_TOKEN=<publish_token.txt
echo %PUBLISH_TOKEN%
uv publish --token %PUBLISH_TOKEN%
