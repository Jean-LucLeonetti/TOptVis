#!/bin/bash

# Finder may start this script outside the checkout.
cd -- "$(dirname -- "$0")" || exit 1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

pause_on_error() {
    if [[ -t 0 ]]; then
        read -r -p "Press Return to close this window. "
    fi
}

if [[ ! -x ".venv/bin/python" ]]; then
    echo "The local Python environment is missing."
    echo "See 'Local macOS environment' in README.md to set up .venv first."
    pause_on_error
    exit 1
fi

echo "Starting TOptVis. Keep this window open while using the app."
echo "Press Control-C to stop the server."
".venv/bin/python" -m ml_opt.web --open-browser "$@"
status=$?
if [[ $status -ne 0 ]]; then
    echo
    echo "TOptVis could not start. See the error above."
    pause_on_error
fi
exit "$status"
