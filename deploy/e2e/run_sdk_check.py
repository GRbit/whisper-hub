"""Runs internal/openaiapi/cmd/fakeserver/sdk_check.py, unmodified, against
the real gateway, one top-level statement at a time.

The script stops at its first failure, and a few of its checks assert
behaviour of the canned fake service that the real stack legitimately does
differently (listed in EXPECTED with the reason). Running statement by
statement lets every other check still run; the result fails on any
unexpected failure and on any expected failure that unexpectedly passes,
so the list cannot rot.

    python run_sdk_check.py sdk_check.py BASE_URL AUDIO
"""

import ast
import sys

# Statement source prefix -> why the real stack differs from the fake.
EXPECTED = {
    'check("stream mid error raises APIError"':
        "hub maps a worker's in-stream 'unavailable' to its own client-safe text, not the worker's",
    'check("models.list"':
        "the fake worker also serves the failure models busy/boom/midfail, which the list shows",
    'expect("unauthenticated"':
        "a model named 'unauth' is a fake-service trick; a valid key cannot get a 401 per model",
}


def main():
    script, base, audio = sys.argv[1], sys.argv[2], sys.argv[3]
    src = open(script).read()
    tree = ast.parse(src)
    sys.argv = [script, base, audio]
    ns = {"__name__": "sdk_check"}
    unexpected, fixed = [], []
    for node in tree.body:
        seg = ast.get_source_segment(src, node)
        mod = ast.Module(body=[node], type_ignores=[])
        reason = next((r for p, r in EXPECTED.items() if seg.lstrip().startswith(p) or p in seg.split("\n")[-1]), None)
        try:
            exec(compile(mod, script, "exec"), ns)
            ok, err = True, None
        except SystemExit as e:
            ok, err = False, str(e)
        except Exception as e:  # noqa: BLE001 - report and keep going
            ok, err = False, repr(e)
        if ok and reason:
            fixed.append(seg.split("\n")[0])
        if not ok:
            first = seg.split("\n")[0]
            if reason:
                print(f"expected FAIL: {first}\n    {err}\n    why: {reason}")
            else:
                print(f"FAIL: {first}\n    {err}")
                unexpected.append(first)
    for f in fixed:
        print(f"expected failure now passes, update EXPECTED: {f}")
    print(f"{len(unexpected)} unexpected failure(s)")
    sys.exit(1 if unexpected or fixed else 0)


if __name__ == "__main__":
    main()
