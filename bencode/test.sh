#!/usr/bin/env bash
VALIDATOR="${1:-./validator}"
PASS=0
FAIL=0

expect() {
  local expected="$1" label="$2"
  shift 2
  printf '%s' "$@" | "$VALIDATOR" 2>/dev/null && rc=0 || rc=$?
  if { [ "$expected" = "valid" ] && [ "$rc" -eq 0 ]; } ||
     { [ "$expected" = "invalid" ] && [ "$rc" -ne 0 ]; }; then
    PASS=$((PASS + 1))
  else
    echo "FAIL: $label (exit=$rc, expected $expected)"
    FAIL=$((FAIL + 1))
  fi
}

expect_hex() {
  local expected="$1" label="$2" hex="$3"
  printf '%b' "$(echo "$hex" | sed 's/../\\x&/g')" | "$VALIDATOR" 2>/dev/null && rc=0 || rc=$?
  if { [ "$expected" = "valid" ] && [ "$rc" -eq 0 ]; } ||
     { [ "$expected" = "invalid" ] && [ "$rc" -ne 0 ]; }; then
    PASS=$((PASS + 1))
  else
    echo "FAIL: $label (exit=$rc, expected $expected)"
    FAIL=$((FAIL + 1))
  fi
}

# ── Strings ──────────────────────────────────────────────
expect valid   "string: basic"                "4:spam"
expect valid   "string: empty"                "0:"
expect valid   "string: single char"          "1:a"
expect valid   "string: 10 chars"             "10:0123456789"
expect invalid "string: truncated"            "4:spa"
expect invalid "string: missing colon"        "4spam"
expect invalid "string: missing length"       ":spam"
expect invalid "string: data too short"       "4:"
expect invalid "string: negative length"      "-1:a"
expect invalid "string: leading zero length"  "01:a"

# ── Integers ─────────────────────────────────────────────
expect valid   "int: positive"                "i3e"
expect valid   "int: negative"                "i-3e"
expect valid   "int: zero"                    "i0e"
expect valid   "int: large positive"          "i12345e"
expect valid   "int: large negative"          "i-12345e"
expect valid   "int: very large"              "i99999999999999999999999999999999999999999999999e"
expect invalid "int: negative zero"           "i-0e"
expect invalid "int: leading zero"            "i03e"
expect invalid "int: multiple leading zeros"  "i003e"
expect invalid "int: negative leading zero"   "i-03e"
expect invalid "int: empty"                   "ie"
expect invalid "int: just minus"              "i-e"
expect invalid "int: missing e"               "i3"
expect invalid "int: non-digit"               "iae"
expect invalid "int: plus sign"               "i+3e"
expect invalid "int: space"                   "i 3e"

# ── Lists ────────────────────────────────────────────────
expect valid   "list: empty"                  "le"
expect valid   "list: two strings"            "l4:spam4:eggse"
expect valid   "list: three ints"             "li1ei2ei3ee"
expect valid   "list: nested"                 "ll4:spamee"
expect valid   "list: 4-deep nesting"         "lllleeee"
expect invalid "list: unclosed"               "l"
expect invalid "list: unclosed with content"  "l4:spam"
expect invalid "list: mismatched nesting"     "lllleee"

# ── Dictionaries ─────────────────────────────────────────
expect valid   "dict: empty"                  "de"
expect valid   "dict: basic"                  "d3:cow3:moo4:spam4:eggse"
expect valid   "dict: list value"             "d4:spaml1:a1:bee"
expect valid   "dict: sorted keys"            "d1:a1:b1:c1:de"
expect valid   "dict: byte-order sort"        "d1:A1:x1:a1:ye"
expect invalid "dict: unsorted keys"          "d4:spam4:eggs3:cow3:mooe"
expect invalid "dict: duplicate keys"         "d3:aaa1:b3:aaa1:ce"
expect invalid "dict: reverse sorted"         "d1:b1:x1:a1:ye"
expect invalid "dict: non-string key"         "di3e4:spame"
expect invalid "dict: unclosed"               "d"
expect invalid "dict: reverse byte-order"     "d1:a1:x1:A1:ye"

# ── Top-level ────────────────────────────────────────────
expect invalid "top: empty input"             ""
expect invalid "top: two values"              "4:spam4:eggs"
expect invalid "top: trailing garbage"        "4:spamgarbage"
expect invalid "top: leading whitespace"      " 4:spam"
expect invalid "top: trailing whitespace"     "4:spam "
expect invalid "top: bare e"                  "e"
expect invalid "top: bare l"                  "l"
expect invalid "top: bare d"                  "d"
expect invalid "top: bare i"                  "i"

# ── Binary data ──────────────────────────────────────────
expect_hex valid "string: binary content (null bytes)" "333a000102"

# ── Summary ──────────────────────────────────────────────
total=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then
  echo "ALL $total TESTS PASSED"
else
  echo "$FAIL/$total TESTS FAILED"
  exit 1
fi
