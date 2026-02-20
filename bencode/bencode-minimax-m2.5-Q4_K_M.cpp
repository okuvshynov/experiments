#include <iostream>
#include <string>
#include <vector>
#include <cctype>

using namespace std;

bool parse_string(const string& s, size_t& pos);
bool parse_string_content(const string& s, size_t& pos, string& out);
bool parse_int(const string& s, size_t& pos);
bool parse_list(const string& s, size_t& pos);
bool parse_dict(const string& s, size_t& pos);
bool parse_bencode(const string& s, size_t& pos);

/* ------------------------------------------------------------
   string  –  <length>:
   ------------------------------------------------------------ */
bool parse_string(const string& s, size_t& pos) {
    if (pos >= s.size() || !isdigit(static_cast<unsigned char>(s[pos])))
        return false;

    size_t start = pos;
    while (pos < s.size() && isdigit(static_cast<unsigned char>(s[pos])))
        ++pos;                     // read the decimal length
    if (pos == start) return false;          // no digits
    if (pos >= s.size() || s[pos] != ':') return false;
    ++pos;                       // skip ':'

    // convert length to size_t
    size_t len = 0;
    for (size_t i = start; i < pos - 1; ++i) {
        len = len * 10 + (s[i] - '0');
    }
    if (pos + len > s.size()) return false; // not enough bytes
    pos += len;                // skip the string data
    return true;
}

/* same as above but also return the string content */
bool parse_string_content(const string& s, size_t& pos, string& out) {
    if (pos >= s.size() || !isdigit(static_cast<unsigned char>(s[pos])))
        return false;

    size_t start = pos;
    while (pos < s.size() && isdigit(static_cast<unsigned char>(s[pos])))
        ++pos;
    if (pos == start) return false;
    if (pos >= s.size() || s[pos] != ':') return false;
    ++pos; // skip ':'

    size_t len = 0;
    for (size_t i = start; i < pos - 1; ++i) {
        len = len * 10 + (s[i] - '0');
    }
    if (pos + len > s.size()) return false;
    out = s.substr(pos, len);
    pos += len;
    return true;
}

/* ------------------------------------------------------------
   integer –  i[‑]digits e
   ------------------------------------------------------------ */
bool parse_int(const string& s, size_t& pos) {
    if (pos >= s.size() || s[pos] != 'i') return false;
    ++pos;                                 // skip 'i'

    bool negative = false;
    if (pos < s.size() && s[pos] == '-') {
        negative = true;
        ++pos;
    }
    if (pos >= s.size() || !isdigit(static_cast<unsigned char>(s[pos])))
        return false;

    size_t digit_start = pos;
    while (pos < s.size() && isdigit(static_cast<unsigned char>(s[pos])))
        ++pos;                             // read all digits

    if (pos >= s.size() || s[pos] != 'e') return false;
    string num = s.substr(digit_start, pos - digit_start);

    // leading‑zero checks
    if (num.size() > 1 && num[0] == '0') return false;
    if (negative) {
        if (num == "0") return false;                 // i‑0e
        if (num.size() > 1 && num[0] == '0') return false; // i‑0…
    }

    ++pos; // skip 'e'
    return true;
}

/* ------------------------------------------------------------
   list – l … e
   ------------------------------------------------------------ */
bool parse_list(const string& s, size_t& pos) {
    if (pos >= s.size() || s[pos] != 'l') return false;
    ++pos;                         // skip 'l'

    while (pos < s.size() && s[pos] != 'e') {
        if (!parse_bencode(s, pos))
            return false;
    }
    if (pos >= s.size() || s[pos] != 'e') return false;
    ++pos;                         // skip 'e'
    return true;
}

/* ------------------------------------------------------------
   dictionary – d … e
   ------------------------------------------------------------ */
bool parse_dict(const string& s, size_t& pos) {
    if (pos >= s.size() || s[pos] != 'd') return false;
    ++pos;                         // skip 'd'

    vector<string> keys;
    while (pos < s.size() && s[pos] != 'e') {
        string key;
        if (!parse_string_content(s, pos, key))
            return false;
        keys.push_back(key);

        if (!parse_bencode(s, pos))
            return false;
    }
    if (pos >= s.size() || s[pos] != 'e') return false;
    ++pos;                         // skip 'e'

    // check sorted order and uniqueness
    for (size_t i = 1; i < keys.size(); ++i) {
        if (keys[i] <= keys[i - 1]) return false;
    }
    return true;
}

/* ------------------------------------------------------------
   top‑level dispatcher
   ------------------------------------------------------------ */
bool parse_bencode(const string& s, size_t& pos) {
    if (pos >= s.size()) return false;
    char c = s[pos];
    if (c >= '0' && c <= '9')
        return parse_string(s, pos);
    if (c == 'i')
        return parse_int(s, pos);
    if (c == 'l')
        return parse_list(s, pos);
    if (c == 'd')
        return parse_dict(s, pos);
    return false;                 // unknown leading character
}

/* ------------------------------------------------------------
   main
   ------------------------------------------------------------ */
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // read whole stdin
    string input( (istreambuf_iterator<char>(cin)),
                  istreambuf_iterator<char>() );

    size_t pos = 0;
    bool ok = parse_bencode(input, pos);

    if (ok && pos == input.size())
        return 0;    // valid
    else
        return 1;    // invalid
}
