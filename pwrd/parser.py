import plistlib

def parse_powermetrics(content):
    data = plistlib.loads(content)

if __name__ == '__main__':
    parse_powermetrics()
