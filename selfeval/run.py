import json
import random
import subprocess

sys_compute = "You will perform multiplication and respond with number with decimal digits only, no separators."
sys_verify = "You will verify multiplication result and respond with 'True' or 'False'"

def callmodel(req):
    out = subprocess.check_output(["curl", "-s", "-X", "POST", "http://localhost:8080/v1/chat/completions", "-H", "'Content-Type: application/json'", "-d", json.dumps(req)])
    res = json.loads(out)
    return res['choices'][0]['message']['content']


def gen_next(n_samples):
    correct_compute = 0
    correct_verify = 0
    for i in range(n_samples):
        A = random.randint(100, 999)
        B = random.randint(100, 999)
        msg = {"role": "user", "content": f"{A}*{B}="}
        req = {
            "messages": [{"role": "system", "content": sys_compute}, msg],
            "n_predict": 64,
            "stream": False,
            "cache_prompt": False,
        }
        C = int(callmodel(req))
        if C == A * B:
            correct_compute += 1

        msg = {"role": "user", "content": f"{A}*{B}=={C}"}
        req = {
            "messages": [{"role": "system", "content": sys_verify}, msg],
            "n_predict": 64,
            "stream": False,
            "cache_prompt": False,
        }
        v = json.loads(callmodel(req).lower())
        if v == (A*B==C):
            correct_verify += 1
        print(f"{A} * {B} == {C}, {A * B == C}, {v}")
        print(f"Correct verification: {correct_verify}")
        print(f"Correct compute: {correct_compute}")

if __name__ == '__main__':
    gen_next(100)
