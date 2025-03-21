import requests

proxy = "107.181.187.195:11412"
proxy_url = f"https://hutanu2003:fwuvojgjkx@{proxy}"
proxies = {
    "http": proxy_url
}

try:
    r = requests.get("https://httpbin.org/ip", proxies=proxies, timeout=10)
    r.raise_for_status()
    print("Proxy works! Response:", r.text)
except Exception as e:
    print("Proxy failed:", e)