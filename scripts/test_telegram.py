import requests

from config.settings import settings
from bybit_bot.api.client import BybitClient


def main() -> None:
    client = BybitClient()
    balance = client.get_wallet_balance()
    equity = balance["list"][0]["totalEquity"]
    text = f"Test message: total equity ${equity}"
    url = f"https://api.telegram.org/bot{settings.telegram_token}/sendMessage"
    payload = {"chat_id": settings.telegram_chat_id, "text": text}
    resp = requests.post(url, json=payload, timeout=10)
    print("Telegram status:", resp.status_code)
    print(resp.text)


if __name__ == "__main__":
    main()

