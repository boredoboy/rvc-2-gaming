from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
OVC_DIR = Path("/kaggle/working/OpenVoiceChanger")
VENV_DIR = OVC_DIR / ".venv"
OVC_COMMIT = "8e6a00c628eafc0fd1d68c6bc13f2c17b8a46ea1"
RVC_COMMIT = "7b284a634667c34103eaaeed972b48ccdb4b893e"
HUBERT_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/e6d0c1a17da07c33557852f9dfa2bd44cc75737d/hubert_base.pt"
HUBERT_SHA256 = "f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96"
RMVPE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/e6d0c1a17da07c33557852f9dfa2bd44cc75737d/rmvpe.pt"
RMVPE_SHA256 = "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193"


def kaggle_secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name).strip()
    except Exception:
        return ""


def run(command, *, cwd=None, env=None):
    subprocess.run([str(item) for item in command], cwd=cwd, env=env, check=True)


def install_cloudflared() -> str:
    existing = shutil.which("cloudflared")
    if existing:
        return existing
    if os.geteuid() != 0:
        raise RuntimeError("Kaggle должен запускать notebook от root, чтобы поставить cloudflared из репозитория Cloudflare.")
    key_path = Path("/usr/share/keyrings/cloudflare-main.gpg")
    repo_path = Path("/etc/apt/sources.list.d/cloudflared.list")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request("https://pkg.cloudflare.com/cloudflare-main.gpg", headers={"User-Agent": "RVC2-Kaggle/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        key_path.write_bytes(response.read())
    repo_path.write_text("deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main\n", encoding="utf-8")
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", "cloudflared"])
    found = shutil.which("cloudflared")
    if not found:
        raise RuntimeError("Установка cloudflared завершилась без исполняемого файла.")
    return found


def ensure_node() -> str:
    node = shutil.which("node")
    npm = shutil.which("npm")
    if node and npm:
        try:
            major = int(subprocess.check_output([node, "--version"], text=True).strip().lstrip("v").split(".")[0])
            if major >= 20:
                return npm
        except (ValueError, subprocess.SubprocessError):
            pass

    cache = Path("/kaggle/working/rvc2-node")
    cache.mkdir(parents=True, exist_ok=True)
    listing_url = "https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt"
    request = urllib.request.Request(listing_url, headers={"User-Agent": "RVC2-Kaggle/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        checksums = response.read().decode("ascii")
    # Select the official x64 Linux tarball entry from the current Node 22 LTS release.
    entries = [line.split() for line in checksums.splitlines() if line.split() and re.fullmatch(r"node-v22\.[0-9]+\.[0-9]+-linux-x64\.tar\.xz", line.split()[-1])]
    if not entries:
        raise RuntimeError("Не удалось найти Node.js 22 x64 в официальном списке контрольных сумм.")
    digest, filename = entries[0][0], entries[0][-1]
    archive = cache / filename
    url = f"https://nodejs.org/dist/latest-v22.x/{filename}"
    request = urllib.request.Request(url, headers={"User-Agent": "RVC2-Kaggle/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError("Проверка контрольной суммы архива Node.js не прошла.")
    archive.write_bytes(data)
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(cache, filter="data")
    extracted = cache / filename.removesuffix(".tar.xz")
    os.environ["PATH"] = str(extracted / "bin") + os.pathsep + os.environ.get("PATH", "")
    npm = extracted / "bin" / "npm"
    if not npm.is_file():
        raise RuntimeError("Node.js установлен, но npm не найден.")
    return str(npm)


def install_runtime():
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Этот скрипт нужно запускать в Kaggle Notebook.")
    if not shutil.which("git"):
        raise RuntimeError("В Kaggle image не найден git.")

    print("[1/7] Получаю OpenVoiceChanger и фиксирую версию…", flush=True)
    if OVC_DIR.exists():
        shutil.rmtree(OVC_DIR)
    run(["git", "clone", "--filter=blob:none", "https://github.com/sioaeko/OpenVoiceChanger.git", str(OVC_DIR)])
    run(["git", "checkout", OVC_COMMIT], cwd=OVC_DIR)

    print("[2/7] Ставлю Python 3.10 для совместимости RVC…", flush=True)
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    run([sys.executable, "-m", "uv", "python", "install", "3.10.20"])
    run([sys.executable, "-m", "uv", "venv", "--python", "3.10.20", str(VENV_DIR)])
    python = VENV_DIR / "bin" / "python"
    run([python, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])

    print("[3/7] Ставлю CUDA/RVC зависимости… это самый долгий шаг.", flush=True)
    run([python, "-m", "pip", "install", "-r", OVC_DIR / "backend" / "requirements.txt"])
    fairseq_url = "git+https://github.com/Tps-F/fairseq@ff08af27e302625a27d3502b0791a9367c8af0c7"
    run([python, "-m", "pip", "install", "--no-deps", fairseq_url])
    rvc_url = f"git+https://github.com/RVC-Project/Retrieval-based-Voice-Conversion@{RVC_COMMIT}"
    run([python, "-m", "pip", "install", "--no-deps", rvc_url])

    print("[4/7] Собираю браузерный интерфейс…", flush=True)
    npm = ensure_node()
    frontend = OVC_DIR / "frontend"
    run([npm, "ci"], cwd=frontend)
    run([npm, "run", "build"], cwd=frontend)

    print("[5/7] Скачиваю и проверяю HuBERT/RMVPE веса…", flush=True)
    assets = OVC_DIR / "models" / "assets"
    rmvpe = assets / "rmvpe"
    rmvpe.mkdir(parents=True, exist_ok=True)
    download_verified(HUBERT_URL, assets / "hubert_base.pt", HUBERT_SHA256)
    download_verified(RMVPE_URL, rmvpe / "rmvpe.pt", RMVPE_SHA256)

    print("[6/7] Проверяю, что PyTorch видит GPU…", flush=True)
    check = "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU NOT FOUND')"
    result = subprocess.check_output([python, "-c", check], cwd=OVC_DIR, text=True).strip()
    print(f"    {result}", flush=True)
    if "True" not in result or "GPU NOT FOUND" in result:
        raise RuntimeError("PyTorch не видит CUDA. Приложение не запускаю; проверьте GPU accelerator и Kaggle image.")

    print("[7/7] Подготовка завершена.", flush=True)
    return python


def download_verified(url: str, destination: Path, expected_sha256: str):
    if destination.is_file():
        existing_digest = hashlib.sha256()
        with destination.open("rb") as existing:
            for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                existing_digest.update(chunk)
        if existing_digest.hexdigest() == expected_sha256:
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "RVC2-Kaggle/1.0"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Проверка SHA-256 не прошла для {destination.name}.")
    partial.replace(destination)


class TelegramBot(threading.Thread):
    def __init__(self, token: str, chat_id: str, stop_event: threading.Event):
        super().__init__(name="telegram-bot", daemon=True)
        self.token = token
        self.chat_id = str(int(chat_id))
        self.stop_event = stop_event
        self.url = ""
        self.started_at = int(time.time())

    def api(self, method: str, payload: dict, timeout=35):
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        data = urllib.parse.urlencode(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not result.get("ok"):
            raise RuntimeError("Telegram API returned an error")
        return result.get("result")

    def validate(self):
        result = self.api("getMe", {}, timeout=15)
        if not result or not result.get("is_bot"):
            raise RuntimeError("Telegram token is not a bot token")

    def send(self, text: str):
        self.api("sendMessage", {"chat_id": self.chat_id, "text": text, "disable_web_page_preview": "true"}, timeout=15)

    def ready_message(self):
        return (
            "🎮 Сервер RVC успешно запущен на GPU Kaggle!\n\n"
            f"🔗 Откройте в Chrome/Edge: {self.url}\n"
            "🔐 Введите пароль RVC_ACCESS_PASSWORD из Kaggle Secrets.\n\n"
            "⚙️ На вкладке Studio выберите: Input — микрофон; Output — CABLE Input (VB-Audio); F0 — fcpe или pm; затем Start Voice Changer.\n\n"
            "Команды: /link — ссылка, /status — состояние, /stop — закрыть сервис и освободить GPU."
        )

    def set_url(self, url: str):
        self.url = url
        try:
            self.send(self.ready_message())
            print("Tunnel is ready; link was sent to the configured Telegram chat.", flush=True)
        except Exception:
            print("Не удалось отправить ссылку в Telegram. Проверьте, что пользователь уже нажал Start у бота и секреты заданы верно.", flush=True)

    def run(self):
        offset = None
        while not self.stop_event.is_set():
            payload = {"timeout": 20, "allowed_updates": json.dumps(["message"])}
            if offset is not None:
                payload["offset"] = offset
            try:
                updates = self.api("getUpdates", payload, timeout=30) or []
                for update in updates:
                    offset = max(offset or 0, int(update.get("update_id", 0)) + 1)
                    message = update.get("message") or {}
                    chat = message.get("chat") or {}
                    if str(chat.get("id")) != self.chat_id:
                        continue
                    if int(message.get("date", 0)) < self.started_at:
                        continue
                    command = (message.get("text") or "").split()[0].lower().split("@", 1)[0]
                    if command in {"/start", "/link"}:
                        self.send(self.ready_message() if self.url else "RVC ещё запускается. Ссылка придёт сюда, когда туннель будет готов.")
                    elif command == "/status":
                        state = "работает" if self.url and not self.stop_event.is_set() else "запускается или остановлен"
                        self.send(f"RVC: {state}." + (f"\n{self.url}" if self.url else ""))
                    elif command == "/stop":
                        self.send("Останавливаю туннель и RVC, освобождаю GPU…")
                        self.stop_event.set()
            except Exception:
                if not self.stop_event.is_set():
                    time.sleep(3)


def stream_tunnel_logs(process, ready_event, url_box):
    pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    for raw in iter(process.stdout.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        match = pattern.search(line)
        if match:
            url_box.append(match.group(0))
            ready_event.set()
            line = pattern.sub("[temporary Cloudflare URL hidden]", line)
        if line:
            print(line, flush=True)


def serve(python: Path, cloudflared: str, password: str, bot: TelegramBot):
    stop_event = bot.stop_event
    env = os.environ.copy()
    env.update({
        "RVC_ACCESS_PASSWORD": password,
        "OVC_HOST": "127.0.0.1",
        "OVC_PORT": "8000",
        "OVC_HUBERT_PATH": str(OVC_DIR / "models" / "assets" / "hubert_base.pt"),
        "OVC_RMVPE_ROOT": str(OVC_DIR / "models" / "assets" / "rmvpe"),
        "OVC_UPDATE_CHECK_ENABLED": "false",
        "PYTHONPATH": os.pathsep.join([str(OVC_DIR), str(PROJECT_DIR / "scripts"), env.get("PYTHONPATH", "")]),
    })
    app_process = subprocess.Popen([str(python), str(PROJECT_DIR / "scripts" / "serve.py")], cwd=OVC_DIR, env=env, stdout=None, stderr=None)
    tunnel = None
    try:
        deadline = time.time() + 180
        while time.time() < deadline and not stop_event.is_set():
            if app_process.poll() is not None:
                raise RuntimeError("RVC web server exited during startup; inspect the Kaggle output above.")
            try:
                import socket

                with socket.create_connection(("127.0.0.1", 8000), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("RVC web server did not open port 8000 within 3 minutes.")

        tunnel = subprocess.Popen([cloudflared, "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        ready = threading.Event()
        url_box = []
        threading.Thread(target=stream_tunnel_logs, args=(tunnel, ready, url_box), daemon=True).start()
        if not ready.wait(120):
            raise RuntimeError("Cloudflare tunnel did not produce a URL within 2 minutes.")
        bot.set_url(url_box[0])
        print("RVC ready. Send /stop to the authorized Telegram bot when finished.", flush=True)
        while not stop_event.wait(1):
            if app_process.poll() is not None:
                raise RuntimeError("RVC web server stopped unexpectedly.")
            if tunnel.poll() is not None:
                raise RuntimeError("Cloudflare tunnel stopped unexpectedly.")
    finally:
        for process in (tunnel, app_process):
            if process is not None and process.poll() is None:
                process.terminate()
        for process in (tunnel, app_process):
            if process is not None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        print("RVC and Cloudflare tunnel stopped.", flush=True)


def main():
    access_password = kaggle_secret("RVC_ACCESS_PASSWORD")
    bot_token = kaggle_secret("RVC_TELEGRAM_BOT_TOKEN")
    chat_id = kaggle_secret("RVC_TELEGRAM_CHAT_ID")
    if len(access_password) < 20:
        raise RuntimeError("Добавьте в Kaggle Secrets RVC_ACCESS_PASSWORD длиной не менее 20 символов.")
    if not bot_token or not chat_id:
        raise RuntimeError("Добавьте в Kaggle Secrets RVC_TELEGRAM_BOT_TOKEN (новый, перевыпущенный токен) и RVC_TELEGRAM_CHAT_ID.")
    try:
        chat_id = str(int(chat_id))
    except ValueError as exc:
        raise RuntimeError("RVC_TELEGRAM_CHAT_ID должен быть числом. Узнайте его у @userinfobot.") from exc

    python = install_runtime()
    cloudflared = install_cloudflared()
    stop_event = threading.Event()
    bot = TelegramBot(bot_token, chat_id, stop_event)
    bot.validate()
    bot.start()
    try:
        serve(python, cloudflared, access_password, bot)
    finally:
        stop_event.set()
        bot.join(timeout=5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Получен останов. Завершение…", flush=True)
    except Exception as error:
        print(f"Запуск прерван: {error}", flush=True)
        raise

