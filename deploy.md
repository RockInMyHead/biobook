# Развертывание Lab-Journal

> Актуальные реквизиты доступа, инфраструктура и ключи — в корневом [`../deploy.md`](../deploy.md).

## Быстрый деплой

```bash
# Актуальный сервер: ubuntu@95.163.215.131 (VK Cloud), ключ ./anticus-front.pem
# Не заливать venv и data — на сервере своё окружение и пользовательские данные.
tar -czf /tmp/bio-notebook-deploy.tar.gz \
  -C biobook --exclude=venv --exclude='*/venv' --exclude=.git --exclude=data --exclude='*/data' .
scp -i ./anticus-front.pem /tmp/bio-notebook-deploy.tar.gz ubuntu@95.163.215.131:/tmp/
ssh -i ./anticus-front.pem ubuntu@95.163.215.131 \
  "sudo tar -xzf /tmp/bio-notebook-deploy.tar.gz -C /opt/bio-notebook && \
   sudo rm /tmp/bio-notebook-deploy.tar.gz && sudo systemctl restart biobook"
rm /tmp/bio-notebook-deploy.tar.gz
```

## На сервере

- **URL:** https://anticus.tech/bio-notebook/
- **Путь:** `/opt/bio-notebook/web`
- **Сервис:** `biobook.service` (uvicorn, порт `127.0.0.1:3001`)
- **Админка лаб. журнала (отдельно от сайта):** https://anticus.tech/bio-notebook/lab-admin — ключ `ADMIN_KEY` (`admin123` по умолчанию)
- Старый URL `/bio-notebook/admin` перенаправляет на `/lab-admin`

## Канонический домен (актуально)

- **Прод:** https://labjournal.tech  
- **ВМ:** `user1@176.109.103.209`, код `/opt/bio-notebook/web`, сервис `biobook`  
- **Nginx:** `deploy/nginx-labjournal.tech.conf`  
- Старый **bionotebook.ru** → 301 на labjournal.tech (после DNS+SSL)

Env (`/etc/biobook.env`):

```bash
BASE_PATH=
SITE_ORIGIN=https://labjournal.tech
LEGAL_BASE_URL=https://labjournal.tech
COOKIE_SECURE=true
LEGACY_HOSTS=bionotebook.ru,www.bionotebook.ru,www.labjournal.tech
```

## Яндекс ID (OAuth)

1. Создайте / обновите приложение: https://oauth.yandex.ru/client/new  
   - **Redirect URI:** `https://labjournal.tech/auth/yandex/callback`
   - (legacy) `https://bionotebook.ru/auth/yandex/callback` — пока живёт 301
   - Права: login:email, login:info

2. В `/etc/biobook.env` — `YANDEX_CLIENT_ID` / `YANDEX_CLIENT_SECRET`, `SITE_ORIGIN=https://labjournal.tech`.

3. Перезапуск: `systemctl restart biobook`

Кнопка «Быстрая регистрация через Яндекс» появляется только когда заданы оба ключа.

## OCR через OpenRouter

В `/etc/biobook.env` (права `600`, не коммитить):

```bash
OCR_API_KEY=sk-or-v1-...
OCR_API_URL=https://openrouter.ai/api/v1/chat/completions
OCR_VISION_MODEL=qwen/qwen-2.5-vl-7b-instruct
AI_HTTP_PROXY=http://USER:PASS@HOST:PORT
```

`AI_HTTP_PROXY` используют и OCR, и ИИ-чат (httpx). После изменений: `sudo systemctl restart biobook`.
