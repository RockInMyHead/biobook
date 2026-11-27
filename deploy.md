# Развертывание Lab-Journal

> Актуальные реквизиты доступа, инфраструктура и ключи — в корневом [`../deploy.md`](../deploy.md).

## CI/CD (GitHub Actions)

Репозиторий: [RockInMyHead/biobook](https://github.com/RockInMyHead/biobook)

| Workflow | Когда | Что делает |
|----------|--------|------------|
| `.github/workflows/ci.yml` | PR + push в `main` (web/) | `compileall` + smoke-import FastAPI |
| `.github/workflows/deploy.yml` | push в `main` + manual | rsync `web/` → `/opt/bio-notebook/web` (без `data/`), restart `biobook`, smoke HTTP, IndexNow |

Secrets (Settings → Secrets → Actions): `DEPLOY_SSH_KEY`, `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_PORT`.  
Первичная установка ключа: `./scripts/ci-setup-deploy-key.sh` (нужен SSH `user1` на прод).

Прод: `user1@176.109.103.209`, сервис `biobook`, канон https://labjournal.tech.  
`/etc/biobook.env` и `web/data/` **не** перезаписываются деплоем.

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
- **Админка лаб. журнала (отдельно от сайта):** https://anticus.tech/bio-notebook/lab-admin — задайте ADMIN_KEY в окружении
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

## SMTP (подтверждение email при регистрации)

В `/etc/biobook.env` (права `600`, не коммитить) — Яндекс Почта, приложение:

```bash
SMTP_HOST=smtp.yandex.ru
SMTP_PORT=465
SMTP_SSL=true
SMTP_USER=rmyhead@yandex.ru
SMTP_PASSWORD=<пароль_приложения>
SMTP_FROM=rmyhead@yandex.ru
EMAIL_VERIFY_TTL_HOURS=48
```

После регистрации по паролю письмо со ссылкой `/verify-email?token=…`; вход без подтверждения блокируется. Яндекс ID считается подтверждённым. Существующие аккаунты grandfathered (`email_verified` по умолчанию `true`).

## OCR через OpenRouter

В `/etc/biobook.env` (права `600`, не коммитить):

```bash
OCR_API_KEY=sk-or-v1-...
OCR_API_URL=https://openrouter.ai/api/v1/chat/completions
OCR_VISION_MODEL=qwen/qwen-2.5-vl-7b-instruct
AI_HTTP_PROXY=http://USER:PASS@HOST:PORT
```

`AI_HTTP_PROXY` используют и OCR, и ИИ-чат (httpx). После изменений: `sudo systemctl restart biobook`.

## Журнал Telegram-бота

В режиме опроса команда `/logs` показывает 10 последних записей. `/logs errors` оставляет ошибки и предупреждения. Число записей можно задать командой `/logs 20` или `/logs errors 20`. Журнал доступен чатам, перечисленным через запятую в `TG_CHAT_ID` в `/etc/biobook.env`.

В SQLite хранятся до 500 последних записей. Чтобы проверить сбой, когда бот не отвечает, откройте журнал службы:

```bash
sudo journalctl -u biobook -f | grep --line-buffered 'telegram event='
```

Записи содержат этап сбоя, HTTP-код, код Telegram, класс исключения и длительность запроса, когда эти данные доступны. При сбое опроса журнал показывает задержку до следующей попытки и отмечает восстановление связи.

## Обновление после аудита

- Задайте уникальные `ADMIN_KEY`, `SESSION_SECRET` и, при необходимости, `OAUTH_STATE_SECRET` в закрытом окружении. Панель администратора без `ADMIN_KEY` отключена. После обновления пользователи войдут снова, так как приложение хранит новые сессии в SQLite.
- Перед обновлением сохраните SQLite, каталог вложений и папку блокнотов. При первом запуске приложение связывает прежние загрузки с экспериментами или чатами, где их удалось однозначно определить. Неопределённые загрузки остаются на диске. Для выдачи браузеру требуется подтверждённая связь с аккаунтом или опубликованной записью.
- Для Python-ячеек нужен Linux с bubblewrap и разрешёнными пользовательскими пространствами имён. Если профиль контейнера запрещает создание этих пространств имён, как стандартный профиль Docker Desktop, приложение отключает запуск ячеек. Проверьте запуск на целевом Linux-хосте.
- В Docker Tailwind собирается из lock-файла. При установке через systemd выполните `npm ci` и `npm run build:css` в `web/`. Для PDF установите DejaVu Sans либо задайте путь к подходящему TrueType-шрифту в `PDF_EXPORT_FONT`.
