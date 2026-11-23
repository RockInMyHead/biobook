# Переезд на labjournal.tech — 14.11.2026

## Сделано в коде

- Default `SITE_ORIGIN` / legal texts → `https://labjournal.tech`
- Middleware 301 с `LEGACY_HOSTS` (bionotebook.ru, www.*) на канон
- Nginx-шаблон: `deploy/nginx-labjournal.tech.conf`
- Юр. страницы: URL домена + редакция 14.11.2026

## Статус на ВМ (14.11 ~22:30 МСК) — cutover DONE

- Канон: **https://labjournal.tech** (SSL Let’s Encrypt до 2027-02-12)
- `SITE_ORIGIN` / `LEGAL_BASE_URL` = `https://labjournal.tech`
- www → 301 apex; **bionotebook.ru** → 301 labjournal (path+query)
- robots/sitemap/canonical — labjournal.tech
- Nginx: `sites-available/bio-notebook` = `nginx-labjournal.tech.conf`

## Блокер: DNS

Домен **labjournal.tech** зарегистрирован (NS: `ns1.reg.ru` / `ns2.reg.ru`), но **A-записи нет**.
`bionotebook.ru` → `176.109.103.209` (ВМ).

В панели reg.ru добавить:

| Тип | Имя | Значение |
|-----|-----|----------|
| A | `@` | `176.109.103.209` |
| A | `www` | `176.109.103.209` |

После появления A (проверка: `dig +short labjournal.tech A` → IP ВМ):

```bash
# на ВМ
sudo mkdir -p /var/www/letsencrypt
# временно HTTP-only server_name labjournal.tech → ACME + proxy (или certbot standalone)
sudo certbot certonly --webroot -w /var/www/letsencrypt -d labjournal.tech -d www.labjournal.tech
sudo cp /opt/.../nginx-labjournal.tech.conf /etc/nginx/sites-available/bio-notebook
sudo nginx -t && sudo systemctl reload nginx
sudo sed -i 's|SITE_ORIGIN=.*|SITE_ORIGIN=https://labjournal.tech|' /etc/biobook.env
sudo sed -i 's|LEGAL_BASE_URL=.*|LEGAL_BASE_URL=https://labjournal.tech|' /etc/biobook.env
sudo systemctl restart biobook
```

## Внешние сервисы (Артём)

1. **Яндекс OAuth** — Redirect URI: `https://labjournal.tech/auth/yandex/callback`
2. **Яндекс Вебмастер / GSC** — добавить labjournal.tech, sitemap, переобход; смена адреса со старого домена
3. **Метрика** — зеркало/фильтр по новому хосту (счётчик тот же)
4. **IndexNow / Bing** — новый домен
