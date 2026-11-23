---
type: status
title: Что сделано по ТЗ / SEO+GEO-аудиту bionotebook.ru — 13.11.2026
created: 2026-11-13
updated: 2026-11-13
status: done-partial
prod: https://bionotebook.ru
vm: 176.109.103.209
verified: curl ~13:10 МСК + пост-приёмка ~14:15 МСК (UA Chrome)
---

# Что уже сделано — Bio-Notebook (bionotebook.ru)

Снимок после выката **13.11.2026 ~13:10 МСК**, плюс правки по независимой приёмке **~14:15 МСК**.  
Источники: полное ТЗ, SEO+GEO-аудит, `ACCEPT-1310-bionotebook-ru-2026-11-13.md`.

---

## Итог

| Категория | Кол-во |
|---|---|
| Закрыто на проде (приёмка curl) | **~20** |
| Частично | **~5** |
| Не сделано / не Артем / нужна ручная приёмка | остальное |

Уже было «не трогать» до правок: 301 со старого раздела, canonical, noindex login/lab-admin, цены 0/319/LabOS в карточках, HEAD 200, http→https.

---

## Закрыто на проде

### SEO / инфраструктура

| ID | Что | Приёмка |
|---|---|---|
| **A1** | `robots.txt` | 200, text/plain, Disallow + Clean-param + Sitemap на apex |
| **A2** | `sitemap.xml` | 200, 13 `<loc>`, все URL 200 |
| **A3** | www → apex | `https://www…/guides/` → 301 `https://bionotebook.ru/guides/`; http www → https apex |
| **A4** | Зеркало по IP | `https://176.109.103.209/` → 444 (код 000 у curl) |
| **A5** | Ссылки anticus в гайде об отчёте | 0 × `anticus.tech/bio-notebook` |
| **A6** | Старые пути `/bio-notebook/static/…` | `site.webmanifest` → `/static/icon-*.png`; start_url `/` |
| **A7** | `/legal/*` | 4× 301 на канонические `/privacy` и т.д. |
| **A9** | gzip / кэш статики / HSTS | lucide: gzip + Cache-Control; HSTS `max-age=31536000` |
| **A11** | WebSite.name | `Bio-Notebook` (Organization остаётся «Антикусь») |

### Юр. / лендинг / контент

| ID | Что | Приёмка |
|---|---|---|
| **C1** | Согласие в форме LabOS | чекбокс + ссылки на consent/privacy; сервер без флага → 400 |
| **C2** | Политика: Метрика / Вебвизор | нет «при её подключении»; есть «Вебвизор»; Google Fonts упомянуты |
| **C3** | «узел/узлы» на публичных | 0 на `/`, гайде «как вести», странице сравнения |
| **C4** | Первый экран | «Лаба сделана - отчёт готов.» + подзаголовок + «без карты» / 319 |
| **C5** | Блок возможностей | `id="vozmozhnosti"`; якоря `#otchet` / `#ocr` сохранены |
| **C7** | Контраст юр. ссылок в подвале | усилен (цвет/размер) |
| **C9** (частично) | Оферта | дата редакции есть; ссылка на `/#ceny`; H1 гайдов с `:` перед `<br>` |
| **N1** | «Пробный период» в гайде об отчёте | убрано; Free 0 ₽ без срока / Pro 319 / LabOS |
| **N2** | Битые `/bio-notebook/…` на сравнении | 0 × `href="/bio-notebook` |
| **N4** | featureList JSON-LD | без «узлов» |
| **N5** | Дата хаба `/guides/` | lastmod / «Обновлено» от последней статьи |

### Аналитика / продукт (код на проде)

| ID | Что | Приёмка |
|---|---|---|
| **D1** | Цели Метрики | подключён `/static/metrika-goals.js` (`bnGoal` / reachGoal) |
| **D2** (код) | Вебвизор в кабинете | в шаблоне `webvisor: false`, если есть `user` |
| **B5** | «Оформить Pro» | `login?mode=register&plan=pro` + плашка на форме |
| **B2** (частично) | «узел» → «этап» в UI | правки в шаблонах этапа/чата |
| **B4** (частично) | Тулбар | flex-wrap на topbar/toolbar |
| **B6** (код) | PPTX | API 403 на Free; на лендинге/гайде как Pro |
| **B7** (код) | OCR-статус | замена сообщения, не дубль |
| **B8** (код) | Версии «назад» | skip текущего снимка + busy-флаг |
| **B9** (код) | Cookie | localStorage + cookie ~365 дней |

### Прочее (вне таблицы ТЗ, но на проде)

- Аватар профиля: загрузка/удаление фото, показ в меню и друзьях.
- Регистрация: Free без авто-trial; `bn_signup` query после email/Yandex.

---

## Частично / не закрыто приёмкой

| ID | Статус | Комментарий |
|---|---|---|
| **A8** | Частично | на **bionotebook.ru** `/index.html` → 301 `/`; хвосты pricing/guides 2 шага — только nginx anticus.tech |
| **A10** | Частично | fonts css2 = 1; JPEG/lucide/hero-video ещё не урезаны |
| **B1, B3, B6–B10** | Нужен вход / app-accept | app-accept: Free/Pro OCR/PPTX/DOCX ок; склонения/фото OCR — хвосты |
| **C6** | HTML ок | `#ceny` без `<img>`; визуал 1440×900 — глазами |
| **C8** | Не добито | hero гайда — визуал |
| **C9** автор | Ждёт [У] | Article author = Organization |
| **C10** | [У] | контент-план |
| **D1** | Почти | срочные селекторы/таймер/double-init выкачены; цели создать в интерфейсе Метрики; `bn_signup` / paid / first_* — кабинет |
| **D3** | Не делалось | «Откуда узнали?» |
| **E1–E4** | Артем | Вебмастер, кэш, GSC/IndexNow/Bing, контроль 7/14 дней |

---

## Пост-приёмка ~14:15 (выкат)

- `static/metrika-goals.js` — `__bnMetrikaGoalsInit`, `#btn-yandex*`, `#auth-form` только register, `setTimeout(2000)` для ceny, FAQ toggle-only  
- `login.html` — `data-yandex-login`, один include goals  
- hero «—»; даты юр. 13.11; опечатка оферты; lastmod гайдов/сравнения 13.11; H1 ГОСТ без «:» перед «по»; подвал ≥0.875rem; cookie на публичных страницах + текст про Метрику; PPTX Pro в featureList/карточке  

Приёмка: `wiki/syntheses/ACCEPT-1310-bionotebook-ru-2026-11-13.md`.  
Аудит: `wiki/syntheses/seo-geo-audit-bionotebook-ru-2026-11-13.md`.

---

## Следующие шаги (коротко)

1. **Артем** — E1 сразу; цели Метрики JS-событиями; E2–E4.  
2. **A8** — nginx anticus.tech (pricing/guides без 2 шагов).  
3. **A10** — WebP / lucide / вес hero-video.  
4. `?_ym_debug=1` после создания целей; `bn_signup` сценарий с тестовым аккаунтом.
