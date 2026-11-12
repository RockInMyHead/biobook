# ТЗ для разработчика: технические SEO-правки anticus.tech / Bio-Notebook

Версия 1, 10.11.2026. Подготовил SEO по Bio-Notebook. Основание: проверка сайта 10.11.2026 (сырой HTML, заголовки ответов, sitemap, robots.txt).
Статус: к выполнению после согласования с Артемом. Пункты с пометкой «ждём решения» не делать, пока Артем не ответит.

## Сводка задач

| № | Приоритет | Задача | Где | Оценка сложности |
|---|---|---|---|---|
| 1 | Срочно | Исправить цену в разметке JSON-LD (сейчас 0 ₽, на странице 379 ₽/мес) | /bio-notebook/ | малая |
| 2 | Срочно | Убрать /shop из sitemap.xml | /sitemap.xml | малая |
| 3 | Высокий | Закрыть страницу входа от индексации и убрать из sitemap | /bio-notebook/login | малая |
| 4 | Высокий | Редирект 301 с www.anticus.tech на anticus.tech | весь хост www | малая |
| 5 | Высокий | Несуществующие страницы должны отдавать 404, а не копию главной | весь сайт | средняя |
| 6 | Высокий | HEAD-запрос к страницам журнала отдаёт 405 — должен отдавать 200 | /bio-notebook/* | малая |
| 7 | Средний | Ссылка «IT → Подробнее» на главной должна вести на журнал | / | малая |
| 8 | Средний (ждём решения) | «Python-ячейки» в описании страницы, но не в тексте | /bio-notebook/ | малая |
| 9 | Средний | Подготовить раздел /bio-notebook/guides/ под статьи | новый раздел | средняя |

---

## 1. Цена в разметке JSON-LD (срочно)

**Проблема.** В JSON-LD страницы https://anticus.tech/bio-notebook/ указано `offers.price: "0"` («Бесплатный тариф для старта»). В видимом тексте: тариф «Студент», первый месяц 0 ₽ / 30 дней, далее 379 ₽/мес. Поисковики и ИИ-ассистенты (Яндекс Нейро, Алиса) берут цену из разметки и могут называть продукт бесплатным.

**Сделать.** Заменить весь блок `<script type="application/ld+json">` на странице на фрагмент ниже. Блок должен быть в серверном HTML (не добавляться через JS).
Поля в квадратных скобках — ждём решения Артема; до решения поставить так: `name` = «Антикусь», `description` = текущий первый абзац страницы без упоминания Python-ячеек.

```json
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "Organization",
      "@id": "https://anticus.tech/#organization",
      "name": "[Антикусь — ждём подтверждения написания]",
      "legalName": "Общество с ограниченной ответственностью «АНТИКУСЬ»",
      "url": "https://anticus.tech/",
      "logo": "https://anticus.tech/bio-notebook/static/icon-512.png",
      "email": "anticus.tech@internet.ru",
      "sameAs": ["https://github.com/RockInMyHead/biobook"]
    },
    {
      "@type": "WebSite",
      "@id": "https://anticus.tech/#website",
      "url": "https://anticus.tech/",
      "name": "[то же имя, что в Organization.name]",
      "publisher": { "@id": "https://anticus.tech/#organization" }
    },
    {
      "@type": "SoftwareApplication",
      "@id": "https://anticus.tech/bio-notebook/#software",
      "name": "Bio-Notebook",
      "alternateName": "Электронный лабораторный журнал Bio-Notebook",
      "url": "https://anticus.tech/bio-notebook/",
      "description": "[одна фраза-описание продукта — ждём от Артема]",
      "applicationCategory": "EducationalApplication",
      "applicationSubCategory": "Electronic Lab Notebook",
      "operatingSystem": "Web",
      "inLanguage": "ru",
      "image": "https://anticus.tech/bio-notebook/static/og-image.png",
      "publisher": { "@id": "https://anticus.tech/#organization" },
      "featureList": [
        "Ведение этапов, узлов, протоколов и результатов эксперимента",
        "Распознавание рукописных записей и фото приборов (OCR)",
        "ИИ-чат по эксперименту",
        "Экспорт отчётов в DOCX и PDF",
        "Презентация из отчёта (PPTX и PDF)",
        "Голосовой ввод"
      ],
      "offers": {
        "@type": "AggregateOffer",
        "priceCurrency": "RUB",
        "lowPrice": "0",
        "highPrice": "379",
        "offerCount": 2,
        "offers": [
          {
            "@type": "Offer",
            "name": "Тариф «Студент», пробный месяц",
            "price": "0",
            "priceCurrency": "RUB",
            "description": "Первый месяц тарифа «Студент»: 0 ₽ / 30 дней, без карты."
          },
          {
            "@type": "Offer",
            "name": "Тариф «Студент»",
            "price": "379",
            "priceCurrency": "RUB",
            "description": "Далее 379 ₽ в месяц, отмена в любой момент.",
            "priceSpecification": {
              "@type": "UnitPriceSpecification",
              "price": "379",
              "priceCurrency": "RUB",
              "unitCode": "MON",
              "billingDuration": 1
            }
          }
        ]
      }
    }
  ]
}
```

Важно: значения в разметке должны совпадать с текстом страницы. Если цена на странице поменяется — меняется и здесь. `@id` организации `https://anticus.tech/#organization` уже используется на главной, не менять.

**Принято, если:**
- `curl -sL https://anticus.tech/bio-notebook/ | grep -o '"price": *"[0-9]*"'` показывает и 0, и 379;
- Яндекс Вебмастер → Инструменты → Валидатор микроразметки: нет ошибок (предупреждения о необязательных полях допустимы).

## 2. Убрать /shop из sitemap (срочно)

**Проблема.** В https://anticus.tech/sitemap.xml есть https://anticus.tech/shop. Эта страница отдаёт 200, но содержимое — копия главной с canonical на `/`. Для поисковика это мусорный URL.

**Сделать.** Удалить `<url>` с `/shop` из sitemap. В sitemap должны быть только живые страницы с кодом 200 и собственным содержимым. Если sitemap генерируется автоматически — исключить /shop в генераторе.

**Принято, если:** `curl -s https://anticus.tech/sitemap.xml | grep shop` ничего не выводит.

## 3. Страница входа: noindex и убрать из sitemap

**Проблема.** https://anticus.tech/bio-notebook/login (без слеша) в sitemap, отдаёт 200, нет запрета индексации. Служебная страница не должна быть в поиске.

**Сделать.**
1. Убрать URL из sitemap.
2. Добавить в `<head>`: `<meta name="robots" content="noindex, nofollow">` и HTTP-заголовок `X-Robots-Tag: noindex, nofollow`.
3. Не закрывать через `Disallow` в robots.txt (иначе робот не увидит noindex).
4. Ссылка регистрации с лендинга `/bio-notebook/login?mode=register` должна продолжать работать.

**Принято, если:** `curl -sI https://anticus.tech/bio-notebook/login` содержит `x-robots-tag: noindex, nofollow`; в sitemap URL нет; регистрация открывается.

## 4. Редирект www → без www

**Проблема.** https://www.anticus.tech/ и https://www.anticus.tech/bio-notebook/ отдают 200 — сайт доступен по двум адресам (дубль).

**Сделать.** Постоянный редирект 301 с `https://www.anticus.tech/<путь>?<параметры>` на `https://anticus.tech/<путь>?<параметры>` (путь и параметры сохранять). Одним шагом, без цепочек. Заодно проверить, что `http://anticus.tech/` и `http://www.anticus.tech/` одним редиректом 301 ведут на `https://anticus.tech/` (мы это проверить не смогли).

**Принято, если:**
```
curl -sI https://www.anticus.tech/             → 301, Location: https://anticus.tech/
curl -sI https://www.anticus.tech/bio-notebook/ → 301, Location: https://anticus.tech/bio-notebook/
curl -sI http://anticus.tech/                  → 301, Location: https://anticus.tech/
```

## 5. Настоящие 404 вместо копии главной

**Проблема.** Несуществующие адреса отдают 200 и HTML главной (canonical на `/`). Проверено: `/shop`, `/blog`, `/llms.txt`, `/NeuroMod/`, `/about`, `/pricing`, `/contacts`, `/contact`, `/app`. Поисковики считают это ошибкой («soft 404»), ссылка «Neuro» в меню ведёт на такую пустышку.

**Сделать.**
1. Для любых путей без своей страницы отдавать код **404** со страницей «Страница не найдена» (с ссылками на главную и на журнал) и заголовком `X-Robots-Tag: noindex`. В разделе /bio-notebook/* это уже работает правильно — сделать так же на основном сайте.
2. Пункт меню «Neuro» (`/NeuroMod/`): либо вести на существующую страницу, либо убрать из меню.
3. Не сломать живые страницы: `/`, `/bio/`, `/fabrika/`, `/bio-notebook/`, юридические страницы.

**Принято, если:** `curl -s -o /dev/null -w "%{http_code}" https://anticus.tech/blog` → 404 (и для каждого пути из списка); живые страницы → 200.

## 6. HEAD-запрос к журналу отдаёт 405

**Проблема.** `curl -sI https://anticus.tech/bio-notebook/` возвращает 405 (JSON-ошибка). Часть роботов и сервисов мониторинга проверяют страницу HEAD-запросом и считают её недоступной. На `/` HEAD работает (200).

**Сделать.** Разрешить HEAD для публичных страниц /bio-notebook/*: тот же код и заголовки, что у GET, без тела (в большинстве фреймворков — добавить HEAD к методам роута или включить автоматическую обработку HEAD).

**Принято, если:** `curl -sI https://anticus.tech/bio-notebook/` → 200, заголовки совпадают с GET.

## 7. Ссылка «IT → Подробнее» на главной

**Проблема.** На https://anticus.tech/ карточка «IT» («Цифровые решения для лабораторий: электронный журнал…») ведёт кнопкой «Подробнее» на `/bio` — это магазин, а не журнал.

**Сделать.** Сменить ссылку на `/bio-notebook/`. Текст кнопки желательно сделать говорящим: «Электронный журнал Bio-Notebook».

**Принято, если:** клик по кнопке открывает https://anticus.tech/bio-notebook/ ; в HTML главной `href="/bio-notebook/"`.

## 8. «Python-ячейки» (ждём решения Артема)

**Проблема.** Фраза «Python-ячейки с выполнением кода» есть в `<meta name="description">` и в JSON-LD, но её нет в тексте страницы. Обещание в описании, которого нет на странице, — плохой сигнал для поисковиков и пользователей.

**Сделать после ответа Артема — один из вариантов:**
- функция есть → добавить на страницу видимый блок о ней (в серверном HTML);
- функции нет → убрать фразу из meta description, OG-описания и JSON-LD.

## 9. Раздел под статьи /bio-notebook/guides/

**Зачем.** Первая статья — «Как оформить отчёт по лабораторной работе» (4 538 показов в месяц по Wordstat, снято 10.11.2026). Дальше будут другие. Текст готовлю я, от разработчика нужен шаблон.

**Сделать.**
1. Шаблон страницы статьи по адресу `https://anticus.tech/bio-notebook/guides/<slug>/` (со слешем на конце).
2. Весь текст, заголовки, таблицы и ссылки — в серверном HTML (страница должна читаться с отключённым JS).
3. В шаблоне: свой `<title>`, `<meta name="description">`, `<link rel="canonical">` на себя, один H1, хлебные крошки (Главная → Bio-Notebook → Статьи → Статья), дата публикации и обновления, автор, блок-призыв «Попробовать Bio-Notebook» со ссылкой на регистрацию, поддержка таблиц и файлов для скачивания (DOCX).
4. JSON-LD Article + BreadcrumbList (заполню сам, нужен только слот в шаблоне).
5. Новые статьи автоматически добавляются в sitemap.xml.
6. Страница-список `/bio-notebook/guides/` со ссылками на все статьи; ссылка на неё из подвала лендинга.

**Принято, если:** тестовая статья открывается по адресу со слешем, отдаёт 200, видна без JS, есть в sitemap, title/description/canonical свои.

---

## После выкладки

1. Яндекс Вебмастер → Индексирование → Переобход страниц: `/`, `/bio-notebook/`, `/bio-notebook/login`, `/shop`, `/NeuroMod/`, `https://www.anticus.tech/`, `https://www.anticus.tech/bio-notebook/`.
2. Вебмастер → Индексирование → Файлы Sitemap: отправить sitemap.xml заново.
3. Написать мне (через Артема), что выкачено — я перепроверю все пункты по критериям приёмки.

## Чек-лист приёмки

- [ ] 1. В JSON-LD цены 0 и 379, валидатор без ошибок
- [ ] 2. /shop нет в sitemap
- [ ] 3. login: noindex, нет в sitemap, регистрация работает
- [ ] 4. www → без www, 301 одним шагом; http → https
- [ ] 5. Несуществующие пути → 404; «Neuro» в меню исправлен
- [ ] 6. HEAD /bio-notebook/ → 200
- [ ] 7. «IT → Подробнее» → /bio-notebook/
- [ ] 8. Python-ячейки: решение принято и выполнено
- [ ] 9. Шаблон /bio-notebook/guides/<slug>/ готов
