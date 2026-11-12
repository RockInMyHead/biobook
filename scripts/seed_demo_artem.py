#!/usr/bin/env python3
"""Seed reagents, consumables, projects and Gantt tasks for user Artem."""
import datetime as dt
import os
import sqlite3
import uuid
from pathlib import Path

USER = "Artem"
DB = os.environ.get("BIOBOOK_DB") or str(
    Path(__file__).resolve().parents[1] / "web" / "data" / "db.sqlite"
)


def _id() -> str:
    return str(uuid.uuid4())[:8]


def main() -> None:
    conn = sqlite3.connect(str(DB))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM reagents WHERE user_id=?", (USER,))
    reag_n = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM consumables WHERE user_id=?", (USER,))
    cons_n = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM projects WHERE user_id=?", (USER,))
    proj_n = cur.fetchone()[0]

    if reag_n < 3:
        reagents = [
            ("LB-бульон (готовый)", "ХимМед", "Жидкий", "", "", "конц. 1×", "+4 °C", "450", "100", "мл", "2027-03-15", "Культивирование E. coli", "Холодильник №2"),
            ("Хлорид натрия", "Реактив", "Твёрдый", "7647-14-5", "NaCl", "ч.д.а.", "комнатная t°", "2.5", "0.5", "кг", "2028-01-01", "Физическая химия", "Шкаф A2"),
            ("Агар Difco", "Sigma-Aldrich", "Твёрдый", "9002-18-0", "", "порошок", "сухое место", "180", "50", "г", "2026-12-01", "Микробиология", "Сухой шкаф B1"),
            ("ИПТГ 0.1 M", "Thermo", "Жидкий", "367-93-1", "", "0.1 M", "-20 °C", "12", "2", "мл", "2026-08-20", "Экспрессия белка", "Морозилка F4"),
            ("Аммоний сульфат", "Реактив", "Твёрдый", "7783-20-2", "(NH4)2SO4", "ч.д.а.", "комнатная t°", "750", "200", "г", "2029-06-01", "Очистка фермента", "Шкаф A3"),
        ]
        for r in reagents:
            cur.execute(
                "SELECT 1 FROM reagents WHERE user_id=? AND name=?",
                (USER, r[0]),
            )
            if cur.fetchone():
                continue
            cur.execute(
                """INSERT INTO reagents (id,user_id,name,supplier,category,cas,formula,concentration,storage,amount,min_amount,unit,expiry,project,location)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (_id(), USER, *r),
            )

    if cons_n < 3:
        consumables = [
            ("Наконечники 200 мкл", "Eppendorf", "Пластик", "2400", "500", "шт", "Стеллаж C1", "стерильные"),
            ("Чашки Петри 90 мм", "Nunc", "Пластик", "120", "30", "шт", "Стеллаж C2", "для E. coli"),
            ("Флаконы 50 мл", "Duran", "Стекло", "36", "10", "шт", "Сушилка", "автоклав"),
            ("Перчатки нитриловые M", "ХимСнаб", "СИЗ", "8", "2", "уп", "Коридор 204", "100 шт/уп"),
        ]
        for c in consumables:
            cur.execute("SELECT 1 FROM consumables WHERE user_id=? AND name=?", (USER, c[0]))
            if cur.fetchone():
                continue
            cur.execute(
                """INSERT INTO consumables (id,user_id,name,supplier,category,amount,min_amount,unit,location,comment)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (_id(), USER, *c),
            )

    if proj_n < 1:
        now = dt.datetime.now().isoformat()
        projects = [
            (
                "Культивирование E. coli (ЛР №3)",
                "Активный",
                "Рост на LB, ОД600, отчёт в Bio-Notebook.",
                "2026-11-25",
                [
                    ("Подготовка среды и инокулят", "Выполнено", "Высокий", "2026-11-05", "2026-11-07"),
                    ("Инкубация и замеры ОД600", "В работе", "Высокий", "2026-11-08", "2026-11-12"),
                    ("Графики и оформление отчёта", "Открыто", "Средний", "2026-11-13", "2026-11-18"),
                    ("Сдача преподавателю", "Открыто", "Средний", "2026-11-19", "2026-11-25"),
                ],
            ),
            (
                "Очистка фермента — пилот",
                "Активный",
                "Осаждение (NH4)2SO4, диализ, активность.",
                "2026-12-15",
                [
                    ("Раствор культуры", "Выполнено", "Средний", "2026-11-01", "2026-11-04"),
                    ("Осаждение сульфатом", "В работе", "Высокий", "2026-11-10", "2026-11-16"),
                    ("Центрифугирование", "Открыто", "Высокий", "2026-11-17", "2026-11-22"),
                    ("Диализ и активность", "Открыто", "Средний", "2026-11-23", "2026-12-05"),
                    ("Финальный отчёт", "Открыто", "Низкий", "2026-12-06", "2026-12-15"),
                ],
            ),
            (
                "Диплом: архив протоколов",
                "Приостановлен",
                "Сводка экспериментов, экспорт DOCX.",
                "2027-01-20",
                [
                    ("Инвентаризация опытов", "Открыто", "Средний", "2026-11-11", "2026-11-20"),
                    ("Шаблон по методичке", "Открыто", "Средний", "2026-11-21", "2026-12-10"),
                    ("Согласование с научруком", "Открыто", "Высокий", "2026-12-11", "2027-01-20"),
                ],
            ),
        ]
        for name, status, desc, end_date, tasks in projects:
            pid = _id()
            cur.execute(
                "INSERT INTO projects (id,user_id,name,status,description,end_date) VALUES (?,?,?,?,?,?)",
                (pid, USER, name, status, desc, end_date),
            )
            cur.execute(
                "INSERT INTO project_members (id,project_id,username) VALUES (?,?,?)",
                (_id(), pid, USER),
            )
            for title, tstatus, priority, start, end in tasks:
                cur.execute(
                    """INSERT INTO project_tasks
                    (id,project_id,title,description,assignee,status,priority,start_date,end_date,created_by,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (_id(), pid, title, "", USER, tstatus, priority, start, end, USER, now),
                )

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM reagents WHERE user_id=?", (USER,))
    r = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM consumables WHERE user_id=?", (USER,))
    c = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM projects WHERE user_id=?", (USER,))
    p = cur.fetchone()[0]
    cur.execute(
        """SELECT COUNT(*) FROM project_tasks t JOIN projects p ON p.id=t.project_id WHERE p.user_id=?""",
        (USER,),
    )
    t = cur.fetchone()[0]
    conn.close()
    print(f"OK user={USER} reagents={r} consumables={c} projects={p} gantt_tasks={t}")


if __name__ == "__main__":
    main()
