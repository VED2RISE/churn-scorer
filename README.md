# churn-scorer

Небольшой проект команды роста: по журналу событий продукта считаем признаки пользователей
на дату среза (cutoff) и обучаем модель, которая оценивает риск ухода (churn) в ближайшие 30 дней.
Скоры уходят в CRM: команда удержания связывается с пользователями из верхушки списка.

## Как устроено

```
events.csv ──► load_events ──► build_features(cutoff) ──► train / predict ──► метрики, скоры
labels.csv ──► load_labels ──────────────────────────────┘
```

| Модуль | Назначение |
| --- | --- |
| `churn/data.py` | чтение и валидация `events` / `labels` |
| `churn/features.py` | признаки пользователя на момент `cutoff` (окно 30 дней, давность последней сессии, стаж) |
| `churn/model.py` | логистическая регрессия поверх `StandardScaler`, сохранение через joblib |
| `churn/report.py` | текстовые отчёты для консоли |
| `churn/cli.py` | команды `train`, `evaluate`, `score` |
| `churn/config.py` | пути, даты срезов, константы |

## Данные

`data/*_events.csv` — журнал событий, одна строка на событие:

| колонка | описание |
| --- | --- |
| `event_id` | идентификатор события |
| `user_id` | пользователь |
| `ts` | время события, UTC, ISO-8601 |
| `event_type` | `session`, `purchase` или `support_ticket` |
| `amount` | сумма покупки (только для `purchase`) |

`data/*_labels.csv` — `user_id`, `churned`. **churned = 1**, если у пользователя не было ни одной
сессии в течение 30 дней после даты среза.

Два набора:

- **train** — полная выгрузка таблицы событий из аналитической БД, сделана 2025-07-01.
  Дата среза `2025-06-01`, метки посчитаны по активности 2025-06-01 … 2025-07-01.
- **holdout** — пилот на новой когорте. Снапшот событий сделан в день среза `2025-08-01`,
  метки собраны 2025-09-01.

## Окружение

Все зависимости уже стоят в подготовленном окружении `candidate-python-environment/.venv`
(pandas, numpy, scikit-learn, joblib, pytest, mypy). Ничего доустанавливать не нужно.

macOS / Linux:

```bash
source /путь/к/candidate-python-environment/.venv/bin/activate
```

Windows PowerShell:

```powershell
& "C:\путь\к\candidate-python-environment\.venv\Scripts\Activate.ps1"
```

## Команды

Запускать из корня проекта (папка с `pyproject.toml`):

```bash
python -m pytest -q          # тесты
python -m mypy churn         # типы
python -m churn train        # обучить на train, сохранить models/churn_model.joblib
python -m churn evaluate     # оценить сохранённую модель на holdout
python -m churn score        # посчитать скоры для пользователей holdout -> scores.csv
python -m churn --help
```

У всех команд есть флаги `--events`, `--labels`, `--cutoff YYYY-MM-DD`, `--model` — см. `--help`.
