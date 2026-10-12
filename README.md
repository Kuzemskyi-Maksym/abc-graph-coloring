# abc-graph-coloring

Бджолині алгоритми (ABC) для задачі розфарбування графа: порівняння існуючих модифікацій і розробка власної. Наукова робота, керівник: М. Головченко (КПІ ім. Ігоря Сікорського, ФІОТ).

**Задача:** розфарбувати вершини неорієнтованого графа так, щоб суміжні мали різні кольори, і мінімізувати кількість кольорів.

## Запуск
Python 3.10+, без сторонніх залежностей.

```bash
python run.py --time 3 --seeds 5 --out results/run.csv data/*.col
```

Параметри: `--time` (секунд на запуск), `--seeds` (кількість запусків), `--algs` (через кому, за замовчуванням усі), `--procs` (процеси).

## Структура
- `abc_gc/graph.py`: читання DIMACS `.col`, перевірка правильності розфарбування
- `abc_gc/heuristics.py`: greedy first-fit, DSATUR, RLF
- `abc_gc/bees.py`: алгоритми та словник `ALGORITHMS`
  - `vertex_bee`: Chahkandi & Mirzaei (з лекції): вершина = джерело їжі, нектар = степінь
  - `site_bee`: версія з лекції: ділянка = ціле розфарбування, фуражири спорожнюють останній колір
  - `habc`: Fister Jr., Fister, Brest (2012), адаптовано на мінімізацію k
  - `tabucol`: TabuCol, не-бджолиний еталон
  - `own_abc`: чернетка власної модифікації + абляції
- `run.py`: однаковий бюджет часу на кожен запуск, результати в CSV
- `data/`: 18 графів DIMACS (копія з [dynaroars/npbench](https://github.com/dynaroars/npbench))
- `results/`: сирі результати прогонів

## Відхилення від оригінальних статей
- **HABC:** декодер first-fit за порядком ваг (у статті DSatur); мета мінімізувати k замість фіксованого 3-розфарбування; ймовірність спостерігачів рахується за якістю (у статті формула (5) рахує за кількістю конфліктів).
- **site_bee:** «заміна кольору вершини з суміжною» трактується так: вершина з останнього кольору бере колір c сусіда u, якщо u її єдиний сусід кольору c, а u переходить в інший нижчий колір.

## Статус
Перший прогін (2026-10-12): 3 с на запуск, 5 запусків, попередні результати. Чернетка `own_abc` поки не краща за TabuCol, і бджолина структура в ній нічого не додає. Власна модифікація ще в роботі.

## Джерела
- Fister Jr., I., Fister, I., Brest, J. A Hybrid Artificial Bee Colony Algorithm for Graph 3-Coloring. LNCS 7269, 2012. [arXiv:1206.1012](https://arxiv.org/abs/1206.1012)
- Consoli, P., Collera, A., Pavone, M. Swarm Intelligence heuristics for Graph Coloring Problem. IEEE CEC 2013.
- Dorrigiv, M., Markib, H.Y. Algorithms for the graph coloring problem based on swarm intelligence. AISP 2012.
- Chahkandi, V., Mirzaei, O. Vertex Coloring Based on Artificial Bee Colony Algorithm (за лекцією «Метаевристичний підхід»).
- Brélaz, D. New methods to color the vertices of a graph. CACM, 1979 (DSATUR).
- Hertz, A., de Werra, D. Using tabu search techniques for graph coloring. Computing, 1987 (TabuCol).
