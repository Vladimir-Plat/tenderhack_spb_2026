import json
import sys
import re
from difflib import SequenceMatcher
from statistics import mean, median
from typing import Any, Dict, List, Literal, Tuple, Optional

SourceName = str
AggregationMode = Literal["mean", "sum", "median"]

DEFAULT_SOURCES = ["ozone", "wb", "yandex", "other"]
DEFAULT_QUERY = "белая меховая зимняя куртка"

# Веса для специальных параметров
WEIGHT_PRICE = 0.2
WEIGHT_DISTANCE = 0.2
WEIGHT_TEXT_SIMILARITY = 0.2
WEIGHT_OTHER_PARAMS_TOTAL = 0.4


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("JSON должен содержать список словарей")
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Каждый элемент JSON должен быть словарём")
        if "источник" not in item:
            raise ValueError(f"У словаря нет ключа 'источник': {item}")
    return data


def extract_keywords(query: str) -> List[str]:
    words = re.findall(r'[а-яёa-z]+', query.lower())
    return [w for w in words if len(w) > 2]


def calculate_price_score(price: Any, min_price: float, max_price: float) -> float:
    """
    ⭐ ЧЕМ МЕНЬШЕ ЦЕНА → ТЕМ ВЫШЕ ОЦЕНКА ⭐
    
    Формула: score = 1 - (цена - мин_цена) / (макс_цена - мин_цена)
    
    Пример:
    - Самая дешёвая цена (мин_цена) → score = 1.0
    - Самая дорогая цена (макс_цена) → score = 0.0
    - Цена посередине → score = 0.5
    """
    if price is None:
        return 0.5
    
    try:
        price = float(price)
    except (ValueError, TypeError):
        return 0.5
    
    if min_price == max_price:
        return 1.0 if price <= min_price else 0.0
    
    # Обратная нормализация: чем меньше цена, тем выше оценка
    score = 1.0 - (price - min_price) / (max_price - min_price)
    return max(0.0, min(1.0, score))


def calculate_distance_score(distance: Any, min_dist: float, max_dist: float) -> float:
    """
    ⭐ ЧЕМ МЕНЬШЕ ДАЛЬНОСТЬ → ТЕМ ВЫШЕ ОЦЕНКА ⭐
    
    Формула: score = 1 - (расстояние - мин_расст) / (макс_расст - мин_расст)
    
    Пример:
    - Самое близкое расстояние (мин_расст) → score = 1.0
    - Самое дальнее расстояние (макс_расст) → score = 0.0
    - Расстояние посередине → score = 0.5
    """
    if distance is None:
        return 0.5
    
    try:
        distance = float(distance)
    except (ValueError, TypeError):
        return 0.5
    
    if min_dist == max_dist:
        return 1.0 if distance <= min_dist else 0.0
    
    # Обратная нормализация: чем меньше расстояние, тем выше оценка
    score = 1.0 - (distance - min_dist) / (max_dist - min_dist)
    return max(0.0, min(1.0, score))


def calculate_text_similarity(item_text: str, query: str) -> float:
    """Рассчитывает похожесть текста товара на запрос"""
    if not item_text or not query:
        return 0.0
    return SequenceMatcher(None, item_text.lower(), query.lower()).ratio()


def extract_param_value(item: Dict[str, Any], keyword: str) -> bool:
    """Проверяет, присутствует ли ключевое слово в объекте"""
    keyword_lower = keyword.lower()
    for key, value in item.items():
        if key == "источник" or key == "цена" or key == "дальность":
            continue
        if value is None:
            continue
        if keyword_lower in str(value).lower():
            return True
        if keyword_lower in key.lower():
            if keyword_lower in ["зим", "зимн"] and "зим" in str(value).lower():
                return True
            if keyword_lower in ["мех", "мехов"] and "мех" in str(value).lower():
                return True
    return False


def calculate_keyword_match_score(item: Dict[str, Any], keywords: List[str], weights: Dict[str, float]) -> float:
    """Рассчитывает взвешенную оценку совпадения с ключевыми словами"""
    total_score = 0.0
    for keyword in keywords:
        if keyword in weights:
            match = extract_param_value(item, keyword)
            total_score += weights[keyword] * (1.0 if match else 0.0)
    return total_score


def dict_to_text(item: Dict[str, Any]) -> str:
    """Преобразует словарь в строку для сравнения (исключая цену и дальность)"""
    parts = []
    exclude_keys = {"источник", "цена", "дальность"}
    for key, value in item.items():
        if key in exclude_keys:
            continue
        if value is None:
            continue
        parts.append(str(value))
    return " ".join(parts).lower().strip()


def group_by_source(items: List[Dict[str, Any]]) -> Dict[SourceName, List[Dict[str, Any]]]:
    grouped: Dict[SourceName, List[Dict[str, Any]]] = {}
    for original_index, item in enumerate(items):
        source = item["источник"]
        enriched_item = dict(item)
        enriched_item["_original_index"] = original_index
        enriched_item["_text_for_similarity"] = dict_to_text(item)
        enriched_item["_scores"] = []
        enriched_item["_debug"] = {}
        enriched_item["_relevance_score"] = 0.0
        grouped.setdefault(source, []).append(enriched_item)
    return grouped


def get_keyword_weights(keywords: List[str]) -> Dict[str, float]:
    """Распределяет вес 0.4 между ключевыми словами поровну"""
    if not keywords:
        return {}
    weight_per_keyword = WEIGHT_OTHER_PARAMS_TOTAL / len(keywords)
    return {kw: weight_per_keyword for kw in keywords}


def calculate_comprehensive_score(
    item: Dict[str, Any],
    keywords: List[str],
    keyword_weights: Dict[str, float],
    price_stats: Tuple[float, float],
    distance_stats: Tuple[float, float],
    query: str
) -> float:
    """Вычисляет комплексную оценку с учётом: цена(меньше→лучше), дальность(меньше→лучше)"""
    min_price, max_price = price_stats
    min_dist, max_dist = distance_stats
    
    # 1. Оценка по цене (ЧЕМ МЕНЬШЕ → ТЕМ ЛУЧШЕ)
    price_score = calculate_price_score(item.get("цена"), min_price, max_price)
    
    # 2. Оценка по дальности (ЧЕМ МЕНЬШЕ → ТЕМ ЛУЧШЕ)
    distance_score = calculate_distance_score(item.get("дальность"), min_dist, max_dist)
    
    # 3. Оценка текстовой похожести
    item_text = dict_to_text(item)
    text_similarity = calculate_text_similarity(item_text, query)
    
    # 4. Оценка совпадения ключевых слов
    keyword_score = calculate_keyword_match_score(item, keywords, keyword_weights)
    
    # Итоговая оценка
    total_score = (
        WEIGHT_PRICE * price_score +
        WEIGHT_DISTANCE * distance_score +
        WEIGHT_TEXT_SIMILARITY * text_similarity +
        keyword_score
    )
    
    # Отладочная информация
    item["_debug"]["price"] = item.get("цена")
    item["_debug"]["price_score"] = price_score
    item["_debug"]["distance"] = item.get("дальность")
    item["_debug"]["distance_score"] = distance_score
    item["_debug"]["text_similarity"] = text_similarity
    item["_debug"]["keyword_score"] = keyword_score
    item["_debug"]["total_score"] = total_score
    
    return total_score


def run_relevance_ranking(grouped: Dict[SourceName, List[Dict[str, Any]]], query: str = DEFAULT_QUERY) -> None:
    """Ранжирует объекты по релевантности"""
    keywords = extract_keywords(query)
    print(f"\n🔍 Ключевые слова запроса: {keywords}")
    
    keyword_weights = get_keyword_weights(keywords)
    print(f"📊 Веса ключевых слов: {keyword_weights}")
    print(f"📊 Вес цены (чем меньше → тем лучше): {WEIGHT_PRICE}")
    print(f"📊 Вес дальности (чем меньше → тем лучше): {WEIGHT_DISTANCE}")
    print(f"📊 Вес текстовой похожести: {WEIGHT_TEXT_SIMILARITY}")
    
    # Собираем статистику для нормализации
    all_prices = []
    all_distances = []
    
    for source, items in grouped.items():
        for item in items:
            price = item.get("цена")
            if price is not None and isinstance(price, (int, float)):
                all_prices.append(float(price))
            
            distance = item.get("дальность")
            if distance is not None and isinstance(distance, (int, float)):
                all_distances.append(float(distance))
    
    min_price = min(all_prices) if all_prices else 0
    max_price = max(all_prices) if all_prices else 1
    min_dist = min(all_distances) if all_distances else 0
    max_dist = max(all_distances) if all_distances else 1
    
    print(f"\n💰 Цены: от {min_price:,} до {max_price:,} руб.")
    print(f"   (чем меньше цена → тем выше оценка)")
    print(f"📍 Дальность: от {min_dist} до {max_dist} км")
    print(f"   (чем меньше дальность → тем выше оценка)")
    
    # Вычисляем оценки
    for source, items in grouped.items():
        for item in items:
            score = calculate_comprehensive_score(
                item, keywords, keyword_weights,
                (min_price, max_price), (min_dist, max_dist), query
            )
            item["_relevance_score"] = score


def priority_score(index_in_source: int, total_in_source: int) -> float:
    if total_in_source <= 1:
        return 1.0
    return 1.0 - (index_in_source / (total_in_source - 1))


def string_similarity_simple(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def run_cross_validation_iteration(reference_source: SourceName, grouped: Dict[SourceName, List[Dict[str, Any]]]) -> None:
    reference_items = grouped.get(reference_source, [])
    if not reference_items:
        return
    
    for source, items in grouped.items():
        if not items:
            continue
        
        if source == reference_source:
            total = len(items)
            for index, item in enumerate(items):
                score = priority_score(index, total)
                item["_scores"].append(score)
            continue
        
        for item in items:
            similarities = [string_similarity_simple(item["_text_for_similarity"], ref_item["_text_for_similarity"]) 
                          for ref_item in reference_items]
            best_score = max(similarities) if similarities else 0.0
            item["_scores"].append(best_score)


def run_full_cross_validation(grouped: Dict[SourceName, List[Dict[str, Any]]], sources: List[SourceName] = None) -> None:
    if sources is None:
        sources = list(grouped.keys())
    for source in sources:
        if source in grouped:
            run_cross_validation_iteration(source, grouped)


def aggregate_scores(grouped: Dict[SourceName, List[Dict[str, Any]]], mode: AggregationMode = "mean") -> Dict[SourceName, List[Tuple[int, float]]]:
    results = {}
    for source, items in grouped.items():
        source_results = []
        for item in items:
            scores = item["_scores"]
            if not scores:
                agg_score = 0.0
            elif mode == "mean":
                agg_score = mean(scores)
            elif mode == "median":
                agg_score = median(scores)
            else:
                agg_score = sum(scores)
            source_results.append((item["_original_index"], agg_score))
            item["_cv_aggregated_score"] = agg_score
        results[source] = source_results
    return results


def find_duplicates_by_threshold(grouped: Dict[SourceName, List[Dict[str, Any]]], threshold: float = 0.8) -> Dict[str, List[Dict[str, Any]]]:
    duplicates = []
    sources = list(grouped.keys())
    for i, source1 in enumerate(sources):
        for source2 in sources[i+1:]:
            for item1 in grouped[source1]:
                for item2 in grouped[source2]:
                    similarity = string_similarity_simple(item1["_text_for_similarity"], item2["_text_for_similarity"])
                    if similarity >= threshold:
                        duplicates.append({
                            "source1": source1,
                            "source2": source2,
                            "similarity": similarity,
                            "item1": {k: v for k, v in item1.items() if not k.startswith("_")},
                            "item2": {k: v for k, v in item2.items() if not k.startswith("_")}
                        })
    return {"duplicates": duplicates}


def sort_by_relevance(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Сортирует по релевантности (чем выше оценка, тем выше в списке)"""
    return sorted(items, key=lambda x: x.get("_relevance_score", 0), reverse=True)


def print_statistics(grouped: Dict[SourceName, List[Dict[str, Any]]]):
    print("\n" + "=" * 70)
    print("СТАТИСТИКА ПО ДАННЫМ")
    print("=" * 70)
    
    for source, items in grouped.items():
        print(f"\n📦 Источник: {source.upper()}")
        print(f"   Количество: {len(items)}")
        
        if items:
            scores = [item.get("_relevance_score", 0) for item in items]
            print(f"   🎯 Средняя релевантность: {mean(scores):.3f}")
            print(f"   📈 Макс релевантность: {max(scores):.3f}")
            print(f"   📉 Мин релевантность: {min(scores):.3f}")
            
            # Демонстрация работы цены и дальности
            cheapest = min(items, key=lambda x: x.get("цена", float('inf')))
            closest = min(items, key=lambda x: x.get("дальность", float('inf')))
            print(f"   💰 Самый дешёвый: {cheapest.get('Артикул', 'N/A')} ({cheapest.get('цена', 'N/A')} руб.)")
            print(f"   📍 Самый близкий: {closest.get('Артикул', 'N/A')} ({closest.get('дальность', 'N/A')} км)")


def main(query: Optional[str] = None):
    if query is None:
        query = DEFAULT_QUERY
    
    print("=" * 70)
    print("ВАЛИДАТОР JSON С РЕЛЕВАНТНОСТЬЮ К ЗАПРОСУ")
    print("=" * 70)
    print(f"\n🔎 Поисковый запрос: \"{query}\"")
    print("\n⭐ Правила оценки:")
    print("   1. Цена: ЧЕМ МЕНЬШЕ → ТЕМ ВЫШЕ ОЦЕНКА")
    print("   2. Дальность: ЧЕМ МЕНЬШЕ → ТЕМ ВЫШЕ ОЦЕНКА")
    print("   3. Текст: ЧЕМ БОЛЬШЕ ПОХОЖ НА ЗАПРОС → ТЕМ ВЫШЕ ОЦЕНКА")
    print("   4. Ключевые слова: ЧЕМ БОЛЬШЕ СОВПАДЕНИЙ → ТЕМ ВЫШЕ ОЦЕНКА")
    
    try:
        data = load_json("products.json")
        print(f"\n✅ Загружено {len(data)} товаров")
    except FileNotFoundError:
        print("\n❌ Файл products.json не найден!")
        print("   Сначала запустите generate_products.py")
        return
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        return
    
    grouped = group_by_source(data)
    print("\n📊 Распределение по источникам:")
    for source, items in grouped.items():
        print(f"   {source}: {len(items)} товаров")
    
    run_relevance_ranking(grouped, query)
    run_full_cross_validation(grouped)
    aggregate_scores(grouped, mode="mean")
    duplicates = find_duplicates_by_threshold(grouped, threshold=0.8)
    
    print_statistics(grouped)
    
    if duplicates['duplicates']:
        print(f"\n🔍 Найдено дубликатов: {len(duplicates['duplicates'])}")
    
    # Сортировка по релевантности
    all_items = []
    for source, items in grouped.items():
        all_items.extend(items)
    
    sorted_items = sort_by_relevance(all_items)
    
    # Очистка и сохранение
    cleaned_items = []
    for item in sorted_items:
        cleaned = {k: v for k, v in item.items() if not k.startswith("_")}
        cleaned["_рейтинг_релевантности"] = item.get("_relevance_score", 0)
        cleaned_items.append(cleaned)
    
    with open("products_sorted_by_relevance.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_items, f, ensure_ascii=False, indent=2)
    
    # Топ-10
    print("\n" + "=" * 70)
    print("🏆 ТОП-10 САМЫХ РЕЛЕВАНТНЫХ ТОВАРОВ")
    print("=" * 70)
    for i, item in enumerate(cleaned_items[:10]):
        print(f"\n{i+1}. {item.get('Артикул', 'N/A')} | {item.get('источник', 'N/A').upper()}")
        print(f"   📝 {item.get('Тип', 'N/A')} | {item.get('Бренд в одежде и обуви', 'N/A')}")
        print(f"   💰 Цена: {item.get('цена', 'N/A')} руб. | 📍 Дальность: {item.get('дальность', 'N/A')} км")
        print(f"   🎯 Оценка: {item.get('_рейтинг_релевантности', 0):.4f}")
    
    print("\n✨ Готово! Результаты сохранены в products_sorted_by_relevance.json")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(" ".join(sys.argv[1:]))
    else:
        main()